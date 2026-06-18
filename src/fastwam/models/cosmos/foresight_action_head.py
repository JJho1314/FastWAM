"""Standalone DiT action head for the AGRA ("Making Foresight Actionable",
arXiv 2606.12217) world<->action coupling.

This is the ACTION side of the AGRA cross-attention architecture. It is a small,
self-contained DiT (NOT a Cosmos block) so it stays **pure torch** — importable
and shape-testable WITHOUT the cosmos_predict2 env. The world model (Cosmos video
DiT) is run separately and its multi-layer "foresight" hidden states are fed in
here as per-layer cross-attention contexts.

Architecture (paper, cross-attn part):
  - ``num_layers`` (=8) AdaLN-zero DiT blocks. Each block is
        self-attn(AdaLN) -> cross-attn(AdaLN, to contexts[j]) -> MLP(AdaLN)
    with zero-init output gates (AdaLN-zero / DiT convention), exactly the same
    residual+modulation layout as the Cosmos ``Block`` (minimal_v4_dit.py:1257),
    but cross-attention reads the per-layer foresight ``contexts[j]`` (the video
    layer ``ell_j`` hidden, already projected to ``crossattn_dim``) instead of text.
  - Input tokens = the K-step noisy action chunk embedded ``Linear(action_dim->hidden)``,
    with the robot proprio state ``s0`` embedded ``Linear(proprio_dim->hidden)`` and
    PREPENDED as one extra token. Readout drops the proprio token.
  - The action noise level ``tau_a`` is embedded (sinusoidal + 2-layer MLP) and
    conditions ALL AdaLN modulations (one timestep for the whole chunk, so the
    modulation is [B, hidden] broadcast over tokens — same as Cosmos' T=1 case).
  - Flow-matching velocity readout: final AdaLN + ``Linear(hidden->action_dim)``
    (zero-init) gives per-action-step velocity ``[B, K, action_dim]``.

Positional scheme on the action self-attention: NONE by default (a short K-step
chunk + one proprio token; learned/none is fine and keeps this fully standalone).
RoPE is intentionally NOT wired here to avoid coupling to Cosmos' RoPE; if a
sequence-length-sensitive scheme is later wanted, add it inside ``_SelfAttention``
(its q/k are [B, S, H, Dh] before SDPA — the natural RoPE application point).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# timestep embedding (standard sinusoidal + 2-layer MLP)                       #
# --------------------------------------------------------------------------- #
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal embedding of a [B] timestep tensor -> [B, dim].

    Mirrors the DiT/diffusion convention (half cos, half sin). ``t`` may be a
    float flow-matching noise level in [0, 1] (it is just used as a scalar).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad odd dims
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding -> 2-layer MLP -> [B, hidden] conditioning vector."""

    def __init__(self, hidden: int, freq_dim: int | None = None):
        super().__init__()
        self.freq_dim = int(freq_dim or hidden)
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        freqs = sinusoidal_timestep_embedding(t, self.freq_dim).to(self.mlp[0].weight.dtype)
        return self.mlp(freqs)  # [B, hidden]


# --------------------------------------------------------------------------- #
# attention primitives (plain SDPA; multi-head)                               #
# --------------------------------------------------------------------------- #
class _SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]. NOTE: positional scheme is "none" — to add RoPE, rotate
        # q,k here (shape [B, H, S, Dh]) before SDPA.
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, S, Dh]
        out = F.scaled_dot_product_attention(q, k, v)    # [B, H, S, Dh]
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


class _CrossAttention(nn.Module):
    """Query = action tokens (dim ``q_dim``); Key/Value = foresight context
    (dim ``kv_dim`` == crossattn_dim). Q is projected ``q_dim->attn_dim``, out
    projected ``attn_dim->q_dim`` (so the action stream stays at ``q_dim``)."""

    def __init__(self, q_dim: int, kv_dim: int, attn_dim: int, num_heads: int):
        super().__init__()
        assert attn_dim % num_heads == 0, f"attn_dim {attn_dim} not divisible by num_heads {num_heads}"
        self.num_heads = int(num_heads)
        self.head_dim = attn_dim // num_heads
        self.to_q = nn.Linear(q_dim, attn_dim)
        self.to_k = nn.Linear(kv_dim, attn_dim)
        self.to_v = nn.Linear(kv_dim, attn_dim)
        self.proj = nn.Linear(attn_dim, q_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, S, q_dim]; context: [B, L, kv_dim]
        B, S, _ = x.shape
        L = context.shape[1]
        q = self.to_q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # [B, H, S, Dh]
        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        return self.proj(out)


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------- #
# AdaLN-zero DiT block: self-attn + cross-attn(foresight) + MLP               #
# --------------------------------------------------------------------------- #
def _modulate(norm_out: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # norm_out: [B, S, D]; shift/scale: [B, D] (one timestep, broadcast over S).
    return norm_out * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ForesightActionBlock(nn.Module):
    """One AdaLN-modulated DiT block. The single ``adaln`` head emits all nine
    modulations (shift/scale/gate x {self-attn, cross-attn, mlp}); the three gates
    are zero-init (AdaLN-zero) so the block starts as identity."""

    def __init__(self, hidden: int, num_heads: int, crossattn_dim: int, attn_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.self_attn = _SelfAttention(hidden, num_heads)
        self.norm_cross = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.cross_attn = _CrossAttention(hidden, crossattn_dim, attn_dim, num_heads)
        self.norm_mlp = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = _MLP(hidden, mlp_ratio)
        # AdaLN: timestep emb -> 9*hidden (shift/scale/gate for each of the 3 subblocks)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 9 * hidden))
        # zero-init the AdaLN head so all shift=scale=gate=0 at start (AdaLN-zero):
        # gates 0 -> block is identity; the residual stream passes through cleanly.
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, S, hidden]; t_emb: [B, hidden]; context: [B, L, crossattn_dim]
        mod = self.adaln(t_emb)  # [B, 9*hidden]
        (sh_sa, sc_sa, g_sa,
         sh_ca, sc_ca, g_ca,
         sh_mlp, sc_mlp, g_mlp) = mod.chunk(9, dim=-1)

        x = x + g_sa.unsqueeze(1) * self.self_attn(_modulate(self.norm_self(x), sh_sa, sc_sa))
        x = x + g_ca.unsqueeze(1) * self.cross_attn(_modulate(self.norm_cross(x), sh_ca, sc_ca), context)
        x = x + g_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm_mlp(x), sh_mlp, sc_mlp))
        return x


class ForesightActionHead(nn.Module):
    """8-layer cross-attention action DiT (AGRA, paper cross-attn part).

    forward(noisy_action [B,K,action_dim], t_a [B], proprio0 [B,proprio_dim],
            contexts: list[Tensor [B,Sv,crossattn_dim]] len=num_layers)
        -> velocity [B, K, action_dim]
    """

    def __init__(
        self,
        action_dim: int,
        proprio_dim: int,
        num_layers: int = 8,
        hidden: int = 1024,
        num_heads: int = 32,
        crossattn_dim: int = 2048,
        action_horizon: int | None = None,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.num_layers = int(num_layers)
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.crossattn_dim = int(crossattn_dim)
        self.action_horizon = None if action_horizon is None else int(action_horizon)

        # input embeds
        self.action_encoder = nn.Linear(self.action_dim, self.hidden)
        self.proprio_encoder = nn.Linear(self.proprio_dim, self.hidden)
        # timestep embedder (sinusoidal + 2-layer MLP) for tau_a -> AdaLN conditioning
        self.t_embedder = TimestepEmbedder(self.hidden)

        # attn_dim == crossattn_dim per the paper (cross-attn dim 2048). The action
        # self-attn also runs at this internal head width so num_heads divides it.
        self.blocks = nn.ModuleList(
            ForesightActionBlock(
                hidden=self.hidden,
                num_heads=self.num_heads,
                crossattn_dim=self.crossattn_dim,
                attn_dim=self.crossattn_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(self.num_layers)
        )

        # final AdaLN + readout (zero-init: start predicting zero velocity, so the
        # flow-matching action loss begins at ~||target||^2 and trains up smoothly).
        self.final_norm = nn.LayerNorm(self.hidden, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden, 2 * self.hidden))
        nn.init.zeros_(self.final_adaln[1].weight)
        nn.init.zeros_(self.final_adaln[1].bias)
        self.head = nn.Linear(self.hidden, self.action_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def num_contexts(self) -> int:
        """#cross-attn contexts the AGRA bridge must supply (one per layer here)."""
        return self.num_layers

    def forward(self, noisy_action, t_a, proprio0, contexts):
        """See class docstring. ``contexts`` must have ``num_layers`` entries."""
        if len(contexts) != self.num_layers:
            raise ValueError(
                f"ForesightActionHead expects {self.num_layers} contexts, got {len(contexts)}"
            )
        B, K, _ = noisy_action.shape

        a_tok = self.action_encoder(noisy_action)            # [B, K, hidden]
        p_tok = self.proprio_encoder(proprio0).unsqueeze(1)  # [B, 1, hidden]
        x = torch.cat([p_tok, a_tok], dim=1)                 # [B, 1+K, hidden] (proprio prepended)

        t_emb = self.t_embedder(t_a)                         # [B, hidden]

        for blk, ctx in zip(self.blocks, contexts):
            x = blk(x, t_emb, ctx)

        # final AdaLN (shift/scale only; no gate at readout) + linear velocity head
        shift, scale = self.final_adaln(t_emb).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        vel = self.head(x)                                   # [B, 1+K, action_dim]
        return vel[:, 1:, :]                                 # drop the proprio token
