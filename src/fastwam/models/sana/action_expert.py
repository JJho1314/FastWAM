"""Action expert for the SANA-Video FastWAM.

A compact flow-matching DiT over an action chunk. Unlike the Wan22 MoT path
(which mixes video/action tokens with shared joint attention), this expert is
self-contained and reads the video world model through **cross-attention**:
each block does action self-attention, then cross-attends to (a) the video
DiT's token features and (b) the text/proprio context, then an MLP. All
sub-layers are AdaLN-modulated by the (action) diffusion timestep.

This module has no SANA/Wan dependencies and can be unit-tested in isolation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding. `t` is a 1D tensor of shape (B,)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-head attention supporting self- and cross-attention via SDPA."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx=None, key_padding_mask=None):
        # x: (B, Nq, D); ctx: (B, Nk, D); key_padding_mask: (B, Nk) bool, True == keep
        if ctx is None:
            ctx = x
        B, Nq, D = x.shape
        Nk = ctx.shape[1]
        q = self.q(x).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(ctx).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(ctx).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, Nk, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return self.proj(out)


class ActionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, num_heads)
        self.norm_vid = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn_vid = Attention(dim, num_heads)
        self.norm_txt = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn_txt = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
        # AdaLN: shift/scale/gate for self-attn and mlp
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, t_emb, video_mem, text_mem, text_mask=None):
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.ada(t_emb).chunk(6, dim=-1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_sa, scale_sa))
        x = x + self.cross_attn_vid(self.norm_vid(x), video_mem)
        x = x + self.cross_attn_txt(self.norm_txt(x), text_mem, key_padding_mask=text_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class SanaActionExpert(nn.Module):
    """Flow-matching action DiT, cross-attending to a video world model.

    Args:
        action_dim: per-step action dimension.
        hidden_size: transformer width.
        depth: number of blocks.
        num_heads: attention heads.
        video_feat_dim: dim of the video DiT token features it cross-attends to.
        text_dim: dim of the (gemma) text context.
        max_action_len: max action-chunk length (for the learned positional table).
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int = 1024,
        depth: int = 12,
        num_heads: int = 8,
        video_feat_dim: int = 2240,
        text_dim: int = 2304,
        mlp_ratio: float = 4.0,
        max_action_len: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_in = nn.Linear(action_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_action_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.video_proj = nn.Linear(video_feat_dim, hidden_size)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [ActionBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        self.action_out = nn.Linear(hidden_size, action_dim)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, noisy_action, timestep, video_feats, text_context, text_mask=None):
        """
        noisy_action: (B, T, action_dim)
        timestep:     (B,) flow-matching timestep
        video_feats:  (B, Nv, video_feat_dim) video DiT token features
        text_context: (B, L, text_dim)
        text_mask:    (B, L) bool, True == keep
        returns velocity prediction (B, T, action_dim)
        """
        B, T, _ = noisy_action.shape
        if T > self.pos_embed.shape[1]:
            raise ValueError(f"action chunk len {T} exceeds max_action_len {self.pos_embed.shape[1]}")
        x = self.action_in(noisy_action) + self.pos_embed[:, :T]
        t_emb = self.t_embed(timestep_embedding(timestep, self.hidden_size).to(x.dtype))
        vid = self.video_proj(video_feats.to(x.dtype))
        txt = self.text_proj(text_context.to(x.dtype))
        for blk in self.blocks:
            x = blk(x, t_emb, vid, txt, text_mask)
        shift, scale = self.ada_out(t_emb).chunk(2, dim=-1)
        x = modulate(self.norm_out(x), shift, scale)
        return self.action_out(x)
