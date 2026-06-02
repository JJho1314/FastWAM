"""TextDiT expert for FastWAM trimodal (Uni-ViGU-style).

The text expert mirrors the structure of `ActionDiT` so it plugs straight
into the existing `MoT` framework (shared mixed self-attention across
experts). The substitution from action to text is:

  - Input encoder:  Linear(action_dim, hidden_dim)  ->  Embedding(vocab_size, hidden_dim)
  - Output head:    Linear(hidden_dim, action_dim)  ->  ModulatedHead -> vocab_size logits
  - Positional emb: 1D RoPE (shared `precompute_freqs_cis`, same as action)

Constraints (enforced):
  - num_heads / attn_head_dim must match the video and action experts.
  - num_layers must match the video / action experts.

Token output is `[B, L, vocab_size]` logits suitable for cross-entropy
against ground-truth tokens (see `WanDiscreteFlowMatchScheduler`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)

logger = get_logger(__name__)


class TextHead(nn.Module):
    """Modulated projection: hidden -> vocab logits.

    Mirrors `ActionHead` style (LayerNorm + AdaLN modulation by time
    embedding) but the output dim is the vocab size instead of action_dim.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (
            self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)
        ).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.proj(x)


class TextDiT(nn.Module):
    """Discrete-flow text expert for FastWAM trimodal.

    Forward signature is intentionally compatible with `ActionDiT` so it
    can be plugged into the same MoT.forward(...) iteration loop.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        max_seq_len: int = 1024,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")
        if vocab_size <= 0:
            raise ValueError(f"`vocab_size` must be > 0, got {vocab_size}")
        if max_seq_len <= 0:
            raise ValueError(f"`max_seq_len` must be > 0, got {max_seq_len}")

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.max_seq_len = max_seq_len

        # Token embedding (uses 0 as pad; padding_idx avoids polluting gradients).
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        # Conditioning embeddings (T5 context + Wan-style timestep).
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = TextHead(hidden_dim=hidden_dim, vocab_size=vocab_size, eps=eps)
        # 1D RoPE positions.
        self.freqs = precompute_freqs_cis(attn_head_dim, end=max_seq_len)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    # ------------------------------------------------------------------
    # pre/post split for MoT
    # ------------------------------------------------------------------

    def pre_dit(
        self,
        text_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if text_tokens.ndim != 2:
            raise ValueError(
                f"`text_tokens` must be 2D [B, L], got shape {tuple(text_tokens.shape)}"
            )
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got {tuple(context.shape)}")
        batch_size, seq_len = text_tokens.shape
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between text tokens and context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D, got shape {tuple(timestep.shape)}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Text token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(
                    f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}"
                )
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got "
                    f"{tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.token_embedding(text_tokens.long())
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """Project hidden states -> vocab logits using the time embedding `t`."""
        t = pre_state["t"]
        return self.head(tokens, t)

    # ------------------------------------------------------------------
    # standalone forward (kept for symmetry with ActionDiT; not used by MoT)
    # ------------------------------------------------------------------

    def forward(
        self,
        text_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            text_tokens=text_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        ctx_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=ctx_mask,
                )
            else:
                x = block(x, context_emb, t_mod, freqs, context_mask=ctx_mask)

        return self.post_dit(x, pre_state)
