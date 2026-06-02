"""Discrete flow-matching scheduler for text token sequences.

Implements the same uniform-source mixture path used in Uni-ViGU
(MixtureDiscreteProbPath with a linear convex scheduler). Kept self-contained
so this repo does not need the upstream `flow_matching` package.

Path:
    For each token position, with probability `t` keep the target token
    (ground truth), with probability `(1-t)` replace it with a sample from
    the source distribution (uniform over [0, vocab_size)).
    => x_t = sample where Bernoulli(t) ? target : uniform.

Training:
    Model receives `x_t` plus timestep `t` (mapped to Wan-style timestep
    [0, num_train_timesteps] for compatibility with MoT modulation).
    Loss is token-wise cross-entropy against the target tokens.

This mirrors the way Uni-ViGU's WanUnifiedTransformer expects `time_t` in
the same numerical range as `time_v` so the shared MoT modulation works.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class WanDiscreteFlowMatchScheduler:
    def __init__(
        self,
        vocab_size: int,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        mask_token_id: Optional[int] = None,
        eps: float = 1e-10,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(f"`vocab_size` must be positive, got {vocab_size}")
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.vocab_size = int(vocab_size)
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.mask_token_id = None if mask_token_id is None else int(mask_token_id)
        self.eps = float(eps)

    # ------------------------------------------------------------------
    # time / sigma helpers (kept consistent with WanContinuousFlowMatchScheduler)
    # ------------------------------------------------------------------

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample timestep in [0, num_train_timesteps] for training.

        Matches the convention used by `WanContinuousFlowMatchScheduler` so
        the same MoT time modulation pipeline (sinusoidal -> linear)
        consumes both video & text timesteps identically.
        """
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Constant weighting (=1). Discrete CE already has its own scale."""
        return torch.ones_like(timestep, dtype=torch.float32)

    # ------------------------------------------------------------------
    # forward path (build noisy token sequence)
    # ------------------------------------------------------------------

    def add_noise(
        self,
        target_tokens: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mixture path: per-token Bernoulli(sigma) keeps target, else samples source.

        Args:
            target_tokens: [B, L] int64 ground-truth tokens.
            timestep:      [B] in [0, num_train_timesteps]. sigma = t / T in [0, 1]
                           is the *noise* level (sigma=1 => fully noisy).
                           So keep-target probability is (1 - sigma).
            attention_mask: [B, L] bool, True for valid positions; padded positions
                            are left unchanged (kept as target / pad token).
        Returns:
            noisy_tokens: [B, L] int64.
        """
        if target_tokens.ndim != 2:
            raise ValueError(
                f"`target_tokens` must be 2D [B, L], got shape {tuple(target_tokens.shape)}"
            )
        if timestep.ndim != 1 or timestep.shape[0] != target_tokens.shape[0]:
            raise ValueError(
                f"`timestep` must be 1D [B={target_tokens.shape[0]}], got {tuple(timestep.shape)}"
            )

        device = target_tokens.device
        batch_size, seq_len = target_tokens.shape
        sigma = (timestep.to(dtype=torch.float32) / float(self.num_train_timesteps)).clamp(0.0, 1.0)
        sigma = sigma.view(batch_size, 1).to(device)

        # Bernoulli(sigma) decides which positions are corrupted.
        corrupt = torch.rand((batch_size, seq_len), device=device, dtype=torch.float32) < sigma

        if self.mask_token_id is None:
            source = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(batch_size, seq_len),
                device=device,
                dtype=target_tokens.dtype,
            )
        else:
            source = torch.full(
                (batch_size, seq_len),
                fill_value=int(self.mask_token_id),
                device=device,
                dtype=target_tokens.dtype,
            )

        noisy = torch.where(corrupt, source, target_tokens)
        if attention_mask is not None:
            if attention_mask.shape != target_tokens.shape:
                raise ValueError(
                    "`attention_mask` shape must match `target_tokens`, "
                    f"got {tuple(attention_mask.shape)} vs {tuple(target_tokens.shape)}"
                )
            noisy = torch.where(attention_mask.to(torch.bool), noisy, target_tokens)
        return noisy

    # ------------------------------------------------------------------
    # loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cross_entropy_loss(
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Per-token CE, optionally masking padded positions.

        Args:
            logits: [B, L, V] float.
            target_tokens: [B, L] int64.
            attention_mask: [B, L] bool, True = include in loss.
            reduction: "mean", "sum", or "none" (returns [B, L]).
        Returns: scalar (mean/sum) or [B, L] tensor.
        """
        if logits.ndim != 3:
            raise ValueError(f"`logits` must be 3D [B, L, V], got {tuple(logits.shape)}")
        if target_tokens.ndim != 2 or target_tokens.shape != logits.shape[:2]:
            raise ValueError(
                f"`target_tokens` shape mismatch: got {tuple(target_tokens.shape)} vs "
                f"logits {tuple(logits.shape)}"
            )

        ce_token = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            target_tokens.reshape(-1).long(),
            reduction="none",
        ).view(target_tokens.shape)

        if reduction == "none":
            return ce_token

        if attention_mask is not None:
            mask = attention_mask.to(device=ce_token.device, dtype=ce_token.dtype)
            ce_token = ce_token * mask
            denom = mask.sum().clamp(min=1.0)
        else:
            denom = float(ce_token.numel())

        if reduction == "mean":
            return ce_token.sum() / denom
        if reduction == "sum":
            return ce_token.sum()
        raise ValueError(f"Unknown reduction: {reduction}")

    # ------------------------------------------------------------------
    # inference (kept minimal; FastWAM doesn't decode text at inference today,
    # but we provide a simple "iterative refinement" loop for completeness)
    # ------------------------------------------------------------------

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)
