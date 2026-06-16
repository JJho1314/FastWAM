"""Base interface for a video<->action coupling strategy."""
from __future__ import annotations


class Coupling:
    """A coupling owns how the video DiT and action stream interact in the forward
    pass, and (via ``setup``) any coupling-specific submodules on the model.

    Methods receive the ``FastWAMCosmos`` instance as ``model`` so couplings stay
    stateless w.r.t. weights — all trainable params live on the model (so FSDP /
    checkpointing see them).
    """
    name = "base"

    def setup(self, model) -> None:
        """Create coupling-specific submodules on ``model`` (a FastWAMCosmos).

        Called once from ``FastWAMCosmos.__init__`` BEFORE accelerate.prepare, so
        any modules registered here are seen by FSDP and the optimizer.
        """

    def forward(self, model, noisy_latents, t_v, noisy_action, t_a, crossattn_emb):
        """Return (pred_v_latent [B,C,T,H,W], pred_a [B,Ta,action_dim])."""
        raise NotImplementedError
