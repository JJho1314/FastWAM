"""Video expert wrapper around a SANA-Video DiT (`SanaMSVideo`).

Wraps the SANA-Video linear-attention transformer and exposes
`forward_with_features`, which runs the DiT normally (returning the predicted
flow-matching velocity) while also capturing the token features of a chosen
transformer block via a forward hook. Those features are the cross-attention
memory consumed by the action expert.

The wrapper does not duplicate SANA's forward body, so it stays correct if the
upstream SANA implementation changes. Gradient checkpointing inside SANA
(`use_reentrant=False`) is compatible with the forward hook.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SanaVideoExpert(nn.Module):
    def __init__(self, dit: nn.Module, feature_layer: int = -1):
        """
        Args:
            dit: a constructed `SanaMSVideo` module (weights already loaded).
            feature_layer: index into `dit.blocks` whose output is used as the
                cross-attention memory for the action expert. Negative indexes
                from the end (default -1 = last block).
        """
        super().__init__()
        self.dit = dit
        blocks = self.dit.blocks
        idx = feature_layer if feature_layer >= 0 else len(blocks) + feature_layer
        if not (0 <= idx < len(blocks)):
            raise ValueError(f"feature_layer {feature_layer} out of range for {len(blocks)} blocks")
        self.feature_layer = idx
        self._captured: dict[str, torch.Tensor] = {}
        blocks[idx].register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output):
        # SanaVideoMSBlock returns the updated token tensor (B, N, hidden_size).
        self._captured["feats"] = output

    @property
    def hidden_size(self) -> int:
        return int(self.dit.hidden_size)

    @property
    def out_channels(self) -> int:
        return int(self.dit.out_channels)

    def forward(self, x, timestep, y, mask=None, **kwargs):
        return self.dit(x, timestep, y, mask=mask, **kwargs)

    def forward_with_features(self, x, timestep, y, mask=None, **kwargs):
        """Run the video DiT, returning (velocity_pred, block_features).

        block_features: (B, N_tokens, hidden_size) from `feature_layer`.
        """
        self._captured.clear()
        pred = self.dit(x, timestep, y, mask=mask, **kwargs)
        feats = self._captured.get("feats", None)
        if feats is None:
            raise RuntimeError(
                "Video feature hook did not fire; cannot provide cross-attention memory."
            )
        return pred, feats
