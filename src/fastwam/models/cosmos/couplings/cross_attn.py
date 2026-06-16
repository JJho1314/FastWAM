"""Cross-attention coupling: the video DiT runs standalone and the action DiT
cross-attends to ONE of its hidden layers (``model.feature_layer``). One-directional
(video unaffected by the action). (Logic relocated verbatim from
``FastWAMCosmos.cross_attn_forward``; the ``video_feat_proj`` projection that was
created in ``__init__`` now lives in ``setup``.)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import register_coupling
from .base import Coupling


@register_coupling("cross_attn")
class CrossAttnCoupling(Coupling):
    name = "cross_attn"

    def setup(self, model) -> None:
        # project the video DiT hidden (model_channels) -> crossattn_dim so the
        # action cross-attention can consume [text ; proprio ; video features].
        vdim = int(getattr(model.video_expert.net, "model_channels", 2048))
        model.video_feat_proj = nn.Linear(vdim, model.crossattn_dim).to(
            device=model.device, dtype=model.torch_dtype
        )

    def forward(self, model, noisy_latents, t_v, noisy_action, t_a, crossattn_emb):
        pred_v, vfeat = model.video_expert.forward_standalone(
            noisy_latents, t_v, crossattn_emb, feature_layer=model.feature_layer
        )
        vfeat = model.video_feat_proj(vfeat)  # [B, Sv, D] -> [B, Sv, crossattn_dim]
        # action cross-attention memory = [text(+proprio) ; video features]
        action_context = torch.cat([crossattn_emb, vfeat], dim=1)
        pred_a = model.action_expert.forward_cross_attn(
            noisy_action, t_a, action_context, model.video_expert.net
        )
        return pred_v, pred_a
