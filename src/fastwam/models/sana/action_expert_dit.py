"""Action expert built on the original Wan22 ``ActionDiT``.

Instead of the from-scratch ``SanaActionExpert``, this reuses the original
FastWAM ActionDiT — a standard self-attention (RoPE) + cross-attention + FFN
DiT over action tokens — initialised from the interpolated Wan-video-DiT
backbone (``ActionDiT_linear_interp_Wan22``), exactly the non-random init the
reference uses. The world-model coupling is the ActionDiT's cross-attention:
its ``context`` memory is set to ``[SANA video features ; gemma text/proprio]``.

Only the two input projections (video/text -> ActionDiT text_dim) are new; the
ActionDiT backbone (blocks, text/time embedding) comes from the pretrained
interpolated checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..wan22.action_dit import ActionDiT


class SanaActionExpertDiT(nn.Module):
    def __init__(self, action_dit: ActionDiT, video_feat_dim: int, text_feat_dim: int):
        super().__init__()
        self.dit = action_dit
        ctx_dim = int(action_dit.text_dim)
        # New (random-init) projections of the cross-attention memory into the
        # ActionDiT's context dim, so its text_embedding loads from the ckpt.
        self.video_proj = nn.Linear(int(video_feat_dim), ctx_dim)
        self.text_proj = nn.Linear(int(text_feat_dim), ctx_dim)
        self.action_dim = int(action_dit.action_dim)

    def forward(self, noisy_action, timestep, video_feats, text_context, text_mask=None):
        """
        noisy_action: (B, Ta, action_dim)
        timestep:     (B,)
        video_feats:  (B, Nv, video_feat_dim)  — SANA video DiT block features
        text_context: (B, L, text_feat_dim)    — gemma text (+ optional proprio token)
        text_mask:    (B, L) bool
        """
        dt = noisy_action.dtype
        vctx = self.video_proj(video_feats.to(dt))
        tctx = self.text_proj(text_context.to(dt))
        context = torch.cat([vctx, tctx], dim=1)  # (B, Nv+L, ctx_dim)

        B = context.shape[0]
        vmask = torch.ones((B, vctx.shape[1]), dtype=torch.bool, device=context.device)
        if text_mask is None:
            text_mask = torch.ones((B, tctx.shape[1]), dtype=torch.bool, device=context.device)
        mask = torch.cat([vmask, text_mask.to(torch.bool)], dim=1)  # (B, Nv+L)

        return self.dit(noisy_action, timestep, context, mask)
