"""AGRA coupling: world<->action CROSS-ATTENTION ("Making Foresight Actionable",
arXiv 2606.12217), cross-attn architecture part only (NOT the AGRA alignment loss).

The world model (Cosmos video DiT, 28 blocks) is run TWICE per step:
  1. The standard random-tau_v forward (``forward_standalone``) -> ``pred_v`` for the
     VIDEO loss, identical to the cross_attn coupling.
  2. A SEPARATE foresight pass at FIXED high noise tau_v=1 (pure-noise latent),
     conditioned on the current observation o0 (first latent frame) + text, to
     extract 8 layers' hidden states (the "foresight"). These are projected per-
     layer and fed as cross-attention contexts to the standalone action DiT
     (``ForesightActionHead``, 8 layers).

Multi-layer bridge: action layer j (0..7) reads video layer
    ell_j = round(j * (M-1) / (N-1)),  M=28 video blocks, N=8 action layers
        -> {0, 4, 8, 12, 15, 19, 23, 27}.
Each video hidden ``H^vid_{ell_j} [B, Sv, vdim]`` is projected to the cross-attn
dim (2048) by a per-layer ``Proj_j`` (``model.agra_video_projs[j]``, created in
``setup`` so FSDP/optimizer see them).

This module imports NO cosmos_predict2 at top level (the action head is pure torch
and the video calls go through ``model.video_expert``), so the registry imports
cleanly without the cosmos env.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import register_coupling
from .base import Coupling

# Multi-layer bridge: M=28 video blocks, N=8 action layers ->
# ell_j = round(j*(M-1)/(N-1)) for j in 0..7  ==  {0,4,8,12,15,19,23,27}.
AGRA_VIDEO_LAYERS = [0, 4, 8, 12, 15, 19, 23, 27]
AGRA_CROSSATTN_DIM = 2048


@register_coupling("agra")
class AGRACoupling(Coupling):
    name = "agra"

    def setup(self, model) -> None:
        # per-layer projection of the video hidden (model_channels) -> the action head's
        # cross-attn dim. One per action CROSS block; submodules on the model so
        # FSDP/optimizer see them. The action head itself is built in
        # runtime.create_fastwam_cosmos (ForesightActionHead OR the real GR00T DiT) and
        # set as model.action_expert (read via model.action_head property below).
        #
        # #contexts (= #cross blocks) and the cross-attn dim are read from the action
        # head: ForesightActionHead -> 8; GR00T DiT (32-layer interleaved) -> 16. The
        # multi-layer bridge (Eq.14) maps the video DiT's M blocks onto those N contexts:
        # layer round(j*(M-1)/(N-1)) feeds cross block j.
        from ..gr00t_action_dit import bridge_video_layers

        vdim = int(getattr(model.video_expert.net, "model_channels", AGRA_CROSSATTN_DIM))
        head = model.action_head
        n_ctx = int(getattr(head, "num_contexts", len(AGRA_VIDEO_LAYERS)))
        cdim = int(getattr(head, "crossattn_dim", AGRA_CROSSATTN_DIM))
        n_vblocks = len(model.video_expert.net.blocks)
        model.agra_video_layers = bridge_video_layers(n_vblocks, n_ctx)
        model.agra_video_projs = nn.ModuleList(
            [nn.Linear(vdim, cdim) for _ in range(n_ctx)]
        ).to(device=model.device, dtype=model.torch_dtype)

    def forward(self, model, noisy_latents, t_v, noisy_action, t_a, crossattn_emb):
        # (a) random-tau_v forward -> pred_v for the video loss (same as cross_attn).
        pred_v, _ = model.video_expert.forward_standalone(
            noisy_latents, t_v, crossattn_emb, feature_layer=-1
        )

        # (b) foresight pass at FIXED sigma=1 (pure noise), o0-conditioned.
        noise = torch.randn_like(noisy_latents)
        B, _, T = noise.shape[0], noise.shape[1], noise.shape[2]
        # per-frame timesteps: pure noise (sigma=1) everywhere; the o0-conditioning
        # path inside forward_foresight overrides the first frame to ~clean (t=0).
        # NB: the scheduler convention is t in [0, num_train_timesteps] (sigma = t/N),
        # so pure noise is t=N (NOT 1.0 — that would be sigma~0, i.e. near-clean, which
        # mismatches the pure-noise input and yields garbage foresight features).
        ts = noise.new_full((B, T), float(model.train_video_scheduler.num_train_timesteps))
        feats = model.video_expert.forward_foresight(
            noise, ts, crossattn_emb,
            layers=getattr(model, "agra_video_layers", AGRA_VIDEO_LAYERS),
            o0_latent=getattr(model, "_agra_o0_latent", None),
            cond_frames=1,
        )

        # (c) per-layer projection -> cross-attn contexts.
        G = [proj(f.to(proj.weight.dtype)) for proj, f in zip(model.agra_video_projs, feats)]

        # (d) action head reads the multi-layer foresight (proprio prepended inside).
        proprio0 = getattr(model, "_agra_proprio0", None)
        if proprio0 is None:
            # proprio is required for the AGRA action head's prepended state token.
            raise RuntimeError(
                "AGRA coupling needs model._agra_proprio0 (set in build_inputs/training_loss); "
                "the dataset must provide proprio and proprio_dim must be configured."
            )
        pred_a = model.action_head(noisy_action, t_a, proprio0, G)
        return pred_v, pred_a
