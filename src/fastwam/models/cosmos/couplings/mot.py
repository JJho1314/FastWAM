"""MoT coupling: original-FastWAM mixture-of-transformers joint attention.

Both streams run layer-by-layer through ``mot_block_forward``: ONE joint
self-attention over the concatenated K/V of both streams, then per-stream
cross-attn(text)+MLP. Video is independent of the action unless
``model.mot_bidirectional`` is set. (Logic relocated verbatim from
``FastWAMCosmos.mot_forward``.)
"""
import os

import torch

from . import register_coupling
from .base import Coupling
from ..mot import CosmosMoTStream, mot_block_forward


@register_coupling("mot")
class MoTCoupling(Coupling):
    name = "mot"

    def forward(self, model, noisy_latents, t_v, noisy_action, t_a, crossattn_emb):
        # First-frame (current-observation) conditioning: keep latent frame 0 clean
        # via the Cosmos video2world FRAME_REPLACE path (mask channel + per-frame t),
        # so the video tokens the action joint-attends to carry the current image.
        # Threaded by training_loss / infer_action as model._mot_o0_latent.
        o0 = getattr(model, "_mot_o0_latent", None)
        cond_frames = int(getattr(model, "_mot_cond_frames", 1))
        v = model.video_expert.prepare(noisy_latents, t_v, crossattn_emb,
                                       o0_latent=o0, cond_frames=cond_frames)
        a = model.action_expert.prepare(noisy_action, t_a, crossattn_emb, model.video_expert.net)

        vs = CosmosMoTStream(v["tokens"], v["rope"], v["t_emb"], v["crossattn"],
                             v["THW"][0], v["THW"][1] * v["THW"][2], v["adaln_lora"])
        as_ = CosmosMoTStream(a["tokens"], a["rope"], a["t_emb"], a["crossattn"],
                              a["THW"][0], a["THW"][1] * a["THW"][2], a["adaln_lora"])

        # Official FastWAM attention masks (Wan fastwam.py:394-407 + wan_video_dit.py
        # build_video_to_video_mask "first_frame_causal"). Built once (same per block):
        #   - action -> [first-frame video tokens ; all action tokens]  (mask[Sv:,:ff],
        #     mask[Sv:,Sv:]); action does NOT see the noised/predicted future frames.
        #   - video  -> "first_frame_causal": the first-frame tokens do NOT attend to
        #     later frames, so their hidden state is a CLEAN current-observation
        #     representation (what the action reads), uncontaminated by the noised
        #     future. Later frames attend bidirectionally.
        # Skipped when not o0-conditioned (legacy full-attention behaviour).
        action_mask = None
        video_mask = None
        if o0 is not None:
            T, H, W = v["THW"]
            HW = H * W
            Sv = T * HW
            Sa = a["tokens"].shape[1]
            ff = cond_frames * HW  # first cond-frame(s) video tokens
            action_mask = torch.zeros(Sa, Sv + Sa, dtype=torch.bool, device=as_.x.device)
            action_mask[:, :ff] = True   # conditioning-frame video tokens
            action_mask[:, Sv:] = True   # all action tokens (self-attention)
            video_mask = torch.ones(Sv, Sv, dtype=torch.bool, device=vs.x.device)
            video_mask[:ff, ff:] = False  # first-frame tokens don't attend to later frames

        # Per-MoT-layer activation checkpointing (recompute in backward) to fit a large
        # micro-batch; enabled via FASTWAM_MOT_CHECKPOINT=1 (the cosmos SAC can't reach
        # the MoT path since it bypasses Block.forward).
        use_ckpt = os.environ.get("FASTWAM_MOT_CHECKPOINT", "0") == "1"
        for vblk, ablk in zip(model.video_expert.net.blocks, model.action_expert.blocks):
            mot_block_forward(vblk, ablk, vs, as_, bidirectional=model.mot_bidirectional,
                              video_mask=video_mask, action_mask=action_mask,
                              use_checkpoint=use_ckpt)

        pred_v = model.video_expert.finalize(vs.x, v["t_emb"], v["adaln_lora"], v["THW"])
        pred_a = model.action_expert.finalize(as_.x)
        return pred_v, pred_a
