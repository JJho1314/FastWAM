"""FastWAMSana: world-action model with a SANA-Video backbone.

Architecture (cross-attention coupling, *not* MoT joint attention):

    video latents --(+noise)--> SanaMSVideo  --> video velocity  (video flow loss)
                                      |  block features (B, Nv, Dv)
                                      v
    action chunk  --(+noise)--> SanaActionExpert (cross-attends video feats
                                + gemma text/proprio context) --> action velocity
                                                                  (action flow loss)

The flow-matching scheduler, data pipeline, and trainer are reused unchanged
from the Wan22 path. The module exposes the same trainer-facing surface as
`FastWAM`: `training_loss(sample) -> (loss, dict)`, a `.dit` module holding the
trainable parameters, an optional `proprio_encoder`, and
`save_checkpoint`/`load_checkpoint`.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from ..wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .action_expert import SanaActionExpert
from .video_expert import SanaVideoExpert

logger = get_logger(__name__)


class FastWAMSana(nn.Module):
    def __init__(
        self,
        video_expert: SanaVideoExpert,
        action_expert: SanaActionExpert,
        vae,
        vae_name: str = "WanVAE",
        vae_encode_fn: Optional[Callable] = None,
        text_dim: int = 2304,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_video_expert: bool = True,
        detach_video_feats: bool = False,
        video_train_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.vae = vae
        self.vae_name = vae_name
        self._vae_encode_fn = vae_encode_fn
        self.text_dim = int(text_dim)
        self.device = device
        self.torch_dtype = torch_dtype
        self.detach_video_feats = bool(detach_video_feats)
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        # Trainer trains exactly `self.dit.parameters()` (+ proprio_encoder).
        dit = nn.ModuleDict({"action": action_expert})
        if train_video_expert:
            dit["video"] = video_expert
        self.dit = dit
        self.train_video_expert = bool(train_video_expert)

        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_train_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_train_shift
        )

    # ------------------------------------------------------------------ utils
    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        if self._vae_encode_fn is None:
            raise RuntimeError("`vae_encode_fn` was not provided to FastWAMSana.")
        return self._vae_encode_fn(self.vae_name, self.vae, video, device=self.device)

    def _append_proprio_to_context(self, context, context_mask, proprio):
        # proprio: (B, proprio_dim) -> one extra valid context token
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype)
        ).unsqueeze(1)  # (B, 1, D)
        context = torch.cat([context, proprio_token.to(context.dtype)], dim=1)
        extra = torch.ones(
            (context_mask.shape[0], 1), device=context_mask.device, dtype=context_mask.dtype
        )
        context_mask = torch.cat([context_mask, extra], dim=1)
        return context, context_mask

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError("FastWAMSana requires `sample['context']` and `sample['context_mask']`.")
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)

        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,a_dim], got {tuple(action.shape)}")

        action_is_pad = sample.get("action_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)

        # video is in [-1, 1] (matches Wan/SANA WanVAE convention).
        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._vae_encode(video).to(self.torch_dtype)

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} / {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # Proprio is conditioning for the *action* expert only. The SANA video
        # DiT's caption embedder is sized to model_max_length, so we must not
        # change the video text-token count.
        proprio_first = None
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` required when `proprio_dim` is set.")
            proprio_first = proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype)

        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "proprio_first": proprio_first,
        }

    # ------------------------------------------------------------- train loss
    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        latents = inputs["input_latents"]
        B = latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]

        # ---- video branch (world-model / imagination objective) ----
        noise_v = torch.randn_like(latents)
        t_v = self.train_video_scheduler.sample_training_t(B, self.device, latents.dtype)
        noisy_v = self.train_video_scheduler.add_noise(latents, noise_v, t_v)
        target_v = self.train_video_scheduler.training_target(latents, noise_v, t_v)

        # SANA DiT consumes y of shape (B, 1, L, text_dim) and a (B, L) mask.
        y = context.unsqueeze(1)
        pred_v, video_feats = self.video_expert.forward_with_features(
            noisy_v, t_v, y, mask=context_mask
        )

        loss_v_token = F.mse_loss(pred_v.float(), target_v.float(), reduction="none").mean(dim=(1, 3, 4))
        loss_v_per = loss_v_token.mean(dim=1)
        w_v = self.train_video_scheduler.training_weight(t_v).to(loss_v_per)
        loss_video = (loss_v_per * w_v).mean()

        # ---- action branch (cross-attends video features) ----
        noise_a = torch.randn_like(action)
        t_a = self.train_action_scheduler.sample_training_t(B, self.device, action.dtype)
        noisy_a = self.train_action_scheduler.add_noise(action, noise_a, t_a)
        target_a = self.train_action_scheduler.training_target(action, noise_a, t_a)

        vf = video_feats.detach() if self.detach_video_feats else video_feats

        # Build action-expert text context (gemma context + optional proprio token).
        action_context, action_context_mask = context, context_mask
        if self.proprio_encoder is not None and inputs["proprio_first"] is not None:
            ptok = self.proprio_encoder(inputs["proprio_first"]).unsqueeze(1).to(context.dtype)
            action_context = torch.cat([context, ptok], dim=1)
            ones = torch.ones(
                (context_mask.shape[0], 1), device=context_mask.device, dtype=context_mask.dtype
            )
            action_context_mask = torch.cat([context_mask, ones], dim=1)

        pred_a = self.action_expert(noisy_a, t_a, vf, action_context, action_context_mask)

        a_loss_token = F.mse_loss(pred_a.float(), target_a.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(a_loss_token)
            a_loss_per = (a_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            a_loss_per = a_loss_token.mean(dim=1)
        w_a = self.train_action_scheduler.training_weight(t_a).to(a_loss_per)
        loss_action = (a_loss_per * w_a).mean()

        loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss": loss.detach(),
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
        }
        return loss, loss_dict

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)

    # ------------------------------------------------------------- checkpoint
    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {"dit_sana": self.dit.state_dict(), "step": step}
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "dit_sana" in payload:
            self.dit.load_state_dict(payload["dit_sana"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing `dit_sana` key: {path}")
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
