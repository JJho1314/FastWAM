"""FastWAM with a Cosmos-Predict2.5 video backbone, coupled to an action expert
via the original-FastWAM MoT (joint masked attention).

Both streams (video DiT + action DiT) are run layer-by-layer through
``mot_block_forward`` (cosmos/mot.py), which does ONE joint self-attention over
the concatenated K/V of both streams, then per-stream cross-attn(text)+MLP. Each
stream is then read out (video -> velocity latent, action -> velocity actions)
and trained with flow matching (velocity target = noise - sample), matching the
Cosmos rectified-flow objective and the original FastWAM loss.

Requires video_expert.net.blocks and action_expert.blocks to have equal length.
UNTESTED until a GPU frees on HPC3 (QOS-blocked at time of writing).
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger
from ..wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .mot import CosmosMoTStream, mot_block_forward

logger = get_logger(__name__)


class FastWAMCosmos(nn.Module):
    def __init__(
        self,
        video_expert,
        action_expert,
        vae,
        vae_encode_fn: Optional[Callable] = None,
        vae_name: str = "CosmosWan2pt1",
        crossattn_dim: int = 1024,
        qwen_dim: int = 3584,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_train_shift: float = 3.0,
        action_train_shift: float = 3.0,
        num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.vae = vae
        self._vae_encode_fn = vae_encode_fn
        self.vae_name = vae_name
        self.crossattn_dim = int(crossattn_dim)
        self.device = device
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        nv, na = len(video_expert.net.blocks), len(action_expert.blocks)
        if nv != na:
            raise ValueError(f"MoT needs equal block counts: video={nv} action={na}")

        # Qwen2.5-VL text embeds are 3584-dim; the MiniTrainDIT crossattn wants 1024.
        # A learned projection (trained) avoids needing Cosmos' exact text projection.
        self.text_proj = (
            nn.Linear(int(qwen_dim), self.crossattn_dim).to(device=device, dtype=torch_dtype)
            if int(qwen_dim) != self.crossattn_dim else nn.Identity()
        )

        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = (
            nn.Linear(self.proprio_dim, self.crossattn_dim).to(torch_dtype)
            if self.proprio_dim is not None else None
        )

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=video_train_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=action_train_shift
        )

    # The trainer optimizes exactly these (video DiT + action DiT [+ proprio]).
    @property
    def dit(self):
        return nn.ModuleDict({"video": self.video_expert, "action": self.action_expert})

    def _vae_encode(self, video):
        if self._vae_encode_fn is None:
            raise RuntimeError("`vae_encode_fn` was not provided to FastWAMCosmos.")
        return self._vae_encode_fn(self.vae_name, self.vae, video, device=self.device)

    # ------------------------------------------------------------- MoT forward
    def mot_forward(self, noisy_latents, t_v, noisy_action, t_a, crossattn_emb):
        """Run both streams through the matched blocks with joint attention.

        Returns (pred_v_latent [B,C,T,H,W], pred_a [B,Ta,action_dim]).
        """
        v = self.video_expert.prepare(noisy_latents, t_v, crossattn_emb)
        a = self.action_expert.prepare(noisy_action, t_a, crossattn_emb, self.video_expert.net)

        vs = CosmosMoTStream(v["tokens"], v["rope"], v["t_emb"], v["crossattn"],
                             v["THW"][0], v["THW"][1] * v["THW"][2], v["adaln_lora"])
        as_ = CosmosMoTStream(a["tokens"], a["rope"], a["t_emb"], a["crossattn"],
                              a["THW"][0], a["THW"][1] * a["THW"][2], a["adaln_lora"])

        for vblk, ablk in zip(self.video_expert.net.blocks, self.action_expert.blocks):
            mot_block_forward(vblk, ablk, vs, as_)

        pred_v = self.video_expert.finalize(vs.x, v["t_emb"], v["adaln_lora"], v["THW"])
        pred_a = self.action_expert.finalize(as_.x)
        return pred_v, pred_a

    # ------------------------------------------------------------- build inputs
    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)
        proprio = sample.get("proprio", None)

        input_latents = self._vae_encode(video).to(self.torch_dtype)

        proprio_first = None
        if self.proprio_encoder is not None and proprio is not None:
            proprio_first = proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool)
        return {
            "input_latents": input_latents,
            "context": context,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "proprio_first": proprio_first,
        }

    # ------------------------------------------------------------- train loss
    def training_loss(self, sample, tiled: bool = False):
        inp = self.build_inputs(sample, tiled=tiled)
        latents = inp["input_latents"]
        action = inp["action"]
        crossattn = self.text_proj(inp["context"])  # Qwen 3584 -> crossattn_dim (1024)
        B = latents.shape[0]

        # append proprio token to the text context (action-conditioning side info)
        if self.proprio_encoder is not None and inp["proprio_first"] is not None:
            ptok = self.proprio_encoder(inp["proprio_first"]).unsqueeze(1).to(crossattn.dtype)
            crossattn = torch.cat([crossattn, ptok], dim=1)

        noise_v = torch.randn_like(latents)
        t_v = self.train_video_scheduler.sample_training_t(B, self.device, latents.dtype)
        noisy_v = self.train_video_scheduler.add_noise(latents, noise_v, t_v)
        target_v = self.train_video_scheduler.training_target(latents, noise_v, t_v)

        noise_a = torch.randn_like(action)
        t_a = self.train_action_scheduler.sample_training_t(B, self.device, action.dtype)
        noisy_a = self.train_action_scheduler.add_noise(action, noise_a, t_a)
        target_a = self.train_action_scheduler.training_target(action, noise_a, t_a)

        pred_v, pred_a = self.mot_forward(noisy_v, t_v, noisy_a, t_a, crossattn)

        loss_v = F.mse_loss(pred_v.float(), target_v.float())
        w_v = self.train_video_scheduler.training_weight(t_v).to(loss_v)
        loss_video = (loss_v * w_v.mean())

        a_tok = F.mse_loss(pred_a.float(), target_a.float(), reduction="none").mean(dim=2)
        if inp["action_is_pad"] is not None:
            valid = (~inp["action_is_pad"]).to(a_tok)
            a_per = (a_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            a_per = a_tok.mean(dim=1)
        w_a = self.train_action_scheduler.training_weight(t_a).to(a_per)
        loss_action = (a_per * w_a).mean()

        loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss, {
            "loss": loss.detach(),
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
        }

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)

    # ------------------------------------------------------------- checkpoint
    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {"dit_cosmos": self.dit.state_dict(), "step": step}
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        self.dit.load_state_dict(payload["dit_cosmos"], strict=False)
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"])
