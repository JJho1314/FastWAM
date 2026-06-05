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

import os
import time
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
        video_temporal_downsample: int = 4,
        video_train_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        vae_latent_cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.vae = vae
        self.vae_name = vae_name
        self._vae_encode_fn = vae_encode_fn
        # Optional precomputed VAE-latent cache (see scripts/precompute_vae_latents_sana.py).
        # When set, build_inputs loads latents by `sample['cache_key']` and skips the
        # (frozen, deterministic) VAE encode — the single biggest per-step cost (~28%).
        self.vae_latent_cache_path = vae_latent_cache_path
        self._latent_cache = None  # lazily opened: dict(mmap=..., n=..., shape=...)
        self.text_dim = int(text_dim)
        self.device = device
        self.torch_dtype = torch_dtype
        self.detach_video_feats = bool(detach_video_feats)
        self.video_temporal_downsample = int(video_temporal_downsample)
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        # The trainer trains exactly `self.dit.parameters()`. `dit` is a
        # *property* (below), NOT a registered submodule, so the experts are
        # registered only once (as self.video_expert / self.action_expert).
        # Registering them a second time under self.dit would make FSDP's
        # recursive wrap visit the same module twice -> AssertionError.
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

    @property
    def dit(self):
        """The trainable expert(s) the trainer optimizes. A non-registered view
        (built on access) so the experts aren't double-registered as submodules
        (which breaks FSDP's recursive wrap)."""
        modules = {"action": self.action_expert}
        if self.train_video_expert:
            modules["video"] = self.video_expert
        return nn.ModuleDict(modules)

    # ------------------------------------------------------------------ utils
    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        if self._vae_encode_fn is None:
            raise RuntimeError("`vae_encode_fn` was not provided to FastWAMSana.")
        # Kept in fp32: profiling showed bf16 did NOT speed the encode (the WanVAE
        # 3D-conv over 33 frames x 2 cams is op/bandwidth-bound, not precision-bound:
        # ~211ms fp32 vs ~200ms bf16) AND bf16 latents measurably raised the
        # world-model loss (0.48 -> 1.11 at the same early step). The only real way
        # to remove this ~28%-of-step cost is to precompute/cache the latents.
        return self._vae_encode_fn(self.vae_name, self.vae, video, device=self.device)

    def _open_latent_cache(self):
        if self._latent_cache is not None:
            return self._latent_cache
        if not self.vae_latent_cache_path:
            return None
        import json
        import numpy as np
        base = self.vae_latent_cache_path
        with open(os.path.join(base, "meta.json")) as f:
            meta = json.load(f)
        shape = tuple(int(s) for s in meta["shape"])  # per-sample latent shape
        n = int(meta["n"])
        dtype = np.dtype(meta["dtype"])
        mmap = np.memmap(os.path.join(base, "latents.mmap"), dtype=dtype, mode="r", shape=(n, *shape))
        self._latent_cache = {"mmap": mmap, "n": n, "shape": shape}
        logger.info("Opened VAE latent cache %s (n=%d shape=%s dtype=%s)", base, n, shape, dtype)
        return self._latent_cache

    def _load_cached_latents(self, cache_keys):
        cache = self._open_latent_cache()
        if cache is None or cache_keys is None:
            return None
        import numpy as np
        keys = cache_keys.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if (keys < 0).any() or (keys >= cache["n"]).any():
            raise IndexError(f"cache_key out of range [0,{cache['n']}): min={keys.min()} max={keys.max()}")
        arr = np.ascontiguousarray(cache["mmap"][keys])  # (B, *shape)
        return torch.from_numpy(arr).to(device=self.device, dtype=self.torch_dtype)

    def _video_loss_per_sample(self, pred_v, target_v, image_is_pad):
        """Per-sample video flow loss, masking padded frames out (mirrors the
        Wan22 path's `_compute_video_loss_per_sample`). `pred_v/target_v` are
        (B, C, F, H, W); `image_is_pad` is (B, T) at pixel-frame resolution."""
        loss_tok = F.mse_loss(pred_v.float(), target_v.float(), reduction="none").mean(dim=(1, 3, 4))  # (B, F)
        if image_is_pad is None:
            return loss_tok.mean(dim=1)
        B, n_frames = image_is_pad.shape
        tf = self.video_temporal_downsample
        # Fold pixel-frame padding to latent-frame resolution: frame 0 -> latent 0,
        # then each group of `tf` tail frames -> one latent frame (padded only if all pad).
        if tf <= 0 or (n_frames - 1) % tf != 0:
            return loss_tok.mean(dim=1)
        tail_pad = image_is_pad[:, 1:].view(B, -1, tf).all(dim=2)
        video_is_pad = torch.cat([image_is_pad[:, :1], tail_pad], dim=1)  # (B, F)
        if video_is_pad.shape[1] != loss_tok.shape[1]:
            return loss_tok.mean(dim=1)
        valid = (~video_is_pad).to(loss_tok)
        return (loss_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

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
        # Prefer the precomputed latent cache (skips the ~28%-of-step VAE encode);
        # fall back to encoding when no cache / key is available.
        cached = self._load_cached_latents(sample.get("cache_key", None))
        if cached is not None:
            input_latents = cached
            if os.environ.get("FASTWAM_VERIFY_LATENT_CACHE"):
                ref = self._vae_encode(
                    video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                ).to(self.torch_dtype)
                maxdiff = (ref - input_latents).abs().amax().item()
                refmag = ref.abs().amax().item()
                print(f"[LATENT-CACHE-VERIFY] max|cache-live|={maxdiff:.3e} "
                      f"ref_absmax={refmag:.3e} shape={tuple(input_latents.shape)}", flush=True)
        else:
            video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._vae_encode(video).to(self.torch_dtype)

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} / {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # Proprio is conditioning for the *action* expert only. Appending it to
        # the video DiT's text tokens would make L = model_max_length + 1, which
        # the SANA caption embedder asserts against (L must be <= model_max_length).
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
        _prof = bool(os.environ.get("FASTWAM_SANA_PROFILE"))
        def _now():
            if _prof:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0.0
        _t_start = _now()
        inputs = self.build_inputs(sample, tiled=tiled)
        _t_build = _now()
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
        # Run the video DiT in bf16 for speed (user directive: SANA bf16). FSDP
        # mixed_precision is off (fp32 master weights), so ONLY this autocast region
        # computes in bf16 — the action branch below runs in fp32. video_feats come
        # out bf16 and are cast back to the action latent's fp32 dtype inside the
        # action expert (action_expert_dit.py: `video_feats.to(dt)`).
        y = context.unsqueeze(1)
        _t_sana0 = _now()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_v, video_feats = self.video_expert.forward_with_features(
                noisy_v, t_v, y, mask=context_mask
            )
        _t_sana = _now()

        loss_v_per = self._video_loss_per_sample(pred_v, target_v, inputs["image_is_pad"])
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

        _t_act0 = _now()
        pred_a = self.action_expert(noisy_a, t_a, vf, action_context, action_context_mask)
        _t_act = _now()

        if os.environ.get("FASTWAM_SANA_DEBUG_NAN"):
            def _stat(name, t):
                tf = t.float()
                finite = torch.isfinite(tf).all().item()
                print(f"[NAN-DEBUG] {name}: finite={finite} min={tf.amin().item():.3e} "
                      f"max={tf.amax().item():.3e} absmax={tf.abs().amax().item():.3e}", flush=True)
            for nm, t in [("latents", latents), ("context", context), ("video_feats", video_feats),
                          ("pred_v", pred_v), ("target_v", target_v), ("noisy_a", noisy_a),
                          ("pred_a", pred_a)]:
                _stat(nm, t)

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
        if _prof:
            _t_end = _now()
            vae = _t_build - _t_start          # build_inputs (dominated by VAE encode)
            sana = _t_sana - _t_sana0          # SANA video DiT forward
            act = _t_act - _t_act0             # action DiT forward
            rest = (_t_end - _t_start) - vae - sana - act  # noise/scheduler + losses
            self._prof_record(vae=vae, sana_fwd=sana, action_fwd=act, rest_fwd=rest)
        return loss, loss_dict

    def _prof_record(self, **secs):
        st = getattr(self, "_prof_state", None)
        if st is None:
            st = {"n": 0, "acc": {}}
            self._prof_state = st
        st["n"] += 1
        for k, v in secs.items():
            st["acc"][k] = st["acc"].get(k, 0.0) + v
        if st["n"] % 10 == 0:
            n = st["n"]
            order = ["vae", "sana_fwd", "action_fwd", "rest_fwd"]
            parts = " ".join(f"{k}={st['acc'][k] / n * 1000:.1f}ms" for k in order if k in st["acc"])
            total = sum(st["acc"].values()) / n * 1000
            print(f"[PROFILE] n={n} per-step fwd: {parts} | fwd_total={total:.1f}ms", flush=True)

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
