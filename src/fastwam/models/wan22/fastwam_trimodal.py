"""FastWAM Trimodal: video + action + text experts under one MoT.

This applies the Uni-ViGU "modality-driven MoE" pattern to FastWAM:

  - Video expert: continuous flow-matching (unchanged Wan2.2-TI2V-5B backbone).
  - Action expert: continuous flow-matching (unchanged ActionDiT).
  - Text expert:   discrete flow-matching, see `TextDiT` and
                   `WanDiscreteFlowMatchScheduler`.

All three experts share self-attention via `MoT`. Each carries its own time
embedding, AdaLN modulation, and FFN. Cross-attention to the T5 condition
is reused per-expert through its own `pre_dit`.

The training loss is::

    L_total = lambda_video * MSE_v + lambda_action * MSE_a + lambda_text * CE_t

with (time_v, time_a, time_t) sampled independently per modality so that
each forward pass covers any of: t2v / t2a / v2t / joint denoising.

The training-time `time_*` triple can also be *probabilistically* set to
zero for specific modalities to deliberately bias the model toward
particular conditional flows (e.g., set time_t=0 to learn t2v with text as
clean condition). Controlled by `text_recall_prob` / `text_clean_prob`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .schedulers.scheduler_discrete import WanDiscreteFlowMatchScheduler
from .text_dit import TextDiT

logger = get_logger(__name__)


class FastWAMTrimodal(FastWAM):
    """FastWAM extended with a text expert for joint v+a+t flow matching."""

    def __init__(
        self,
        *,
        video_expert,
        action_expert,
        text_expert: TextDiT,
        mot,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        text_train_shift: float = 1.0,
        text_num_train_timesteps: int = 1000,
        text_mask_token_id: Optional[int] = None,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_text: float = 1.0,
        # Probability that text is presented as a *clean* condition (time_t=0)
        # in a given training step; otherwise text is denoised jointly.
        # text_clean_prob + text_recall_prob + text_joint_prob should sum to 1.
        text_clean_prob: float = 0.4,
        text_recall_prob: float = 0.4,
        text_joint_prob: float = 0.2,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        self.text_expert = text_expert
        self.loss_lambda_text = float(loss_lambda_text)

        if text_expert.vocab_size <= 0:
            raise ValueError("`text_expert.vocab_size` must be > 0.")

        self.train_text_scheduler = WanDiscreteFlowMatchScheduler(
            vocab_size=int(text_expert.vocab_size),
            num_train_timesteps=int(text_num_train_timesteps),
            shift=float(text_train_shift),
            mask_token_id=text_mask_token_id,
        )
        self.infer_text_scheduler = self.train_text_scheduler  # same params at infer

        total_prob = float(text_clean_prob + text_recall_prob + text_joint_prob)
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(
                "text_clean_prob + text_recall_prob + text_joint_prob must sum to 1, "
                f"got {total_prob}"
            )
        self.text_clean_prob = float(text_clean_prob)
        self.text_recall_prob = float(text_recall_prob)
        self.text_joint_prob = float(text_joint_prob)

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        """Disable inherited factory; use `runtime.create_fastwam_trimodal` instead."""
        raise NotImplementedError(
            "FastWAMTrimodal uses `runtime.create_fastwam_trimodal`; do not call "
            "from_wan22_pretrained directly."
        )

    # ------------------------------------------------------------------
    # Attention mask: 3-segment [video | action | text]
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_mot_attention_mask(  # type: ignore[override]
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        text_seq_len: int = 0,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len + text_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        v_start, v_end = 0, video_seq_len
        a_start, a_end = video_seq_len, video_seq_len + action_seq_len
        t_start, t_end = a_end, a_end + text_seq_len

        # video -> video (same logic as base FastWAM / FastWAMJoint)
        mask[v_start:v_end, v_start:v_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action  (full)
        mask[a_start:a_end, a_start:a_end] = True
        # action <-> full video (joint variant: action attends to all video tokens)
        mask[a_start:a_end, v_start:v_end] = True

        if text_seq_len > 0:
            # text -> text  (full self-attn within text segment)
            mask[t_start:t_end, t_start:t_end] = True
            # text <-> video / action  (text reasoning sees everything; mirrors Uni-ViGU)
            mask[t_start:t_end, v_start:v_end] = True
            mask[t_start:t_end, a_start:a_end] = True
            # video / action may also attend to text (instruction is global)
            mask[v_start:v_end, t_start:t_end] = True
            mask[a_start:a_end, t_start:t_end] = True
        return mask

    # ------------------------------------------------------------------
    # build_inputs: extend with text ids
    # ------------------------------------------------------------------

    def build_inputs(self, sample, tiled: bool = False):
        inputs = super().build_inputs(sample, tiled=tiled)
        if "text_input_ids" not in sample or "text_attention_mask" not in sample:
            raise ValueError(
                "FastWAMTrimodal requires `sample['text_input_ids']` and "
                "`sample['text_attention_mask']` (set return_text_tokens=True on the dataset)."
            )
        text_ids = sample["text_input_ids"]
        text_attn = sample["text_attention_mask"]
        if text_ids.ndim != 2:
            raise ValueError(
                f"`sample['text_input_ids']` must be 2D [B, L], got shape {tuple(text_ids.shape)}"
            )
        if text_attn.shape != text_ids.shape:
            raise ValueError(
                "`text_attention_mask` shape must match `text_input_ids`, "
                f"got {tuple(text_attn.shape)} vs {tuple(text_ids.shape)}"
            )
        inputs["text_input_ids"] = text_ids.to(device=self.device, dtype=torch.long, non_blocking=True)
        inputs["text_attention_mask"] = text_attn.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return inputs

    # ------------------------------------------------------------------
    # training_loss: video MSE + action MSE + text CE
    # ------------------------------------------------------------------

    def _sample_text_timestep(self, batch_size: int, dtype: torch.dtype) -> tuple[torch.Tensor, str]:
        """Return (time_t, mode_str). `mode_str` is one of {clean, recall, joint}.

        IMPORTANT: All ranks MUST execute the same set of ops (especially the
        text CE loss). Otherwise DeepSpeed ZeRO's gradient all-reduce sees
        divergent participation patterns and hangs in a collective timeout.
        So we keep the *forward graph* identical across ranks/iterations and
        only let `time_t` vary; clean mode is realized by setting time_t=0
        per-sample, not by skipping the loss.
        """
        r = float(torch.rand((), device="cpu").item())
        if r < self.text_clean_prob:
            mode = "clean"
            t = torch.zeros((batch_size,), device=self.device, dtype=dtype)
        elif r < self.text_clean_prob + self.text_recall_prob:
            mode = "recall"
            t = self.train_text_scheduler.sample_training_t(batch_size, device=self.device, dtype=dtype)
        else:
            mode = "joint"
            t = self.train_text_scheduler.sample_training_t(batch_size, device=self.device, dtype=dtype)
        return t, mode

    def training_loss(self, sample, tiled: bool = False):  # type: ignore[override]
        import torch.nn.functional as F

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        text_input_ids = inputs["text_input_ids"]
        text_attention_mask = inputs["text_attention_mask"]

        # ----- VIDEO branch -----
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        # ----- ACTION branch -----
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # ----- TEXT branch -----
        timestep_text, text_mode = self._sample_text_timestep(batch_size=batch_size, dtype=input_latents.dtype)
        noisy_text_tokens = self.train_text_scheduler.add_noise(
            target_tokens=text_input_ids,
            timestep=timestep_text,
            attention_mask=text_attention_mask,
        )

        # ----- Pre-DiT for all three experts -----
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        text_pre = self.text_expert.pre_dit(
            text_tokens=noisy_text_tokens,
            timestep=timestep_text,
            context=context,
            context_mask=context_mask,
        )

        # ----- Joint MoT forward -----
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            text_seq_len=text_pre["tokens"].shape[1],
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
                "text": text_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
                "text": text_pre["freqs"],
            },
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                "text": {"context": text_pre["context"], "mask": text_pre["context_mask"]},
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
                "text": text_pre["t_mod"],
            },
        )

        # ----- Post-DiT -----
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        text_logits = self.text_expert.post_dit(tokens_out["text"], text_pre)

        # ----- Video loss -----
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        # ----- Action loss -----
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        # ----- Text loss -----
        # Always run CE on the same forward graph; do NOT short-circuit by mode.
        # DeepSpeed ZeRO expects the same set of parameters to receive gradients
        # on every step across ranks; skipping the loss in "clean" mode caused
        # NCCL collective hangs in earlier smoke tests.
        loss_text = self.train_text_scheduler.cross_entropy_loss(
            logits=text_logits,
            target_tokens=text_input_ids,
            attention_mask=text_attention_mask,
            reduction="mean",
        )
        # In clean mode the input already equals target -> CE will be near 0 if
        # the model trivially copies its input, which we don't want to reward.
        # Down-weight (but keep non-zero) so the graph still flows and we don't
        # learn a copy short-circuit. recall/joint contribute full weight.
        if text_mode == "clean":
            loss_text = loss_text * 0.05

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.loss_lambda_text * loss_text
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_text": self.loss_lambda_text * float(loss_text.detach().item()),
            "text_mode_clean": float(text_mode == "clean"),
            "text_mode_recall": float(text_mode == "recall"),
            "text_mode_joint": float(text_mode == "joint"),
        }
        return loss_total, loss_dict

    # ------------------------------------------------------------------
    # Inference: by default we only need action (VLA mode). The action
    # path is identical to the parent; we just need to ensure text tokens
    # are presented as a clean condition (time_t=0, x_t = ground-truth ids).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_joint_noise(  # type: ignore[override]
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_input_ids is None:
            # Fall back to parent (two-expert) joint path.
            return super()._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                gt_action=gt_action,
            )

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        timestep_text = torch.zeros_like(timestep_action)
        text_pre = self.text_expert.pre_dit(
            text_tokens=text_input_ids,
            timestep=timestep_text,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            text_seq_len=text_pre["tokens"].shape[1],
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
                "text": text_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
                "text": text_pre["freqs"],
            },
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                "text": {"context": text_pre["context"], "mask": text_pre["context_mask"]},
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
                "text": text_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(  # type: ignore[override]
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_input_ids is None:
            return super()._predict_action_noise(
                first_frame_latents=first_frame_latents,
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )

        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        timestep_text = torch.zeros_like(timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        text_pre = self.text_expert.pre_dit(
            text_tokens=text_input_ids,
            timestep=timestep_text,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            text_seq_len=text_pre["tokens"].shape[1],
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
                "text": text_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
                "text": text_pre["freqs"],
            },
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                "text": {"context": text_pre["context"], "mask": text_pre["context_mask"]},
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
                "text": text_pre["t_mod"],
            },
        )
        return self.action_expert.post_dit(tokens_out["action"], action_pre)
