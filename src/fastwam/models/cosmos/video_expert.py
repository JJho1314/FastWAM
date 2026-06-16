"""Cosmos-Predict2.5 video DiT (`MiniTrainDIT`) wrapper for FastWAM.

Loads the 2B net from the official config + the `net.`-prefixed EMA checkpoint,
and exposes the pieces the MoT driver (fastwam_cosmos.py) needs to run the video
and action streams layer-by-layer through joint attention:

  - ``net``                 the MiniTrainDIT (blocks, embedders, final_layer)
  - ``prepare(...)``        patch-embed + RoPE + timestep emb (everything before
                            the block loop), returning FLAT tokens [B, T*H*W, D]
  - ``finalize(...)``       final_layer + unpatchify -> velocity latent

Built with ``atten_backend="torch"`` so attention is plain SDPA (the MoT joint
masked attention path needs a mask-aware op and no context-parallel).
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_cosmos_2b_net(atten_backend: str = "torch"):
    """Instantiate the Predict2.5-2B MiniTrainDIT from the official LazyDict config."""
    from cosmos_predict2._src.predict2.configs.text2world.defaults.net import (
        COSMOS_V1_2B_NET_MININET,
    )
    from cosmos_predict2._src.imaginaire.lazy_config import instantiate

    cfg = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
    cfg.atten_backend = atten_backend
    # The Predict2.5-2B base ckpt was trained with in_channels=17 (16 VAE latent +
    # 1 video2world conditioning channel); the default config says 16. Match the
    # ckpt so x_embedder is (17+1)*patch = 72 (not 68).
    cfg.in_channels = 17
    # Disable cosmos selective activation checkpointing (SAC) to trade memory for
    # speed — skip the backward recompute. We have GPU-memory headroom at this batch.
    # mode "none" makes enable_selective_checkpoint() a no-op (minimal_v4_dit.py:1801),
    # so block activations are kept instead of recomputed.
    try:
        cfg.sac_config.mode = "none"
    except Exception:
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
        cfg.sac_config = SACConfig(mode="none")
    return instantiate(cfg)


# how many of the 17 input channels are the conditioning channel (rest = VAE latent)
COSMOS_LATENT_CHANNELS = 16
COSMOS_IN_CHANNELS = 17


def add_conditioning_channel(latent_B_C_T_H_W):
    """Append the video2world conditioning channel (zeros = unconditional / predict
    the whole clip, which is the FastWAM world-model objective)."""
    if latent_B_C_T_H_W.shape[1] >= COSMOS_IN_CHANNELS:
        return latent_B_C_T_H_W
    B, C, T, H, W = latent_B_C_T_H_W.shape
    cond = latent_B_C_T_H_W.new_zeros(B, COSMOS_IN_CHANNELS - C, T, H, W)
    return torch.cat([latent_B_C_T_H_W, cond], dim=1)


# --------------------------------------------------------------------------- #
# o0 (current-observation) conditioning — Cosmos video2world FRAME_REPLACE     #
# convention (video2world_model_rectified_flow.py:92-137, conditioner.py:151). #
# Used by the AGRA coupling's foresight pass; the default world-model path      #
# (add_conditioning_channel above) is UNCHANGED.                                #
# --------------------------------------------------------------------------- #

# Cosmos default near-clean noise level for the conditioning frame(s)
# (cosmos_policy_experiment_configs.py:357 uses 0; SFT_2B_RF.py uses 0.1). We use
# 0 -> the conditioning frame is treated as fully clean (timestep 0 = clean,
# 1 = pure noise in the rectified-flow / FastWAM convention).
COSMOS_CONDITIONAL_FRAME_TIMESTEP = 0.0


def build_o0_conditioned_input(noisy_latent_B_C_T_H_W, o0_latent, cond_frames: int = 1):
    """Build the Cosmos FRAME_REPLACE video2world conditioned net input.

    Mirrors ``Video2WorldModelRectifiedFlow.denoise`` (lines 92-106) +
    ``conditioner.py`` mask construction (line 151-155):
      1. REPLACE the first ``cond_frames`` latent frames of the noisy latent with
         the clean gt (``o0_latent``): ``xt = gt*mask + xt*(1-mask)``.
      2. Set the 17th (conditioning) channel = the mask (1 on conditioning frames,
         0 elsewhere) — instead of the all-zero channel ``add_conditioning_channel``
         appends for the unconditional world-model path.

    Args:
        noisy_latent_B_C_T_H_W: [B, 16, T, H, W] noisy VAE latent (16 channels).
        o0_latent: clean latent for the conditioning frame(s). Accepts
            [B, 16, T, H, W] (only the first ``cond_frames`` frames are read) or
            [B, 16, cond_frames, H, W] (just the conditioning frame(s)).
        cond_frames: number of leading latent frames treated as the observation.

    Returns:
        x_in [B, 17, T, H, W]: VAE latent with first frame(s) replaced by gt and
            the appended conditioning-mask channel set to 1 on those frames.
        mask_B_1_T_1_1 [B, 1, T, 1, 1]: per-frame conditioning mask (1=cond frame),
            for building the per-frame timesteps.
    """
    B, C, T, H, W = noisy_latent_B_C_T_H_W.shape
    if C != COSMOS_LATENT_CHANNELS:
        raise ValueError(
            f"build_o0_conditioned_input expects {COSMOS_LATENT_CHANNELS} latent channels, got {C}"
        )
    # per-frame conditioning mask [B, 1, T, 1, 1] (1 on the first cond_frames frames)
    mask_B_1_T_1_1 = noisy_latent_B_C_T_H_W.new_zeros(B, 1, T, 1, 1)
    mask_B_1_T_1_1[:, :, :cond_frames] = 1.0

    # broadcast the gt over the conditioning frame(s) (o0 may carry just those frames)
    gt = o0_latent.to(noisy_latent_B_C_T_H_W.dtype)
    if gt.shape[2] == cond_frames and T != cond_frames:
        gt_full = noisy_latent_B_C_T_H_W.clone()
        gt_full[:, :, :cond_frames] = gt
        gt = gt_full
    # frame-replace: first cond_frames = clean gt, rest = noisy
    m = mask_B_1_T_1_1  # broadcasts over C, H, W
    latent = gt * m + noisy_latent_B_C_T_H_W * (1 - m)

    # appended conditioning channel = the mask (1 on cond frames)
    cond_chan = mask_B_1_T_1_1.expand(B, COSMOS_IN_CHANNELS - C, T, H, W)
    x_in = torch.cat([latent, cond_chan], dim=1)  # [B, 17, T, H, W]
    return x_in, mask_B_1_T_1_1


def build_per_frame_timesteps(timesteps_B_T, mask_B_1_T_1_1,
                              conditional_frame_timestep: float = COSMOS_CONDITIONAL_FRAME_TIMESTEP):
    """Per-frame timesteps: conditioning frames at ``conditional_frame_timestep``
    (near-clean), the rest at the given ``timesteps_B_T``. Mirrors denoise() 108-121.

    Args:
        timesteps_B_T: [B, T] (or [B] broadcast) noise level for non-cond frames.
        mask_B_1_T_1_1: [B, 1, T, 1, 1] conditioning mask from build_o0_conditioned_input.
    Returns:
        [B, T] per-frame timesteps.
    """
    mask_B_T = mask_B_1_T_1_1[:, 0, :, 0, 0]  # [B, T]
    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1).expand_as(mask_B_T)
    cond_ts = torch.full_like(mask_B_T, float(conditional_frame_timestep))
    return cond_ts * mask_B_T + timesteps_B_T * (1 - mask_B_T)


def load_net_state_dict(net, ckpt_path: str, strict: bool = False):
    """Load a `net.`-prefixed Cosmos EMA checkpoint into a MiniTrainDIT."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = {}
    for k, v in ck.items():
        if not k.startswith("net."):
            continue
        kk = k[len("net."):]
        if kk.startswith("accum_"):  # training counters, not params
            continue
        sd[kk] = v
    missing, unexpected = net.load_state_dict(sd, strict=strict)
    miss = [m for m in missing if not m.endswith("_extra_state")]
    unexp = [u for u in unexpected if not u.endswith("_extra_state")]
    logger.info("Cosmos net load: %d tensors, missing(non-extra)=%d unexpected(non-extra)=%d",
                len(sd), len(miss), len(unexp))
    if miss:
        logger.warning("Cosmos net missing keys (sample): %s", miss[:8])
    if unexp:
        logger.warning("Cosmos net unexpected keys (sample): %s", unexp[:8])
    return net


class CosmosVideoExpert(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net  # MiniTrainDIT

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: Optional[str] = None,
        atten_backend: str = "torch",
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "CosmosVideoExpert":
        net = build_cosmos_2b_net(atten_backend=atten_backend)
        if ckpt_path:
            load_net_state_dict(net, ckpt_path)
        else:
            logger.info("CosmosVideoExpert: no ckpt_path, random init.")
        net = net.to(device=device, dtype=torch_dtype)
        return cls(net)

    @property
    def blocks(self):
        return self.net.blocks

    def prepare(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb, fps=None, padding_mask=None,
                o0_latent=None, cond_frames: int = 1):
        """Run MiniTrainDIT's pre-block-loop stages; return the per-stream MoT state.

        Mirrors MiniTrainDIT.forward (minimal_v4_dit.py:1712-1768) up to the block
        loop. Returns FLAT tokens so the MoT block fn can concat K/V across streams.

        If ``o0_latent`` is given, the first ``cond_frames`` latent frame(s) are
        conditioned on the current observation via the Cosmos video2world FRAME_REPLACE
        path (clean first frame + conditioning-mask channel=1 + per-frame timestep=0),
        so the world model + MoT action stream are grounded on the current image.
        Otherwise the unconditional world-model path (zero conditioning channel) is used.
        """
        net = self.net
        if o0_latent is not None:
            x_B_C_T_H_W, mask_B_1_T_1_1 = build_o0_conditioned_input(
                x_B_C_T_H_W, o0_latent, cond_frames=cond_frames
            )
            timesteps_B_T = build_per_frame_timesteps(timesteps_B_T, mask_B_1_T_1_1)
        else:
            x_B_C_T_H_W = add_conditioning_channel(x_B_C_T_H_W)  # 16 -> 17 channels
        if fps is None and getattr(net, "rope_enable_fps_modulation", True):
            base_fps = float(getattr(net.pos_embedder, "base_fps", 16))
            fps = torch.full((x_B_C_T_H_W.shape[0],), base_fps, device=x_B_C_T_H_W.device)
        if padding_mask is None and net.concat_padding_mask:
            # the model concatenates a padding-mask channel; an all-zero mask = "no pad"
            padding_mask = torch.zeros(
                x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[-2], x_B_C_T_H_W.shape[-1],
                device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype,
            )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos = net.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
        )
        if getattr(net, "use_crossattn_projection", False):
            crossattn_emb = net.crossattn_proj(crossattn_emb)
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_emb_B_T_D, adaln_lora_B_T_3D = net.t_embedder(timesteps_B_T)
        t_emb_B_T_D = net.t_embedding_norm(t_emb_B_T_D)

        B, T, H, W, D = x_B_T_H_W_D.shape
        tokens_B_S_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        return {
            "tokens": tokens_B_S_D,
            "rope": rope_emb_L_1_1_D,
            "t_emb": t_emb_B_T_D,
            "adaln_lora": adaln_lora_B_T_3D,
            "crossattn": crossattn_emb,
            "extra_pos": extra_pos,
            "THW": (T, H, W),
        }

    def finalize(self, tokens_B_S_D, t_emb_B_T_D, adaln_lora_B_T_3D, THW):
        """final_layer + unpatchify on the flat video tokens -> [B, C, T, H, W]."""
        T, H, W = THW
        x_B_T_H_W_D = rearrange(tokens_B_S_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_O = self.net.final_layer(x_B_T_H_W_D, t_emb_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        return self.net.unpatchify(x_B_T_H_W_O)

    def forward_standalone(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb,
                           feature_layer=-1, fps=None, padding_mask=None):
        """Run the FULL standard MiniTrainDIT forward (independent of the action
        stream) and also return a block's hidden state. Used by the cross-attention
        coupling: the video DiT runs on its own, the action DiT cross-attends to
        these features. Returns (pred_v [B,C,T,H,W], video_feats [B, Sv, D]).
        """
        net = self.net
        x_B_C_T_H_W = add_conditioning_channel(x_B_C_T_H_W)
        if fps is None and getattr(net, "rope_enable_fps_modulation", True):
            base_fps = float(getattr(net.pos_embedder, "base_fps", 16))
            fps = torch.full((x_B_C_T_H_W.shape[0],), base_fps, device=x_B_C_T_H_W.device)
        if padding_mask is None and net.concat_padding_mask:
            padding_mask = torch.zeros(
                x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[-2], x_B_C_T_H_W.shape[-1],
                device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype,
            )
        n_blocks = len(net.blocks)
        layer = feature_layer if feature_layer >= 0 else n_blocks + feature_layer
        pred_v, feats = net(
            x_B_C_T_H_W, timesteps_B_T, crossattn_emb, fps=fps, padding_mask=padding_mask,
            intermediate_feature_ids=[layer],
        )
        return pred_v, feats[0]  # video_feats [B, Sv, D]

    def forward_foresight(self, noise_latent_B_C_T_H_W, timesteps_B_T, crossattn_emb,
                          layers, o0_latent=None, cond_frames: int = 1,
                          fps=None, padding_mask=None):
        """Extract the "foresight": run the video DiT (optionally o0-conditioned) and
        return the hidden states of the requested ``layers`` (the multi-layer bridge
        for the AGRA action head). Used at FIXED high noise (tau_v=1, pure-noise
        ``noise_latent``) so the action head reads the model's high-noise dynamics
        prior conditioned on the current observation o0.

        Args:
            noise_latent_B_C_T_H_W: [B, 16, T, H, W] pure-noise latent (the tau_v=1 input).
            timesteps_B_T: [B] or [B, T] noise level. With o0_latent set, the
                conditioning frame(s) are overridden to a near-clean timestep
                (build_per_frame_timesteps); pass tau_v=1 for the non-cond frames.
            crossattn_emb: text (+proprio) context [B, N, crossattn_emb_channels].
            layers: list[int] of block indices whose hidden states to return.
            o0_latent: clean conditioning-frame latent (current observation). If None,
                runs unconditional (zero conditioning channel) like forward_standalone.
            cond_frames: number of leading latent frames used as the observation.
        Returns:
            list[Tensor [B, Sv, D]] of per-layer hidden states (order == ``layers``).
        """
        net = self.net
        if o0_latent is not None:
            x_B_C_T_H_W, mask_B_1_T_1_1 = build_o0_conditioned_input(
                noise_latent_B_C_T_H_W, o0_latent, cond_frames=cond_frames
            )
            timesteps_B_T = build_per_frame_timesteps(timesteps_B_T, mask_B_1_T_1_1)
        else:
            x_B_C_T_H_W = add_conditioning_channel(noise_latent_B_C_T_H_W)
        if fps is None and getattr(net, "rope_enable_fps_modulation", True):
            base_fps = float(getattr(net.pos_embedder, "base_fps", 16))
            fps = torch.full((x_B_C_T_H_W.shape[0],), base_fps, device=x_B_C_T_H_W.device)
        if padding_mask is None and net.concat_padding_mask:
            padding_mask = torch.zeros(
                x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[-2], x_B_C_T_H_W.shape[-1],
                device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype,
            )
        _pred, feats = net(
            x_B_C_T_H_W, timesteps_B_T, crossattn_emb, fps=fps, padding_mask=padding_mask,
            intermediate_feature_ids=list(layers),
        )
        return feats  # list of [B, Sv, D], one per requested layer
