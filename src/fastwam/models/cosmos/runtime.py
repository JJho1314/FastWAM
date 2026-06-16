"""Hydra factory for the Cosmos-Predict2.5 variant of FastWAM.

Builds: Cosmos video DiT (MiniTrainDIT-2B, loaded from the base/pre-trained EMA
ckpt) + a Cosmos-block action expert (copy-initialised from the video DiT) coupled
by MoT, plus the Cosmos Wan2.1 VAE tokenizer. Cosmos deps are imported lazily so
the rest of fastwam still imports without the cosmos env.
"""
from __future__ import annotations

import torch

from fastwam.utils.logging_config import get_logger
from .video_expert import CosmosVideoExpert
from .action_expert import CosmosActionExpert
from .foresight_action_head import ForesightActionHead
from .fastwam_cosmos import FastWAMCosmos

logger = get_logger(__name__)


def _cosmos_vae_encode(name, vae, video, device):
    # video: [B, 3, T, H, W] in [-1, 1] -> [B, 16, T/4, H/8, W/8].
    # Call the inner WanVAE.encode (applies the Wan2.1 `scale` normalisation, same
    # as the SANA path) and SKIP the Wan2pt1VAEInterface's extra img/video mean-std
    # normalisation, whose constants live in s3 files we don't have. The DiT adapts
    # to the (scale-only) latent scale during fine-tuning. TODO: recover the Cosmos
    # mean/std for an exact match.
    video = video.to(device)
    # The VAE (Wan2pt1VAEInterface) is frozen and NOT a registered nn.Module submodule,
    # so accelerate/FSDP never relocates it to each rank's GPU — it stays on its load
    # device (cuda:0). Under multi-GPU, rank R's input is on cuda:R while the VAE conv
    # weights sit on cuda:0 -> cross-device conv error. `vae.model` is a WanVAE wrapper
    # (no `.to`); the real nn.Module is `vae.model.model`, and encode() also reads the
    # mean/std/scale tensors. Relocate them all onto the input's device once.
    wanvae = vae.model
    dev = video.device
    if next(wanvae.model.parameters()).device != dev:
        wanvae.model.to(dev)
        wanvae.device = dev
        wanvae.mean = wanvae.mean.to(dev)
        wanvae.std = wanvae.std.to(dev)
        wanvae.scale = [wanvae.mean, 1.0 / wanvae.std]
    return wanvae.encode(video)


def create_fastwam_cosmos(
    video_dit_pretrained_path: str,
    vae=None,
    action_dim: int = 7,
    proprio_dim: int | None = None,
    crossattn_dim: int = 1024,
    coupling: str = "mot",
    mot_bidirectional: bool = False,
    feature_layer: int = -1,
    atten_backend: str = "torch",
    train_video_expert: bool = True,
    # --- AGRA action-head (foresight cross-attention) hyperparameters ---
    action_horizon: int | None = None,   # K (chunk length); informational, head is length-agnostic
    agra_num_layers: int = 8,
    agra_hidden: int = 1024,
    agra_num_heads: int = 32,
    agra_crossattn_dim: int = 2048,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    # Pin the CUDA *current device* to this rank's GPU BEFORE building any submodule,
    # so that anything created with a bare "cuda" lands on this rank's GPU and not on
    # cuda:0. In particular the Wan2.1 VAE (WanVAE hardcodes device="cuda") would
    # otherwise build on cuda:0 for EVERY rank, leaving a ~1.4GB allocation per rank on
    # GPU0 (~10GB wasted on rank 0's GPU) until first use. accelerate sets the same
    # device later during prepare(), so this is a harmless early no-op for rank 0.
    if isinstance(device, str) and device.startswith("cuda:") and torch.cuda.is_available():
        torch.cuda.set_device(int(device.split(":", 1)[1]))

    # ---- video DiT (MiniTrainDIT-2B) ----
    video_expert = CosmosVideoExpert.from_pretrained(
        ckpt_path=video_dit_pretrained_path,
        atten_backend=atten_backend,
        device=device,
        torch_dtype=model_dtype,
    )
    net = video_expert.net
    model_channels = int(net.model_channels) if hasattr(net, "model_channels") else 2048
    num_blocks = len(net.blocks)
    num_heads = int(net.blocks[0].self_attn.n_heads)

    # ---- action DiT ----
    if coupling == "agra":
        # AGRA: a standalone 8-layer cross-attention DiT (ForesightActionHead) that
        # reads the video DiT's multi-layer foresight. NOT a Cosmos-block expert and
        # NOT copy-init from the video DiT (different depth/width). Requires proprio
        # (the prepended state token s0).
        if proprio_dim is None:
            raise ValueError("coupling=agra requires proprio_dim (the prepended s0 token).")
        action_expert = ForesightActionHead(
            action_dim=action_dim,
            proprio_dim=int(proprio_dim),
            num_layers=int(agra_num_layers),
            hidden=int(agra_hidden),
            num_heads=int(agra_num_heads),
            crossattn_dim=int(agra_crossattn_dim),
            action_horizon=action_horizon,
        )
        action_expert = action_expert.to(device=device, dtype=model_dtype)
    else:
        # mot / cross_attn: Cosmos-block action expert (copy-init from the video DiT).
        action_expert = CosmosActionExpert(
            action_dim=action_dim,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            crossattn_emb_channels=crossattn_dim,
        )
        action_expert.copy_init_from_video(net)
        action_expert = action_expert.to(device=device, dtype=model_dtype)

    # ---- Cosmos Wan2.1 VAE tokenizer ----
    vae_model = None
    if vae is not None:
        from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        vae_pth = vae["vae_pth"] if isinstance(vae, dict) else getattr(vae, "vae_pth", vae)
        # load_mean_std=False: the extra Cosmos mean/std files are s3-only; we use
        # the inner WanVAE.encode (scale-normalised) in _cosmos_vae_encode instead.
        vae_model = Wan2pt1VAEInterface(vae_pth=str(vae_pth), load_mean_std=False)

    video_scheduler = video_scheduler or {}
    action_scheduler = action_scheduler or {}
    loss = loss or {}
    model = FastWAMCosmos(
        video_expert=video_expert,
        action_expert=action_expert,
        vae=vae_model,
        vae_encode_fn=_cosmos_vae_encode,
        crossattn_dim=crossattn_dim,
        coupling=coupling,
        mot_bidirectional=mot_bidirectional,
        feature_layer=feature_layer,
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        video_train_shift=float(video_scheduler.get("train_shift", 3.0)),
        action_train_shift=float(action_scheduler.get("train_shift", 3.0)),
        num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    if model.proprio_encoder is not None:
        model.proprio_encoder = model.proprio_encoder.to(device).to(model_dtype)
    n = sum(p.numel() for p in model.dit.parameters())
    logger.info("FastWAMCosmos trainable (dit) params: %.3fB", n / 1e9)
    return model
