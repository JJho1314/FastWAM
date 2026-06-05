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
    return instantiate(cfg)


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

    def prepare(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb, fps=None, padding_mask=None):
        """Run MiniTrainDIT's pre-block-loop stages; return the per-stream MoT state.

        Mirrors MiniTrainDIT.forward (minimal_v4_dit.py:1712-1768) up to the block
        loop. Returns FLAT tokens so the MoT block fn can concat K/V across streams.
        """
        net = self.net
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
