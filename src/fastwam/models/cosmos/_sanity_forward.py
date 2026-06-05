"""Sanity: instantiate Cosmos-Predict2.5-2B MiniTrainDIT, load the base/pre-trained
checkpoint, run one forward. Verifies the env + load API + forward shapes before
building the FastWAM MoT integration. Run on a GPU node.

Usage: COSMOS_PY _sanity_forward.py <ckpt.pt>
"""
import copy
import sys

import torch


def build_2b_net(atten_backend="torch"):
    from cosmos_predict2._src.predict2.configs.text2world.defaults.net import (
        COSMOS_V1_2B_NET_MININET,
    )
    from cosmos_predict2._src.imaginaire.lazy_config import instantiate

    cfg = copy.deepcopy(COSMOS_V1_2B_NET_MININET)
    cfg.atten_backend = atten_backend  # SDPA backend so MoT masking + no-CP works
    cfg.in_channels = 17  # base ckpt: 16 VAE latent + 1 conditioning channel
    net = instantiate(cfg)
    return net


def load_net_ckpt(net, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = {}
    for k, v in ck.items():
        if not k.startswith("net."):
            continue
        kk = k[len("net."):]
        if kk.startswith("accum_"):
            continue
        sd[kk] = v
    missing, unexpected = net.load_state_dict(sd, strict=False)
    miss = [m for m in missing if not m.endswith("_extra_state")]
    unexp = [u for u in unexpected if not u.endswith("_extra_state")]
    print(f"load: matched={len(sd) - len(unexpected)} missing(non-extra)={len(miss)} "
          f"unexpected(non-extra)={len(unexp)}")
    if miss[:8]:
        print("  missing sample:", miss[:8])
    if unexp[:8]:
        print("  unexpected sample:", unexp[:8])
    return net


def main():
    ckpt = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "torch", torch.__version__)

    net = build_2b_net("torch")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"MiniTrainDIT-2B params: {n_params / 1e9:.3f}B  blocks={len(net.blocks)}")

    load_net_ckpt(net, ckpt)
    net = net.to(dev).to(torch.bfloat16).eval()

    # dummy forward: latent [B, C=16, T, H, W], timesteps [B, T], qwen crossattn [B, N, 1024]
    B, C, T, H, W = 1, 17, 4, 16, 16  # 16 latent + 1 conditioning channel
    from cosmos_predict2._src.predict2.conditioner import DataType
    x = torch.randn(B, C, T, H, W, device=dev, dtype=torch.bfloat16)
    t = torch.full((B, T), 500.0, device=dev, dtype=torch.bfloat16)
    crossattn = torch.randn(B, 16, 1024, device=dev, dtype=torch.bfloat16)
    pad = torch.zeros(B, 1, H, W, device=dev, dtype=torch.bfloat16)  # concat_padding_mask=True needs it
    # fps required when rope_enable_fps_modulation (else the rope takes the image
    # path which asserts T==1); base_fps gives neutral temporal scaling.
    base_fps = float(getattr(net.pos_embedder, "base_fps", 16))
    fps = torch.full((B,), base_fps, device=dev)
    with torch.no_grad():
        out = net(x, t, crossattn, fps=fps, padding_mask=pad, data_type=DataType.VIDEO)
    print("forward OK. out type:", type(out))
    if torch.is_tensor(out):
        print("out shape:", tuple(out.shape), "dtype:", out.dtype)
    print("SANITY-FORWARD-PASSED")


if __name__ == "__main__":
    main()
