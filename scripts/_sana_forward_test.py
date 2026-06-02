"""Smoke test: construct SANA-Video-2B DiT and run a forward on a LIBERO-shaped
latent (random weights). Validates the architecture/env before wiring weights."""
import sys
sys.path.insert(0, "/data/LFT-W02_data/junjie/VLA_WM/Sana")

import torch
from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo_2000M_P2_D20

dev = "cuda"
dtype = torch.bfloat16

model = SanaMSVideo_2000M_P2_D20(
    in_channels=16,
    caption_channels=2304,
    model_max_length=300,
    learn_sigma=False,
    pred_sigma=False,
    attn_type="LiteLAReLURope",
    ffn_type="GLUMBConvTemp",
    mlp_ratio=3,
    mlp_acts=("silu", "silu", None),
    linear_head_dim=112,
    use_pe=True,
    pos_embed_type="wan_rope",
    qk_norm=True,
    cross_norm=True,
    class_dropout_prob=0.0,
    t_kernel_size=3,
).to(dev).to(dtype)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"model params: {n_params/1e9:.3f}B  hidden={model.hidden_size} out_ch={model.out_channels}")

B, C, T, H, W = 1, 16, 3, 28, 56
x = torch.randn(B, C, T, H, W, device=dev, dtype=dtype)
timestep = torch.randint(0, 1000, (B,), device=dev).float()
L = 300
y = torch.randn(B, 1, L, 2304, device=dev, dtype=dtype)
mask = torch.zeros(B, L, device=dev, dtype=torch.long)
mask[:, :20] = 1  # 20 valid text tokens

with torch.no_grad():
    out = model(x, timestep, y, mask=mask)
print("forward OK, output shape:", tuple(out.shape))
assert tuple(out.shape) == (B, C, T, H, W), out.shape
print("SHAPE MATCH OK")
