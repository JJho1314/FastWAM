"""Minimal repro of the SANA LiteLA 'misaligned address' at per-GPU batch>2.
Run one (B, FP32) case per process (a CUDA misaligned-address poisons the
context, so cases must be isolated). Controlled by env B and FP32."""
import os, sys
sys.path.insert(0, "/data/user/jhe724/workspace/Sana")
import torch
from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo_2000M_P2_D20
from diffusion.model.utils import set_fp32_attention

B = int(os.environ.get("B", "8"))
FP32 = os.environ.get("FP32", "1") == "1"
CONTIG = os.environ.get("CONTIG", "0") == "1"

dit = SanaMSVideo_2000M_P2_D20(
    in_channels=16, caption_channels=2304, model_max_length=300, learn_sigma=False,
    pred_sigma=False, attn_type="LiteLAReLURope", ffn_type="GLUMBConvTemp", mlp_ratio=3,
    mlp_acts=("silu", "silu", None), linear_head_dim=112, use_pe=True,
    pos_embed_type="wan_rope", qk_norm=True, cross_norm=True, y_norm=True,
    y_norm_scale_factor=0.01, class_dropout_prob=0.0, t_kernel_size=3,
).cuda().to(torch.bfloat16)
if FP32:
    set_fp32_attention(dit)

CKPT = os.environ.get("CKPT", "1") == "1"
if CKPT:
    from diffusion.model.utils import set_grad_checkpoint
    set_grad_checkpoint(dit, gc_step=1)
dit.train()

x = torch.randn(B, 16, 3, 28, 56, device="cuda", dtype=torch.bfloat16)
t = torch.randint(0, 1000, (B,), device="cuda").float()
y = torch.randn(B, 1, 300, 2304, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(B, 300, dtype=torch.long, device="cuda")
tag = f"B={B} FP32={FP32} CKPT={CKPT}"
try:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = dit(x, t, y, mask=mask)
    out.float().sum().backward()  # mirror training: forward(+ckpt) + backward
    torch.cuda.synchronize()
    print(f"RESULT {tag}: OK out={tuple(out.shape)}")
except Exception as e:
    print(f"RESULT {tag}: FAIL {type(e).__name__}: {str(e)[:120]}")
