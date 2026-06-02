"""Inspect how well the pretrained SANA-Video-2B checkpoint matches the model
arch we construct, for two candidate attention types. Prints matched/missing/
unexpected key counts so we pick the attn_type that best reuses pretrained weights."""
import sys
sys.path.insert(0, "/data/LFT-W02_data/junjie/VLA_WM/Sana")
import torch
from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo_2000M_P2_D20

PTH = "/data/LFT-W02_data/junjie/weights/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth"

payload = torch.load(PTH, map_location="cpu", weights_only=False)
state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
print(f"checkpoint keys: {len(state)}  (sample: {list(state)[:3]})")

COMMON = dict(
    in_channels=16, caption_channels=2304, model_max_length=300,
    learn_sigma=False, pred_sigma=False, mlp_ratio=3,
    mlp_acts=("silu", "silu", None), linear_head_dim=112, use_pe=True,
    pos_embed_type="wan_rope", qk_norm=True, cross_norm=True, y_norm=True,
    y_norm_scale_factor=0.01, class_dropout_prob=0.0, t_kernel_size=3,
)
CANDIDATES = {
    "LiteLAReLURope/GLUMBConvTemp": dict(attn_type="LiteLAReLURope", ffn_type="GLUMBConvTemp"),
    "chunkcausal/ChunkGLUMBConvTemp": dict(attn_type="chunkcausal", ffn_type="ChunkGLUMBConvTemp"),
}

for name, extra in CANDIDATES.items():
    try:
        m = SanaMSVideo_2000M_P2_D20(**COMMON, **extra)
        msd = m.state_dict()
        matched = sum(1 for k, v in state.items() if k in msd and msd[k].shape == v.shape)
        missing = [k for k in msd if k not in state]
        unexpected = [k for k in state if k not in msd]
        print(f"[{name}] model_keys={len(msd)} matched={matched} "
              f"missing(model-side)={len(missing)} unexpected(ckpt-side)={len(unexpected)}")
        del m, msd
    except Exception as e:
        print(f"[{name}] BUILD FAILED: {type(e).__name__}: {e}")
