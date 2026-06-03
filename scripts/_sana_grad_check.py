"""GPU correctness check: build FastWAMSana with grad-checkpointing ON (as in
training) and verify training_loss is finite and gradients flow into BOTH the
action expert and the video expert (i.e. the cross-attention coupling survives
gradient checkpointing — the forward hook must capture a graph-connected tensor)."""
import sys
sys.path.insert(0, "/data/user/jhe724/workspace/Sana")
import torch
from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo_2000M_P2_D20
from diffusion.model.utils import set_grad_checkpoint
from fastwam.models.sana import SanaVideoExpert, SanaActionExpert, FastWAMSana

dev, dtype, L = "cuda", torch.bfloat16, 300
dit = SanaMSVideo_2000M_P2_D20(
    in_channels=16, caption_channels=2304, model_max_length=L, learn_sigma=False,
    pred_sigma=False, attn_type="LiteLAReLURope", ffn_type="GLUMBConvTemp", mlp_ratio=3,
    mlp_acts=("silu", "silu", None), linear_head_dim=112, use_pe=True,
    pos_embed_type="wan_rope", qk_norm=True, cross_norm=True, y_norm=True,
    y_norm_scale_factor=0.01, class_dropout_prob=0.0, t_kernel_size=3,
).to(dev, dtype)
set_grad_checkpoint(dit, gc_step=1)   # <-- checkpointing ON, like training
dit.train()
ve = SanaVideoExpert(dit, -1)
ae = SanaActionExpert(action_dim=7, hidden_size=1024, depth=6, num_heads=8,
                      video_feat_dim=ve.hidden_size, text_dim=2304).to(dev, dtype)

def fake_vae(name, vae, video, device):
    return torch.randn(video.shape[0], 16, 3, 28, 56, device=device, dtype=dtype)

m = FastWAMSana(ve, ae, None, vae_encode_fn=fake_vae, text_dim=2304, proprio_dim=8,
                device=dev, torch_dtype=dtype, train_video_expert=True)
m.proprio_encoder = m.proprio_encoder.to(dev, dtype)
m.train()

B = 1
s = dict(
    video=torch.randn(B, 3, 9, 224, 448, device=dev, dtype=dtype),
    context=torch.randn(B, L, 2304, device=dev, dtype=dtype),
    context_mask=torch.zeros(B, L, dtype=torch.bool, device=dev),
    action=torch.randn(B, 32, 7, device=dev, dtype=dtype),
    proprio=torch.randn(B, 32, 8, device=dev, dtype=dtype),
    action_is_pad=torch.zeros(B, 32, dtype=torch.bool, device=dev),
    image_is_pad=torch.zeros(B, 9, dtype=torch.bool, device=dev),
)
s["context_mask"][:, :20] = True

loss, d = m.training_loss(s)
print("loss:", {k: float(v) for k, v in d.items()})
assert torch.isfinite(loss), "LOSS NOT FINITE"
loss.backward()
ga = sum(p.grad.abs().sum().item() for p in m.action_expert.parameters() if p.grad is not None)
gv = sum(p.grad.abs().sum().item() for p in m.video_expert.parameters() if p.grad is not None)
print(f"grad: action={ga:.3e}  video={gv:.3e}")
print("RESULT:", "VIDEO GRAD FLOWS (coupling OK under checkpointing)" if gv > 0
      else "BUG: NO VIDEO GRAD — hook captured detached tensor under checkpointing")
