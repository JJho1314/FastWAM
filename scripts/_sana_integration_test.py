"""Integration smoke test for FastWAMSana.training_loss with the REAL SANA-Video
DiT (random weights) + the action expert, on synthetic LIBERO-shaped data.
Validates: cross-attention feature hook, both flow losses, and backward."""
import sys
sys.path.insert(0, "/data/LFT-W02_data/junjie/VLA_WM/Sana")

import torch
from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo_2000M_P2_D20

from fastwam.models.sana import SanaVideoExpert, SanaActionExpert, FastWAMSana

dev, dtype = "cuda", torch.bfloat16
L = 300  # text length == model_max_length

sana_dit = SanaMSVideo_2000M_P2_D20(
    in_channels=16, caption_channels=2304, model_max_length=L,
    learn_sigma=False, pred_sigma=False,
    attn_type="LiteLAReLURope", ffn_type="GLUMBConvTemp", mlp_ratio=3,
    mlp_acts=("silu", "silu", None), linear_head_dim=112, use_pe=True,
    pos_embed_type="wan_rope", qk_norm=True, cross_norm=True,
    class_dropout_prob=0.0, t_kernel_size=3,
).to(dev).to(dtype)

video_expert = SanaVideoExpert(sana_dit, feature_layer=-1)
action_expert = SanaActionExpert(
    action_dim=7, hidden_size=1024, depth=6, num_heads=8,
    video_feat_dim=video_expert.hidden_size, text_dim=2304, max_action_len=64,
).to(dev).to(dtype)


def fake_vae_encode(name, vae, video, device):
    B = video.shape[0]
    return torch.randn(B, 16, 3, 28, 56, device=device, dtype=dtype)


model = FastWAMSana(
    video_expert=video_expert, action_expert=action_expert,
    vae=None, vae_name="WanVAE", vae_encode_fn=fake_vae_encode,
    text_dim=2304, proprio_dim=None, device=dev, torch_dtype=dtype,
    train_video_expert=True, loss_lambda_video=1.0, loss_lambda_action=1.0,
).to(dev)

B = 1
sample = {
    "video": torch.randn(B, 3, 9, 224, 448, device=dev, dtype=dtype),
    "context": torch.randn(B, L, 2304, device=dev, dtype=dtype),
    "context_mask": torch.zeros(B, L, dtype=torch.bool, device=dev),
    "action": torch.randn(B, 32, 7, device=dev, dtype=dtype),
    "action_is_pad": torch.zeros(B, 32, dtype=torch.bool, device=dev),
    "image_is_pad": torch.zeros(B, 9, dtype=torch.bool, device=dev),
}
sample["context_mask"][:, :20] = True

loss, d = model.training_loss(sample)
print("loss:", float(loss), {k: float(v) for k, v in d.items()})
loss.backward()

n_train = sum(p.numel() for p in model.dit.parameters() if p.requires_grad)
g_action = sum(p.grad.abs().sum().item() for p in model.action_expert.parameters() if p.grad is not None)
g_video = sum(p.grad.abs().sum().item() for p in model.video_expert.parameters() if p.grad is not None)
print(f"trainable dit params: {n_train/1e9:.3f}B")
print(f"grad flowing -> action_expert: {g_action:.3e}, video_expert: {g_video:.3e}")
assert g_action > 0, "no grad into action expert"
assert g_video > 0, "no grad into video expert (cross-attn coupling broken)"
print("INTEGRATION OK")
