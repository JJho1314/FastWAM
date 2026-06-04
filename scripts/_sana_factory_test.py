"""End-to-end factory test: build FastWAMSana via create_fastwam_sana with the
ActionDiT-based action expert (loaded from the interpolated backbone), real
WanVAE, random video DiT (skip 2B load), and run training_loss + backward."""
import sys
sys.path.insert(0, "/data/LFT-W02_data/junjie/VLA_WM/Sana")
import torch
from fastwam.runtime import create_fastwam_sana

dev, dtype = "cuda", torch.bfloat16
model = create_fastwam_sana(
    sana_repo_path="/data/LFT-W02_data/junjie/VLA_WM/Sana",
    video_dit=dict(model_name="SanaMSVideo_2000M_P2_D20", in_channels=16, caption_channels=2304,
                   model_max_length=300, attn_type="LiteLAReLURope", ffn_type="GLUMBConvTemp",
                   mlp_ratio=3, mlp_acts=["silu", "silu", None], linear_head_dim=112, use_pe=True,
                   pos_embed_type="wan_rope", qk_norm=True, cross_norm=True, y_norm=True,
                   y_norm_scale_factor=0.01, class_dropout_prob=0.0, t_kernel_size=3),
    action_dit=dict(hidden_dim=1024, ffn_dim=4096, text_dim=4096, freq_dim=256, eps=1e-6,
                    num_heads=24, attn_head_dim=128, num_layers=30, use_gradient_checkpointing=False),
    vae=dict(z_dim=16, vae_pth="/data/LFT-W02_data/junjie/weights/SANA-Video_2B_480p/vae/Wan2.1_VAE.pth"),
    video_dit_pretrained_path=None,  # skip 2B load (random video) for a fast test
    action_dit_pretrained_path="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
    action_dim=7, proprio_dim=8, text_dim=2304,
    train_video_expert=False,  # freeze random video so it fits one GPU
    grad_checkpointing=True, fp32_attention=True,
    video_scheduler=dict(train_shift=5.0, num_train_timesteps=1000),
    action_scheduler=dict(train_shift=5.0, num_train_timesteps=1000),
    loss=dict(lambda_video=1.0, lambda_action=1.0),
    model_dtype=dtype, device=dev,
)
model.train()
n_tr = sum(p.numel() for p in model.dit.parameters() if p.requires_grad)
print(f"trainable dit params: {n_tr/1e9:.3f}B")

B, L = 1, 300
s = dict(
    video=torch.randn(B, 3, 9, 224, 448, device=dev, dtype=dtype).clamp(-1, 1),
    context=torch.randn(B, L, 2304, device=dev, dtype=dtype),
    context_mask=torch.zeros(B, L, dtype=torch.bool, device=dev),
    action=torch.randn(B, 32, 7, device=dev, dtype=dtype),
    proprio=torch.randn(B, 32, 8, device=dev, dtype=dtype),
    action_is_pad=torch.zeros(B, 32, dtype=torch.bool, device=dev),
    image_is_pad=torch.zeros(B, 9, dtype=torch.bool, device=dev),
)
s["context_mask"][:, :20] = True

with torch.autocast("cuda", dtype=dtype):
    loss, d = model.training_loss(s)
print("loss:", {k: float(v) for k, v in d.items()})
assert torch.isfinite(loss), "loss not finite"
loss.backward()
g_act = sum(p.grad.abs().sum().item() for p in model.action_expert.parameters() if p.grad is not None)
print(f"grad -> action_expert: {g_act:.3e}")
print("FACTORY E2E OK" if g_act > 0 else "FAIL")
