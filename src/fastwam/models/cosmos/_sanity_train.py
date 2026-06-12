"""Sanity: one full training step of FastWAM-Cosmos on a dummy LIBERO-shaped batch.
Verifies the VAE encode + build_inputs + MoT + flow-matching loss + backward.

Usage: COSMOS_PY _sanity_train.py <video_ckpt.pt> <tokenizer.pth>
"""
import sys
import torch


def main():
    ckpt, tokenizer = sys.argv[1], sys.argv[2]
    coupling = sys.argv[3] if len(sys.argv) > 3 else "mot"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from fastwam.models.cosmos.runtime import create_fastwam_cosmos
    print("coupling:", coupling)

    model = create_fastwam_cosmos(
        video_dit_pretrained_path=ckpt,
        vae={"vae_pth": tokenizer},
        action_dim=7, proprio_dim=8, crossattn_dim=1024, coupling=coupling,
        video_scheduler={"train_shift": 3.0, "num_train_timesteps": 1000},
        action_scheduler={"train_shift": 3.0, "num_train_timesteps": 1000},
        loss={"lambda_video": 1.0, "lambda_action": 1.0},
        model_dtype=torch.bfloat16, device=dev,
    )
    model.train()

    B, Tvid, H, W = 1, 9, 64, 64  # 9 keyframes (LIBERO freq_ratio=4), small spatial
    sample = {
        "video": (torch.rand(B, 3, Tvid, H, W, device=dev, dtype=torch.bfloat16) * 2 - 1),
        "action": torch.randn(B, 32, 7, device=dev, dtype=torch.bfloat16),
        "context": torch.randn(B, 16, 3584, device=dev, dtype=torch.bfloat16),  # Qwen dim
        "proprio": torch.randn(B, 32, 8, device=dev, dtype=torch.bfloat16),
        "action_is_pad": torch.zeros(B, 32, dtype=torch.bool, device=dev),
        "image_is_pad": torch.zeros(B, Tvid, dtype=torch.bool, device=dev),
    }
    loss, ld = model.training_loss(sample)
    print("loss:", float(loss), {k: float(v) for k, v in ld.items()})
    assert torch.isfinite(loss), "non-finite loss"
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.dit.parameters() if p.grad is not None)
    print("grad-sum:", g, "(finite)" if g == g else "(NaN)")
    print("SANITY-TRAIN-PASSED")


if __name__ == "__main__":
    main()
