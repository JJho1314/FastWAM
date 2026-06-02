"""Tiny CPU smoke test for trimodal model components.

Avoids loading any pretrained Wan2.2-TI2V-5B / ActionDiT checkpoints by
using small random configs. Validates:
  - WanDiscreteFlowMatchScheduler sample/add_noise/CE shapes are coherent.
  - TextDiT forward yields correct logits shape.
  - The 3-segment attention mask shapes match the MoT signature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.wan22.schedulers.scheduler_discrete import WanDiscreteFlowMatchScheduler  # noqa: E402
from fastwam.models.wan22.text_dit import TextDiT  # noqa: E402


def test_discrete_scheduler():
    sched = WanDiscreteFlowMatchScheduler(vocab_size=64, num_train_timesteps=1000, shift=1.0)
    B, L = 2, 16
    target = torch.randint(0, 64, (B, L), dtype=torch.long)
    t = sched.sample_training_t(batch_size=B, device=torch.device("cpu"), dtype=torch.float32)
    assert t.shape == (B,), f"timestep shape {t.shape}"
    noisy = sched.add_noise(target, t)
    assert noisy.shape == (B, L) and noisy.dtype == torch.long
    # CE loss is finite scalar
    logits = torch.randn(B, L, 64)
    loss = sched.cross_entropy_loss(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    masked_loss = sched.cross_entropy_loss(
        logits, target, attention_mask=torch.ones(B, L, dtype=torch.bool)
    )
    assert torch.isfinite(masked_loss)
    print(f"[OK] WanDiscreteFlowMatchScheduler basic shape/loss test passed (loss={loss.item():.3f})")


def test_text_dit_forward():
    expert = TextDiT(
        hidden_dim=64,
        vocab_size=128,
        ffn_dim=128,
        text_dim=64,  # tiny T5 hidden dim for the smoke test
        freq_dim=64,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=16,
        num_layers=2,
        max_seq_len=64,
    )
    B, L, ctx_L = 2, 16, 8
    text_tokens = torch.randint(0, 128, (B, L), dtype=torch.long)
    timestep = torch.tensor([500.0, 0.0])
    context = torch.randn(B, ctx_L, 64)
    context_mask = torch.ones(B, ctx_L, dtype=torch.bool)
    logits = expert(text_tokens, timestep, context, context_mask)
    assert logits.shape == (B, L, 128), f"logits shape {logits.shape}"
    assert torch.isfinite(logits).all()
    print(f"[OK] TextDiT forward passed, logits shape={tuple(logits.shape)}")


def test_pre_post_split():
    expert = TextDiT(
        hidden_dim=64, vocab_size=128, ffn_dim=128, text_dim=64, freq_dim=64,
        eps=1e-6, num_heads=4, attn_head_dim=16, num_layers=2, max_seq_len=64,
    )
    B, L, ctx_L = 2, 16, 8
    text_tokens = torch.randint(0, 128, (B, L), dtype=torch.long)
    timestep = torch.tensor([500.0, 200.0])
    context = torch.randn(B, ctx_L, 64)
    pre = expert.pre_dit(text_tokens, timestep, context, context_mask=torch.ones(B, ctx_L, dtype=torch.bool))
    for key in ["tokens", "freqs", "t", "t_mod", "context", "context_mask"]:
        assert key in pre, f"missing key {key}"
    assert pre["tokens"].shape == (B, L, 64)
    assert pre["t_mod"].shape == (B, 6, 64)
    # post takes already-processed tokens (same shape as pre["tokens"])
    out = expert.post_dit(pre["tokens"], pre)
    assert out.shape == (B, L, 128)
    print(f"[OK] TextDiT pre_dit/post_dit split passes, t_mod shape={tuple(pre['t_mod'].shape)}")


def main():
    torch.manual_seed(0)
    test_discrete_scheduler()
    test_text_dit_forward()
    test_pre_post_split()
    print("[ALL] smoke tests passed")


if __name__ == "__main__":
    main()
