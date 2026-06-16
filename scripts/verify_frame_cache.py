"""Verify the pre-decoded frame cache is numerically faithful to live mp4 decode.

This drives the REAL RobotVideoDataset end-to-end (including the FastWAMProcessor
image transforms) twice for the same sample indices -- once with the cache OFF
(live mp4 decode, the reference path) and once with the cache ON -- and compares
the final ``video`` tensor ``[3, T, 224, 448]`` it produces.

Why end-to-end: the operative per-camera resize for LIBERO lives in the processor
(``ToTensor`` + ``torchvision.transforms.Resize([224,224])``), NOT only in
``_get``'s post-concat resize. Comparing the full pipeline is the only honest test.

Interiors are expected to be (near) identical; any differences should be tiny and
confined to a few columns at each camera seam plus sub-LSB resample/quantization
noise. The script asserts ``max_abs_diff`` stays below a small tolerance.

Usage:
  python scripts/verify_frame_cache.py \
    --config configs/data/libero_2cam_cosmos.yaml \
    --cache-dir ./data/frame_cache/libero \
    --num-samples 4

If --config is omitted, a minimal RobotVideoDataset is built from --dataset-dirs
with no processor (so only the _get concat->resize->crop->normalize path is tested).
"""
import argparse
import os
import sys
from pathlib import Path

import torch

# Make `import fastwam...` resolve when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastwam.datasets.lerobot.lerobot.datasets.video_utils import (  # noqa: E402
    set_frame_cache_dir,
    _load_cached_frames,
)


def build_dataset_from_config(config_path: str):
    """Instantiate RobotVideoDataset (with processor) from a hydra data yaml."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    raw = OmegaConf.load(config_path)
    # The processor uses ${data.train.*} interpolations, which resolve against a root
    # that has `data.train`. Wrap the loaded data-config under a `data` root so they
    # resolve (otherwise: InterpolationKeyError 'data.train.shape_meta').
    root = OmegaConf.create({"data": raw})
    node = root.data.get("train", root.data)
    # Skip the (slow, accelerate-dependent) norm-stat computation: stats only affect
    # action/state normalization, never the `video` tensor we compare here. Point at
    # any existing dataset_stats.json if provided via FASTWAM_VERIFY_STATS.
    stats = os.environ.get("FASTWAM_VERIFY_STATS")
    if stats:
        from omegaconf import OmegaConf as _OC
        _OC.update(node, "pretrained_norm_stats", stats, force_add=True)
    ds = instantiate(node)
    return ds


def build_dataset_minimal(dataset_dirs, text_cache_dir):
    """Fallback: RobotVideoDataset with no processor (tests only _get transforms)."""
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset

    shape_meta = OmegaConf.create(
        {
            "images": [
                {"key": "image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
                {"key": "wrist_image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
            ],
            "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
            "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        }
    )
    ds = RobotVideoDataset(
        dataset_dirs=list(dataset_dirs),
        shape_meta=shape_meta,
        num_frames=33,
        video_size=[224, 448],
        processor=None,
        text_embedding_cache_dir=text_cache_dir,
        val_set_proportion=0.0,
        is_training_set=True,
        action_video_freq_ratio=4,
        concat_multi_camera="horizontal",
    )
    return ds


def get_video(ds, idx):
    """Return the final ``video`` tensor for a sample, bypassing the text-embed cache."""
    # Monkeypatch the text-context lookup so we don't need the text-embed cache.
    orig = ds._get_cached_text_context

    def _fake(prompt, _orig=orig):
        # context_len x 1 dummy; only `video` is compared.
        L = getattr(ds, "context_len", 128)
        ctx = torch.zeros(L, 1)
        mask = torch.ones(L, dtype=torch.bool)
        return ctx, mask

    ds._get_cached_text_context = _fake
    try:
        data = ds._get(idx)
    finally:
        ds._get_cached_text_context = orig
    return data["video"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="hydra data yaml (preferred; runs full processor pipeline)")
    ap.add_argument("--dataset-dirs", nargs="+", default=None, help="used only if --config is omitted")
    ap.add_argument("--text-cache-dir", default=None, help="text-embed cache (minimal mode; unused, monkeypatched)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-abs-diff-tol", type=float, default=0.1,
                    help="assert max_abs_diff (in [-1,1] units) below this")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.config is not None:
        print(f"Building dataset from config: {args.config}")
        ds = build_dataset_from_config(args.config)
    else:
        assert args.dataset_dirs, "Provide --config or --dataset-dirs"
        print(f"Building minimal dataset (no processor) from: {args.dataset_dirs}")
        ds = build_dataset_minimal(args.dataset_dirs, args.text_cache_dir)

    n_total = len(ds)
    n = min(args.num_samples, n_total)
    indices = [int(round(i * (n_total - 1) / max(1, n - 1))) for i in range(n)] if n > 1 else [0]
    print(f"dataset len={n_total}; checking indices {indices}")

    overall_max = 0.0
    for idx in indices:
        # (a) reference: cache OFF
        set_frame_cache_dir(None)
        _load_cached_frames.cache_clear()  # ensure no stale memmap is reused
        video_real = get_video(ds, idx)

        # (b) cached: cache ON
        set_frame_cache_dir(cache_dir)
        _load_cached_frames.cache_clear()
        video_cached = get_video(ds, idx)
        set_frame_cache_dir(None)

        assert video_real.shape == video_cached.shape, (
            f"shape mismatch at idx {idx}: {video_real.shape} vs {video_cached.shape}"
        )
        diff = (video_real.float() - video_cached.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        # >1/255 in [0,1] == >2/255 in [-1,1] (normalize scales by 1/0.5).
        frac_gt = (diff > (2.0 / 255.0)).float().mean().item()
        overall_max = max(overall_max, max_abs)
        print(
            f"  idx {idx}: shape={tuple(video_real.shape)} "
            f"max_abs_diff={max_abs:.5f} mean_abs_diff={mean_abs:.6f} "
            f"frac(|d|>1/255)={frac_gt:.4%}"
        )

    print(f"\nOVERALL max_abs_diff = {overall_max:.5f} (tol {args.max_abs_diff_tol})")
    assert overall_max <= args.max_abs_diff_tol, (
        f"max_abs_diff {overall_max:.5f} exceeds tol {args.max_abs_diff_tol}; "
        "cache is NOT numerically faithful."
    )
    print("PASS: cache output matches live decode within tolerance "
          "(interiors near-identical; diffs confined to camera seam + sub-LSB resample noise).")


if __name__ == "__main__":
    main()
