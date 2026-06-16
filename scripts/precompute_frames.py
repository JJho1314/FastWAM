"""Pre-decode every episode/camera mp4 into a resized uint8 frame cache.

This eliminates the runtime mp4-decode bottleneck in RobotVideoDataset. For each
episode/camera video we decode ALL frames once, resize each frame from the native
512x512 down to the per-camera 224x224 target, and store a single uint8 array
``[num_frames, 3, 224, 224]`` plus the video fps, at the cache path produced by
``frame_cache_path`` (the SAME helper the runtime decode uses, so keys match).

IMPORTANT numerical subtlety (read before changing the resize):
  At runtime, RobotVideoDataset concatenates the per-camera frames horizontally
  into a [T, 3, 512, num_cam*512] uint8 image and then resizes the WHOLE concat
  to [224, num_cam*448] (i.e. 512 -> 224 per camera) with BICUBIC + antialias=True
  on the UINT8 tensor. So here we must also resize the UINT8 per-camera frame
  512 -> 224 with BICUBIC + antialias=True. Doing it per-camera on uint8 makes the
  camera interiors match the runtime concat-resize bit-for-bit; only ~1-2 columns
  at each camera seam differ (the concat resize can let a camera's edge pixels
  bleed across the seam). This tiny, expected seam difference is verified by
  scripts/verify_frame_cache.py.

Usage:
  python scripts/precompute_frames.py \
    --dataset-dirs ./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot ... \
    --cache-dir ./data/frame_cache/libero \
    --target 224 \
    --num-workers 8

Resumable: existing cache files are skipped. Safe to re-run.
"""
import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

# Make `import fastwam...` resolve when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Single source of truth for the cache key (shared with the runtime decode), plus
# the torchcodec decoder used to read all frames.
from fastwam.datasets.lerobot.lerobot.datasets.video_utils import (  # noqa: E402
    frame_cache_path,
    frame_cache_meta_path,
    decode_video_frames_torchcodec,
)

DEFAULT_DATASET_DIRS = [
    "./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot",
]

# Default video path template used by lerobot datasets (see datasets/utils.py).
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def load_info(ds_dir: Path) -> dict:
    with open(ds_dir / "meta" / "info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def video_keys_from_info(info: dict) -> list[str]:
    return [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]


def enumerate_videos(ds_dir: Path):
    """Yield (video_path, ep_idx, vid_key) for every episode/camera mp4 in a dataset."""
    info = load_info(ds_dir)
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info.get("chunks_size", 1000))
    video_path_tmpl = info.get("video_path", DEFAULT_VIDEO_PATH)
    vid_keys = video_keys_from_info(info)
    for ep_idx in range(total_episodes):
        ep_chunk = ep_idx // chunks_size
        for vid_key in vid_keys:
            rel = video_path_tmpl.format(
                episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_idx
            )
            yield ds_dir / rel, ep_idx, vid_key


def num_frames_of(video_path: Path) -> tuple[int, float]:
    """Return (num_frames, average_fps) using torchcodec metadata."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    md = decoder.metadata
    n = int(md.num_frames)
    fps = float(md.average_fps)
    return n, fps


_INTERP = {
    "bilinear": transforms_F.InterpolationMode.BILINEAR,
    "bicubic": transforms_F.InterpolationMode.BICUBIC,
}


def decode_and_resize(video_path: Path, target: int, interpolation: str, resize_space: str) -> tuple[np.ndarray, float]:
    """Decode all frames, resize 512->target per-camera, return (uint8 [N,3,target,target], fps).

    The resize is chosen to MATCH the live runtime per-camera resize so the cache is
    numerically faithful. For LIBERO (configs/data/libero_2cam*.yaml) the operative
    per-camera resize is the FastWAMProcessor's ``ToTensor`` (uint8->float/255) followed
    by ``torchvision.transforms.Resize([224,224])`` (BILINEAR, antialias=True) on the
    FLOAT tensor -- hence the defaults ``interpolation=bilinear, resize_space=float``.

    Pass ``--interpolation bicubic --resize-space uint8`` if a config instead relies on
    ``_get``'s post-concat uint8 BICUBIC resize (in that case only ~1-2 columns at each
    camera seam differ, because the cache resizes each camera independently).
    """
    n, fps = num_frames_of(video_path)
    # Request every frame by timestamp; decode_video_frames_torchcodec converts
    # timestamps -> indices via round(ts * fps), so i/fps recovers index i exactly.
    timestamps = [i / fps for i in range(n)]
    # tolerance generous: we want all frames, exact ts match is not required here.
    frames = decode_video_frames_torchcodec(video_path, timestamps, tolerance_s=1.0)  # float32 [N,3,H,W] in [0,1]

    # Match base_lerobot_dataset._get_image: float[0,1] -> uint8 (lossless round-trip
    # of the runtime decode, since _get_image does exactly (image*255).to(uint8)).
    frames_u8 = (frames * 255).to(torch.uint8)  # [N, 3, H, W]

    interp = _INTERP[interpolation]
    if resize_space == "float":
        # Replicate the processor: ToTensor (uint8->float/255) then Resize on FLOAT,
        # then quantize back to uint8 for compact storage.
        x = frames_u8.to(torch.float32) / 255.0
        x = transforms_F.resize(x, size=[target, target], interpolation=interp, antialias=True)
        resized = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    else:  # "uint8" -> resize directly on the uint8 tensor (matches _get's uint8 resize)
        resized = transforms_F.resize(frames_u8, size=[target, target], interpolation=interp, antialias=True)
    return resized.numpy(), fps


def process_one(args_tuple) -> tuple[str, int, str]:
    """Worker: decode one video and write its cache. Returns (status, num_bytes, msg)."""
    video_path_str, cache_path_str, meta_path_str, target, interpolation, resize_space = args_tuple
    video_path = Path(video_path_str)
    cache_path = Path(cache_path_str)
    meta_path = Path(meta_path_str)

    # Resumable: both the frames .npy and its fps sidecar must exist to count as done.
    if cache_path.is_file() and meta_path.is_file():
        return ("skipped", cache_path.stat().st_size, str(cache_path))

    if not video_path.is_file():
        return ("missing", 0, f"video not found: {video_path}")

    try:
        arr, fps = decode_and_resize(video_path, target, interpolation, resize_space)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically (tmp then rename) so an interrupted run leaves no partial file.
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        with open(tmp_path, "wb") as fh:
            np.save(fh, arr)  # single uint8 array -> truly mmap-able .npy at runtime
        os.replace(tmp_path, cache_path)
        # fps sidecar (write after frames so the presence of both = complete).
        meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
        with open(meta_tmp, "w", encoding="utf-8") as fh:
            json.dump({"fps": float(fps), "num_frames": int(arr.shape[0]), "target": int(target)}, fh)
        os.replace(meta_tmp, meta_path)
        return ("written", cache_path.stat().st_size, str(cache_path))
    except Exception as err:  # noqa: BLE001
        return ("error", 0, f"{video_path}: {type(err).__name__}: {err}\n{traceback.format_exc()}")


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dirs", nargs="+", default=DEFAULT_DATASET_DIRS)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--target", type=int, default=224, help="per-camera square target size (default 224)")
    ap.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    # Defaults match the LIBERO FastWAMProcessor per-camera resize (BILINEAR on float).
    ap.add_argument("--interpolation", choices=["bilinear", "bicubic"], default="bilinear")
    ap.add_argument("--resize-space", choices=["float", "uint8"], default="float",
                    help="resize on float[0,1] (processor-style) or directly on uint8 (_get-style)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build the full job list across all datasets.
    jobs = []
    for ds in args.dataset_dirs:
        ds_dir = Path(ds)
        if not (ds_dir / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"missing meta/info.json under {ds_dir}")
        for video_path, ep_idx, vid_key in enumerate_videos(ds_dir):
            cache_path = frame_cache_path(video_path, cache_dir)
            meta_path = frame_cache_meta_path(video_path, cache_dir)
            jobs.append((
                str(video_path), str(cache_path), str(meta_path),
                args.target, args.interpolation, args.resize_space,
            ))

    total = len(jobs)
    print(f"{total} episode/camera videos across {len(args.dataset_dirs)} dataset(s) -> {cache_dir}")
    print(f"target={args.target}x{args.target} uint8, resize={args.interpolation}/{args.resize_space}, "
          f"workers={args.num_workers}")

    counts = {"written": 0, "skipped": 0, "missing": 0, "error": 0}
    total_bytes = 0
    errors = []

    def handle(status, nbytes, msg, i):
        counts[status] = counts.get(status, 0) + 1
        nonlocal total_bytes
        total_bytes += nbytes
        if status in ("missing", "error"):
            errors.append(msg)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            done = counts["written"] + counts["skipped"]
            print(
                f"  [{i + 1}/{total}] written={counts['written']} skipped={counts['skipped']} "
                f"missing={counts['missing']} error={counts['error']} "
                f"size≈{human_bytes(total_bytes)}",
                flush=True,
            )

    if args.num_workers <= 1:
        for i, job in enumerate(jobs):
            status, nbytes, msg = process_one(job)
            handle(status, nbytes, msg, i)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(process_one, job): k for k, job in enumerate(jobs)}
            for i, fut in enumerate(as_completed(futures)):
                status, nbytes, msg = fut.result()
                handle(status, nbytes, msg, i)

    print("\n==== summary ====")
    print(f"written={counts['written']} skipped={counts['skipped']} "
          f"missing={counts['missing']} error={counts['error']}")
    print(f"total cache size ≈ {human_bytes(total_bytes)} (written+skipped files counted)")
    if errors:
        print(f"\n{len(errors)} problem(s); first few:")
        for e in errors[:10]:
            print("  -", e.splitlines()[0])
    print("DONE")


if __name__ == "__main__":
    main()
