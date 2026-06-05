"""Precompute WanVAE latents for the SANA-Video FastWAM (the single biggest
per-step cost, ~28%: a 3D-conv encode of 33 frames x 2 cams every step).

The video clip for each dataset index is deterministic (center crop, fixed
temporal window, skip_padding off), so its latent can be cached once and loaded
at train time, skipping the encode entirely. This writes, into the directory
given by `model.vae_latent_cache_path`:

  latents.mmap   float32 memmap, shape (N, *per_sample_latent_shape), row == idx
  meta.json      {"n": N, "shape": [...], "dtype": "float32"}

Latents are produced by the *exact* training-time `model._vae_encode`, so they
match byte-for-byte (verify at train time with FASTWAM_VERIFY_LATENT_CACHE=1).

Launch (8 GPUs), same hydra overrides as training plus the output dir:

  accelerate launch --config_file <ddp_cfg> --num_processes 8 \
    scripts/precompute_vae_latents_sana.py task=libero_sana_2cam224_1e-4 \
    model.vae_latent_cache_path=/abs/path/to/cache_dir  <other path overrides...>
"""
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    acc = Accelerator()
    rank = acc.process_index
    world = acc.num_processes
    device = acc.device

    out_dir = cfg.model.get("vae_latent_cache_path", None)
    if not out_dir:
        raise ValueError("Set model.vae_latent_cache_path=<output dir> for the precompute.")
    out_dir = str(out_dir)
    mmap_path = os.path.join(out_dir, "latents.mmap")
    meta_path = os.path.join(out_dir, "meta.json")

    # Deterministic dataset (no skip-padding retries -> cache_key == idx).
    ds = instantiate(cfg.data.train)
    ds.skip_padding_as_possible = False
    n = len(ds)

    # Build the model purely for its (frozen) VAE + exact _vae_encode path.
    model = instantiate(cfg.model, model_dtype=torch.float32, device=str(device))
    model.eval()

    @torch.no_grad()
    def encode(video):
        return model._vae_encode(video.to(device=device, dtype=torch.float32)).float()

    # Discover per-sample latent shape from one deterministic sample (all ranks agree).
    probe = ds[0]["video"].unsqueeze(0)
    lat_shape = tuple(int(s) for s in encode(probe).shape[1:])
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        # Allocate the full memmap (zero-filled) once, then fill row-by-row.
        mm0 = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(n, *lat_shape))
        mm0.flush()
        del mm0
        with open(meta_path, "w") as f:
            json.dump({"n": int(n), "shape": list(lat_shape), "dtype": "float32"}, f, indent=2)
        print(f"[precompute] n={n} per-sample latent shape={lat_shape} -> {out_dir} "
              f"(~{n * int(np.prod(lat_shape)) * 4 / 1e9:.1f} GB)", flush=True)
    acc.wait_for_everyone()

    # Each rank writes a disjoint stride of indices (no overlap, no padding dups).
    idxs = list(range(rank, n, world))
    loader = DataLoader(
        Subset(ds, idxs),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    mm = np.memmap(mmap_path, dtype=np.float32, mode="r+", shape=(n, *lat_shape))

    done = 0
    for batch in loader:
        keys = batch["cache_key"].cpu().numpy().astype(np.int64).reshape(-1)
        lat = encode(batch["video"]).cpu().numpy()  # (B, *lat_shape)
        mm[keys] = lat
        done += len(keys)
        if rank == 0 and done % (int(cfg.batch_size) * 50) == 0:
            print(f"[precompute] rank0 {done}/{len(idxs)}", flush=True)
    mm.flush()
    del mm
    acc.wait_for_everyone()
    if rank == 0:
        print(f"[precompute] DONE n={n} -> {mmap_path}", flush=True)


if __name__ == "__main__":
    main()
