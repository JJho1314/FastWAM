"""Extract GR00T-N1.7-3B's pretrained action-DiT weights for the AGRA coupling.

Pulls `action_head.model.*` tensors out of the GR00T-N1.7-3B safetensors shards,
strips the `action_head.model.` prefix (-> the BridgeDiT namespace), and saves a single
.pt. This is the ~1.1B-param flow-matching action DiT (transformer_blocks +
timestep_encoder + proj_out_1/2) the AGRA paper "employs from GR00T-N1".

  python scripts/extract_gr00t_action_dit.py
"""
import argparse
import glob
import os

import torch
from safetensors import safe_open

SRC = "/data/LFT-W02_data/junjie/weights/GR00T-N1.7-3B"
OUT = "/data/LFT-W02_data/junjie/weights/gr00t_n1d7_action_dit.pt"
PREFIX = "action_head.model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    assert shards, f"no safetensors in {args.src}"
    sd = {}
    for sh in shards:
        with safe_open(sh, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(PREFIX):
                    sd[k[len(PREFIX):]] = f.get_tensor(k)
    assert sd, "no action_head.model.* keys found"
    # quick structure report
    nblocks = len({k.split(".")[1] for k in sd if k.startswith("transformer_blocks.")})
    total = sum(v.numel() for v in sd.values())
    print(f"extracted {len(sd)} tensors, {nblocks} transformer_blocks, {total/1e9:.3f}B params")
    print("sample keys:", sorted(sd)[:3], "...", sorted(sd)[-3:])
    print("dtypes:", {str(v.dtype) for v in sd.values()})
    torch.save(sd, args.out)
    print("saved ->", args.out, f"({os.path.getsize(args.out)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
