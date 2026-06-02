"""Precompute gemma-2-2b-it text embeddings for the SANA-Video FastWAM.

SANA-Video conditions on gemma-2-2b-it embeddings (2304-dim) instead of T5.
This writes a per-prompt cache compatible with `RobotVideoDataset`'s loader,
which expects files named `{sha256(prompt)}.t5_len{context_len}.wan22ti2v5b.pt`
containing `{context: (L, 2304), context_mask: (L,)}`. (The `t5/wan22` tag in
the filename is just the loader's hardcoded suffix; the contents are gemma.)

Usage:
  python scripts/precompute_text_embeds_gemma.py \
    --gemma /data/.../weights/gemma-2-2b-it \
    --cache-dir ./data/text_embeds_cache/libero_gemma \
    --context-len 300 \
    --dataset-dirs ./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot ...
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

DEFAULT_DATASET_DIRS = [
    "./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot",
    "./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot",
]


def read_prompts(dataset_dirs):
    prompts, seen = [], set()
    for ds in dataset_dirs:
        tasks_path = Path(ds) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(tasks_path)
        for line in tasks_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            task = str(json.loads(line)["task"])
            prompt = DEFAULT_PROMPT.format(task=task)
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--context-len", type=int, default=300)
    ap.add_argument("--dataset-dirs", nargs="+", default=DEFAULT_DATASET_DIRS)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(args.dataset_dirs)
    print(f"{len(prompts)} unique prompts -> {cache_dir} (len={args.context_len})")

    tok = AutoTokenizer.from_pretrained(args.gemma)
    tok.padding_side = "right"
    enc = (
        AutoModelForCausalLM.from_pretrained(args.gemma, torch_dtype=dtype)
        .get_decoder()
        .to(device)
        .eval()
    )

    suffix = f"t5_len{args.context_len}.wan22ti2v5b.pt"
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            toks = tok(
                batch,
                max_length=args.context_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            emb = enc(toks.input_ids, attention_mask=toks.attention_mask)[0]  # (B, L, 2304)
            for i, prompt in enumerate(batch):
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                payload = {
                    "context": emb[i].detach().to("cpu", torch.bfloat16).contiguous(),
                    "mask": toks.attention_mask[i].detach().to("cpu", torch.bool).contiguous(),
                }
                torch.save(payload, str(cache_dir / f"{hashed}.{suffix}"))
            print(f"  {min(start + args.batch_size, len(prompts))}/{len(prompts)}")
    print("done")


if __name__ == "__main__":
    main()
