"""Precompute Qwen2.5-VL text embeddings for the Cosmos-Predict2.5 FastWAM.

Cosmos conditions on Qwen2.5-VL (3584-dim) text features. This writes a per-prompt
cache compatible with RobotVideoDataset's loader, which expects files named
``{sha256(prompt)}.t5_len{context_len}.wan22ti2v5b.pt`` containing
``{context: (L, 3584), mask: (L,)}``. (The t5/wan22 tag is just the loader's
hardcoded suffix; the contents are Qwen.) The model's learned text_proj maps
3584 -> 1024 (the MiniTrainDIT crossattn dim) at train time.

Usage:
  python scripts/precompute_text_embeds_qwen.py \
    --qwen /data/.../weights/Qwen2.5-VL-7B-Instruct \
    --cache-dir ./data/text_embeds_cache/libero_qwen \
    --context-len 128 \
    --dataset-dirs ./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot ...
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

# hardcoded to avoid importing the lerobot stack (heavy deps) just for one string;
# must match fastwam.datasets.lerobot.robot_video_dataset.DEFAULT_PROMPT
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

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


def load_qwen_text(qwen_path, device, dtype):
    """Load Qwen2.5-VL and return (tokenizer, a fn prompt->[L,3584] hidden states)."""
    from transformers import AutoTokenizer
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenCls
    except Exception:
        from transformers import AutoModelForCausalLM as QwenCls
    tok = AutoTokenizer.from_pretrained(qwen_path)
    model = QwenCls.from_pretrained(qwen_path, torch_dtype=dtype).to(device).eval()
    # the language-model submodule (text path); .model is the decoder stack
    lm = getattr(model, "model", model)

    @torch.no_grad()
    def encode(text):
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        out = lm(input_ids=ids, output_hidden_states=True)
        h = out.last_hidden_state[0]  # [L, 3584]
        return h.float().cpu()

    return tok, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--dataset-dirs", nargs="+", default=DEFAULT_DATASET_DIRS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    L = args.context_len

    prompts = read_prompts(args.dataset_dirs)
    print(f"{len(prompts)} unique prompts -> {cache_dir} (len={L})")

    tok, encode = load_qwen_text(args.qwen, device, torch.bfloat16)
    for i, prompt in enumerate(prompts):
        h = encode(prompt)  # [Lp, 3584]
        D = h.shape[1]
        context = torch.zeros(L, D)
        mask = torch.zeros(L, dtype=torch.bool)
        n = min(L, h.shape[0])
        context[:n] = h[:n]
        mask[:n] = True
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        out_path = cache_dir / f"{hashed}.t5_len{L}.wan22ti2v5b.pt"
        torch.save({"context": context, "mask": mask}, out_path)
        if i % 5 == 0:
            print(f"  [{i + 1}/{len(prompts)}] len={h.shape[0]} -> {out_path.name}")
    print("DONE")


if __name__ == "__main__":
    main()
