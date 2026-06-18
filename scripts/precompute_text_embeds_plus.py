"""Precompute Qwen2.5-VL text embeddings for an arbitrary prompt list (LIBERO-Plus).

Applies the SAME FastWAM preprocessing as scripts/precompute_text_embeds_qwen.py:
the DEFAULT_PROMPT template is wrapped around each instruction, and the cache key is
sha256(templated prompt) -- exactly what FastWAMCosmos._load_text_context_from_cache
looks up. Saves {context[L,3584], mask[L]} as bf16 (eval casts to bf16 anyway) to halve
the cache size vs the float32 standard cache.

  python scripts/precompute_text_embeds_plus.py --qwen <Qwen2.5-VL-7B> \
      --prompts libero_plus_prompts.txt --cache-dir <out> --context-len 128
"""
import argparse, hashlib
from pathlib import Path
import torch

# must match fastwam.datasets.lerobot.robot_video_dataset.DEFAULT_PROMPT
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def load_qwen_text(qwen_path, device, dtype):
    from transformers import AutoTokenizer
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenCls
    except Exception:
        from transformers import AutoModelForCausalLM as QwenCls
    tok = AutoTokenizer.from_pretrained(qwen_path)
    model = QwenCls.from_pretrained(qwen_path, torch_dtype=dtype).to(device).eval()
    lm = getattr(model, "model", model)

    @torch.no_grad()
    def encode(text):
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        out = lm(input_ids=ids, output_hidden_states=True)
        return out.last_hidden_state[0].float().cpu()  # [L, 3584]

    return tok, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--prompts", required=True)  # one instruction (task.language) per line
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--save-dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    L = args.context_len
    sdt = torch.bfloat16 if args.save_dtype == "bf16" else torch.float32

    langs = [l.rstrip("\n") for l in open(args.prompts, encoding="utf-8") if l.strip()]
    seen, items = set(), []
    for lang in langs:
        p = DEFAULT_PROMPT.format(task=lang)
        if p in seen:
            continue
        seen.add(p)
        items.append(p)
    print(f"{len(items)} unique templated prompts -> {cache_dir} (len={L}, dtype={args.save_dtype})")

    tok, encode = load_qwen_text(args.qwen, device, torch.bfloat16)
    done = made = 0
    for i, p in enumerate(items):
        hashed = hashlib.sha256(p.encode("utf-8")).hexdigest()
        out_path = cache_dir / f"{hashed}.t5_len{L}.wan22ti2v5b.pt"
        if out_path.exists():
            done += 1
            continue
        h = encode(p)
        D = h.shape[1]
        context = torch.zeros(L, D)
        mask = torch.zeros(L, dtype=torch.bool)
        n = min(L, h.shape[0])
        context[:n] = h[:n]
        mask[:n] = True
        torch.save({"context": context.to(sdt), "mask": mask}, out_path)
        made += 1
        if made % 200 == 0:
            print(f"  [{i + 1}/{len(items)}] made={made} skipped={done}")
    print(f"DONE made={made} skipped(existing)={done}")


if __name__ == "__main__":
    main()
