"""Inspect a Cosmos-Predict2.5 checkpoint structure (run on login node, CPU).

Usage: COSMOS_PY _sanity_ckpt.py <ckpt.pt>
"""
import sys
import torch


def main():
    path = sys.argv[1]
    # mmap=True: memory-map (lazy) so a 4GB ckpt doesn't OOM the login node.
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    print("top type:", type(ck))
    if not isinstance(ck, dict):
        return
    print("top keys:", list(ck.keys())[:12])
    # locate the net state dict
    sd = ck
    for k in ("model", "state_dict", "net", "ema", "module"):
        if k in ck and isinstance(ck[k], dict):
            print(f"  nested dict '{k}': {len(ck[k])} entries")
            sd = ck[k]
    keys = [k for k in sd.keys() if torch.is_tensor(sd[k])]
    print("num tensors:", len(keys))
    if keys:
        print("dtype:", sd[keys[0]].dtype)
        # prefix histogram (first token of each key)
        from collections import Counter
        pref = Counter(k.split(".")[0] for k in keys)
        print("top-level prefixes:", dict(list(pref.items())[:12]))
        print("sample keys:")
        for k in keys[:12]:
            print("   ", k, tuple(sd[k].shape))
        # look for block-0 keys to confirm the MiniTrainDIT layout
        b0 = [k for k in keys if "blocks.0." in k or "block.0." in k]
        print("block-0 keys (first 10):")
        for k in b0[:10]:
            print("   ", k, tuple(sd[k].shape))


if __name__ == "__main__":
    main()
