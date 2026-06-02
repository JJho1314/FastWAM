"""Data smoke test: compose the SANA LIBERO task config, instantiate the dataset,
fetch one sample, and verify shapes (video / action / proprio / gemma context)."""
import os
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

cfg_dir = os.path.abspath("configs")
with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=libero_sana_2cam224_1e-4"])

ds = instantiate(cfg.data.train)
print("dataset len:", len(ds))
s = ds[0]
for k in ["video", "action", "proprio", "context", "context_mask", "action_is_pad", "image_is_pad"]:
    v = s.get(k, None)
    if v is None:
        print(f"  {k}: None")
    elif hasattr(v, "shape"):
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v)} -> {v}")
print("prompt:", s.get("prompt"))
print("DATA OK")
