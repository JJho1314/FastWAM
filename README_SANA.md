# FastWAM — SANA-Video backbone variant

A parallel implementation of FastWAM that replaces the **Wan2.2 5B MoT** backbone
with a **SANA-Video-2B linear-attention DiT** as the video world model. The action
expert is coupled to the world model via **cross-attention** (it reads the video
DiT's intermediate token features) instead of MoT joint attention — SANA's linear
attention is incompatible with MoT's masked softmax joint attention.

The original `src/fastwam/models/wan22/` path is **untouched**; all SANA code lives
in `src/fastwam/models/sana/`. The data pipeline, flow-matching scheduler, and
trainer are reused unchanged.

```
video latents ──(+noise)──▶ SanaMSVideo (2B, pretrained) ──▶ video velocity   (video flow loss)
                                   │ block features (B, N, 2240)
                                   ▼
action chunk  ──(+noise)──▶ SanaActionExpert ─ cross-attends video feats
                            + gemma text / proprio context ──▶ action velocity (action flow loss)
```

| Component | Choice |
|---|---|
| Video DiT | `SanaMSVideo_2000M_P2_D20` (2.06B, `LiteLAReLURope` linear attn, `wan_rope` 3D RoPE) |
| VAE | Wan2.1 VAE (16-channel latents, stride [4,8,8]) |
| Text encoder | `gemma-2-2b-it` (2304-dim), precomputed to a cache |
| Action expert | 12-layer cross-attention DiT (self-attn + cross-attn to video + text/proprio) |
| Objective | rectified flow matching (velocity target), video + action losses |

---

## 1. Environment

SANA-Video needs a newer dependency stack than the base FastWAM env, so use a
**separate** conda env (do not modify your `fastwam` env). The fastest route is to
clone the working `fastwam` env and layer the extra deps on top — this keeps the
known-good `torch 2.7.1+cu128 / accelerate / deepspeed / lerobot` base.

```bash
# 1) clone the base env (keeps torch 2.7.1+cu128, deepspeed, lerobot, ...)
conda create -y -n fastwam_sana --clone fastwam

# 2) SANA-Video model deps — installed with --no-deps so torch is NOT changed.
#    fla.modules / fla.ops live in `fla-core` (the `flash-linear-attention`
#    PyPI wheel only ships fla.layers and depends on fla-core).
conda run -n fastwam_sana pip install --no-deps fla-core flash-linear-attention timm xformers==0.0.31

# 3) mmcv 1.7.2 — the *lite* (pure-python) build; SANA only uses mmcv.Registry.
#    Build without isolation so it can see the env's setuptools (pkg_resources).
conda run -n fastwam_sana pip install addict yapf
conda run -n fastwam_sana bash -c 'MMCV_WITH_OPS=0 pip install "mmcv==1.7.2" --no-build-isolation --no-deps'

# 4) diffusers >= 0.37 (SANA builder imports AutoencoderKLLTX2Video). FastWAM
#    training code does not import diffusers, so upgrading here is safe.
conda run -n fastwam_sana pip install -U "diffusers==0.38.0" --no-deps

# 5) misc text/util deps pulled in by the SANA import chain
conda run -n fastwam_sana pip install qwen_vl_utils ftfy termcolor easydict
```

Sanity check (should print `OK`):

```bash
conda run -n fastwam_sana python -c "
import sys; sys.path.insert(0,'/path/to/Sana')
from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo_2000M_P2_D20
print('OK')"
```

> Notes
> - `xformers==0.0.31` matches `torch 2.7.1+cu128`; it is required by SANA's text
>   cross-attention path.
> - `flash-linear-attention` on PyPI is just `fla.layers/models` and **requires
>   `fla-core`** for `fla.modules.ShortConvolution`. Install both.

## 2. SANA-Video source

The model code is imported from a local clone of **NVlabs/Sana** (the branch with
SANA-Video / SANA-WM). Set its path in `configs/model/fastwam_sana.yaml`
(`sana_repo_path`). Default used here:

```
/data/LFT-W02_data/junjie/VLA_WM/Sana
```

The factory adds this path to `sys.path` at runtime — no `pip install` of Sana is
needed (which also avoids pulling its heavy inference-only deps).

## 3. Download weights

Download to your weights dir. In mainland China, use the HF mirror with the proxy
disabled (direct access to `huggingface.co` API/CDN is otherwise blocked):

```bash
export HF_ENDPOINT=https://hf-mirror.com
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
W=/data/LFT-W02_data/junjie/weights

# SANA-Video-2B: DiT checkpoint (~8.25 GB) + Wan2.1 VAE (~0.5 GB)
hf download Efficient-Large-Model/SANA-Video_2B_480p --local-dir $W/SANA-Video_2B_480p
# gemma-2-2b-it text encoder
hf download Efficient-Large-Model/gemma-2-2b-it       --local-dir $W/gemma-2-2b-it
```

If the gemma repo ships a merged `gemma-2-2b-it.safetensors` plus a sharded
`model.safetensors.index.json` whose shards are incomplete, point transformers at
the merged file:

```bash
cd $W/gemma-2-2b-it
mv model.safetensors.index.json model.safetensors.index.json.bak
ln -sf gemma-2-2b-it.safetensors model.safetensors
```

Then set the paths in `configs/model/fastwam_sana.yaml`:
`video_dit_pretrained_path`, `vae.vae_pth`.

## 4. Data + text-embedding cache

LIBERO data (LeRobot format) is expected under `./data/libero_mujoco3.3.2/`:
`libero_{spatial,object,goal,10}_no_noops_lerobot`. (See the base FastWAM README
for downloading/converting LIBERO.)

SANA conditions on **gemma** embeddings, so precompute them into a cache the
dataset loader reads (length 300, 2304-dim):

```bash
conda run -n fastwam_sana python scripts/precompute_text_embeds_gemma.py \
  --gemma $W/gemma-2-2b-it \
  --cache-dir ./data/text_embeds_cache/libero_gemma \
  --context-len 300
```

This is referenced by `configs/data/libero_2cam_sana.yaml`
(`text_embedding_cache_dir`).

## 5. Train

Full 2B video-expert finetuning needs optimizer-state sharding, so training uses
**DeepSpeed ZeRO-2** across GPUs (defaults to 2× ~48 GB GPUs):

```bash
conda run --no-capture-output -n fastwam_sana \
  bash scripts/train_sana_libero.sh 2          # <nproc> [hydra overrides...]
```

Useful overrides:

```bash
# quick smoke (random video weights, skip the 2B load):
bash scripts/train_sana_libero.sh 2 model.video_dit_pretrained_path=null max_steps=5

# longer run, larger micro-batch:
bash scripts/train_sana_libero.sh 2 batch_size=2 num_epochs=10 save_every=2000
```

The launch script wraps:

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_sana.yaml \
  --num_processes <nproc> \
  scripts/train.py task=libero_sana_2cam224_1e-4 "$@"
```

Checkpoints/logs go to `./runs/libero_sana_2cam224_1e-4/...`.

### Expected output

```
Loaded SANA-Video weights ... (loaded=417, ..., shape_skipped=['pos_embed'])
... step=2/.. loss=1.9  loss_video=0.3  loss_action=1.6
```

`loss_video ≈ 0.1–0.3` from the first steps confirms the pretrained SANA-Video
prior loaded correctly. `pos_embed` (the unused sincos table; this model uses
RoPE) is skipped on load by design.

## 6. Configuration notes & stability

- **Learning rate / NaN.** Under DeepSpeed, `accelerate.clip_grad_norm_` is a
  no-op — clipping must be set in the DS JSON. `scripts/ds_configs/ds_zero2_sana.json`
  sets `"gradient_clipping": 1.0`. Combined with `learning_rate: 2e-5`
  (in the task config) this trains stably; `1e-4` with a fresh action head
  diverged to NaN around step ~12.
- **Memory.** One 48 GB GPU cannot hold the full 2B AdamW states; use ≥2 GPUs
  (ZeRO-2 shards optimizer states + grads). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is exported by the launch script.
- **Knobs** (`configs/model/fastwam_sana.yaml`):
  - `train_video_expert: false` → freeze SANA, train only the action expert
    (the "frozen world-model encoder" ablation; far less memory).
  - `detach_video_feats: true` → stop action-loss gradients from flowing into the
    video DiT through the cross-attention.
  - `loss.lambda_video` / `loss.lambda_action` → reweight the two objectives.
  - `feature_layer` → which video DiT block's features the action expert reads.

## 7. File map

```
src/fastwam/models/sana/
  action_expert.py   # cross-attention flow-matching action DiT
  video_expert.py    # wraps SanaMSVideo; forward_with_features() exposes block feats
  fastwam_sana.py    # top model: build_inputs / training_loss / .dit / load_checkpoint
src/fastwam/runtime.py            # + create_fastwam_sana(...) factory
configs/model/fastwam_sana.yaml
configs/data/libero_2cam_sana.yaml         # libero_mujoco3.3.2 + gemma cache
configs/task/libero_sana_2cam224_1e-4.yaml
scripts/train_sana_libero.sh               # 2-GPU ZeRO-2 launcher
scripts/precompute_text_embeds_gemma.py
scripts/accelerate_configs/accelerate_zero2_sana.yaml
scripts/ds_configs/ds_zero2_sana.json      # ZeRO-2 + gradient_clipping
scripts/_sana_{forward,integration,data,ckpt}_*.py   # diagnostics / smoke tests
```
