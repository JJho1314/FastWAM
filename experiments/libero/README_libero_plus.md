# LIBERO-Plus robustness eval for FastWAM-Cosmos

Evaluate a FastWAM-Cosmos policy on **LIBERO-Plus** (arXiv 2510.13626): 10,030 perturbed
tasks across 7 dimensions (Objects Layout / Camera Viewpoints / Robot Initial States /
Language Instructions / Light Conditions / Background Textures / Sensor Noise). Protocol:
1 trial/task, report per-dimension + overall.

LIBERO-Plus is a **drop-in `libero` package** (same suite names; the perturbed tasks are
packed into the 4 standard suites as extra task entries). Our standard-LIBERO eval harness
(`cosmos_eval_libero.py` + `eval_libero_single.py`) is reused with three additions, all in
`cosmos_eval_libero_plus.py`: num_trials=1, per-category aggregation via
`task_classification.json`, and `--exclude_categories`.

## 1. Get LIBERO-Plus on the eval node
```bash
# the repo (code + assets, ~9.6GB: assets 9.5G + bddl + init_files); HF assets pre-downloaded
#   git clone https://github.com/SylvestF/LIBERO-plus  (or rsync an existing checkout)
# point a config at it (so bddl_files/init_files/assets resolve):
mkdir -p ~/.libero_plus && cat > ~/.libero_plus/config.yaml <<EOF
benchmark_root: /path/to/LIBERO-plus/libero/libero
bddl_files:     /path/to/LIBERO-plus/libero/libero/bddl_files
init_states:    /path/to/LIBERO-plus/libero/libero/init_files
datasets:       /path/to/LIBERO-plus/libero/libero/../datasets
assets:         /path/to/LIBERO-plus/libero/libero/assets
EOF
```
The eval puts the LIBERO-Plus checkout FIRST on `sys.path` (so its `libero` wins) and sets
`LIBERO_CONFIG_PATH=~/.libero_plus`. Override the checkout with `LIBERO_PLUS_ROOT`.

## 2. ImageMagick + wand (only for the Sensor Noise dimension)
The Sensor-Noise `motion_blur` perturbation (severity 1-10) needs ImageMagick's MagickWand
via `wand` (severity 11-50 use skimage/scipy, already present). On a node with **no sudo**
this installs user-space (no conda needed system-wide):
```bash
# micromamba (single static binary) -> imagemagick from conda-forge
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
MAMBA_ROOT_PREFIX=$PWD/mamba_root ./bin/micromamba create -y -p $PWD/im_env -c conda-forge imagemagick
# wand into the project venv (uv venv has no pip):
VIRTUAL_ENV=<venv> uv pip install wand
# at eval time the launcher exports:
export MAGICK_HOME=$PWD/im_env
export LD_LIBRARY_PATH=$MAGICK_HOME/lib:$LD_LIBRARY_PATH
```
Skip this and pass `--exclude_categories "Sensor Noise"` to eval the other 6 dimensions.

## 3. Precompute the text-embedding cache
LIBERO-Plus bakes the perturbation suffix into `task.language` (e.g. "...place it on the
plate **table 1**"), so ~10,002 prompts are unique and must each be cached. Cache key =
`sha256(DEFAULT_PROMPT.format(task=task.language))` — same template as training
(`robot_video_dataset.DEFAULT_PROMPT`) and inference (`eval_libero_single.py`).
```bash
# extract every unique task.language (load the LIBERO-Plus benchmark, dump get_task(i).language)
# then (on a box with Qwen2.5-VL + a GPU):
python scripts/precompute_text_embeds_plus.py --qwen <Qwen2.5-VL-7B-Instruct> \
    --prompts libero_plus_prompts_all.txt --cache-dir <FASTWAM_TEXT_CACHE_DIR> \
    --context-len 128 --save-dtype bf16          # ~920KB/file bf16, ~9.2GB total
```

## 4. Run (parallel, sharded)
```bash
CPL=agra NIS=10 NPROC=64 EXCL="" \
  RDIR=<run_dir with checkpoints/weights/step_*.pt + dataset_stats.json> \
  OUT=<out_dir> \
  bash experiments/libero/run_cosmos_eval_plus_par.sh
python experiments/libero/combine_plus.py <out_dir>   # per-dimension + per-suite + overall
```
`NPROC=64` = 8 procs/GPU (~7.8GB each). The launcher caps per-proc CPU threads
(`OMP_NUM_THREADS=1` + the script's `torch.set_num_threads(1)`) — WITHOUT this, torch
spawns ~ncpu intra-op threads per proc and oversubscribes the CPU (GPU then starves at
~28% util). With the caps + 8/GPU, GPU util runs 60-92%.

## Result (GR00T-AGRA, step_021700, official raw-prompt protocol)
OVERALL **54.94%** (5510/10030) vs standard LIBERO 96.55%. Per dim: ObjLayout 75.2 /
Language 72.7 / Light 68.0 / SensorNoise 55.0 / Camera 51.3 / RobotInit 36.6 /
BgTexture 18.8.
