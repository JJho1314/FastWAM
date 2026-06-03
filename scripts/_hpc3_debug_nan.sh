#!/usr/bin/env bash
# Reproduce the batch-2 NaN on 1 GPU with REAL weights (video frozen so 2B fits),
# printing per-tensor finiteness each step to locate the NaN source.
set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam_sana
cd /data/user/jhe724/workspace/FastWAM_sana
export FASTWAM_SANA_DEBUG_NAN=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY || true
echo "[dbg] node=$(hostname)"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_sana.yaml --num_processes 1 \
  scripts/train.py task=libero_sana_2cam224_1e-4 \
  model.train_video_expert=false \
  max_steps=60 batch_size=2 num_workers=2 eval_every=999999 save_every=999999 \
  log_every=1 wandb.enabled=false
echo "[dbg] exit=$?"
