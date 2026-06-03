#!/usr/bin/env bash
# 1-GPU wandb smoke: confirm a compute node can log to the intranet wandb server.
set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam_sana
cd /data/user/jhe724/workspace/FastWAM_sana
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY || true
export WANDB_BASE_URL=http://10.12.1.245:8080
echo "[wbsmoke] node=$(hostname) wandb=$WANDB_BASE_URL"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_sana.yaml --num_processes 1 \
  scripts/train.py task=libero_sana_2cam224_1e-4 \
  model.video_dit_pretrained_path=null model.train_video_expert=false \
  max_steps=3 batch_size=1 num_workers=2 eval_every=999 save_every=999 \
  wandb.enabled=true wandb.mode=online wandb.workspace=jjho1314 \
  wandb.project=fastwam-sana-libero "wandb.name=wbsmoke_$(date +%H%M%S)"
echo "[wbsmoke] exit=$?"
