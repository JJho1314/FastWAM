#!/usr/bin/env bash
# 1-GPU end-to-end smoke on a compute node: full import chain + data + 2 steps,
# with random video weights and a frozen video expert (fits one GPU).
set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam_sana
cd /data/user/jhe724/workspace/FastWAM_sana
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY || true
echo "[smoke] node=$(hostname)"; nvidia-smi -L | head -1
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_sana.yaml \
  --num_processes 1 \
  scripts/train.py task=libero_sana_2cam224_1e-4 \
  model.video_dit_pretrained_path=null model.train_video_expert=false \
  max_steps=2 batch_size=1 num_workers=2 eval_every=999 save_every=999 \
  wandb.enabled=false
echo "[smoke] exit=$?"
