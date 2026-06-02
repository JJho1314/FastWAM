#!/usr/bin/env bash
# Train the SANA-Video FastWAM on LIBERO (libero_mujoco3.3.2).
#
# Full 2B video-expert finetune needs optimizer-state sharding, so this uses
# DeepSpeed ZeRO-2 across GPUs. Defaults to 2 GPUs (each ~48GB).
#
# Usage:
#   bash scripts/train_sana_libero.sh [nproc] [extra hydra overrides...]
# Example (smoke, random video weights):
#   bash scripts/train_sana_libero.sh 2 model.video_dit_pretrained_path=null max_steps=5
set -euo pipefail

NPROC="${1:-2}"; shift || true

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_sana.yaml \
  --num_processes "${NPROC}" \
  scripts/train.py \
  task=libero_sana_2cam224_1e-4 \
  "$@"
