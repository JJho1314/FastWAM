#!/usr/bin/env bash
#SBATCH -J sana_libero
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=512G
#SBATCH -t 1-00:00:00
#SBATCH -o /data/user/jhe724/workspace/FastWAM_sana/runs/_slurm/%x_%j.out
#SBATCH -e /data/user/jhe724/workspace/FastWAM_sana/runs/_slurm/%x_%j.err
#
# 8-GPU SANA-Video FastWAM training on LIBERO (HPC3, ZeRO-2).
# Submit:  sbatch scripts/sana_libero_8gpu.sbatch
#   override hydra args via env, e.g.:  EXTRA="max_steps=300 batch_size=2" sbatch ...
set -euo pipefail

REPO=/data/user/jhe724/workspace/FastWAM_sana
mkdir -p "$REPO/runs/_slurm"
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam_sana

export HF_HOME=/data/user/jhe724/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Intranet wandb is on 10.x; the HTTP proxy must stay unset for it to be reachable.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY || true

# --- intranet (on-prem) Weights & Biases ---
# Auth comes from ~/.netrc (machine 10.12.1.245:8080). Override via env if needed.
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.12.1.245:8080}"
WANDB_ENTITY_="${WANDB_ENTITY:-jjho1314}"
WANDB_PROJECT_="${WANDB_PROJECT:-fastwam-sana-libero}"
WANDB_NAME_="sana_libero_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"

NPROC=$(nvidia-smi -L | wc -l)
echo "[sbatch] node=$(hostname) gpus=$NPROC env=fastwam_sana wandb=${WANDB_BASE_URL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_sana.yaml \
  --num_processes "$NPROC" \
  scripts/train.py task=libero_sana_2cam224_1e-4 \
  wandb.enabled=true wandb.mode=online \
  "wandb.workspace=${WANDB_ENTITY_}" "wandb.project=${WANDB_PROJECT_}" "wandb.name=${WANDB_NAME_}" \
  ${EXTRA:-}
