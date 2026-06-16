#!/usr/bin/env bash
#SBATCH -J cosmos_libero
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=0
#SBATCH -t 1-00:00:00
#SBATCH -o /data/user/jhe724/workspace/FastWAM_cosmos/runs/_slurm/%x_%j.out
#SBATCH -e /data/user/jhe724/workspace/FastWAM_cosmos/runs/_slurm/%x_%j.err
#
# 8-GPU Cosmos-Predict2.5 FastWAM training on LIBERO (128 global batch, 10 epochs).
# Uses the reused .venv (torch2.7+cu128 / TE / flash-attn) + fresh official cosmos
# code via PYTHONPATH (see cosmos_env.sh). Submit: sbatch scripts/cosmos_libero_8gpu.sbatch
set -euo pipefail

REPO=/data/user/jhe724/workspace/FastWAM_cosmos
mkdir -p "$REPO/runs/_slurm"
cd "$REPO"

# Cosmos env: venv python + nvidia libs (TE cuDNN) + fresh official cosmos code.
source /data/user/jhe724/workspace/cosmos_env.sh
export PYTHONPATH="$REPO/src:$PYTHONPATH"   # fastwam package
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
unset PYTORCH_CUDA_ALLOC_CONF || true
# Intranet wandb is on 10.x; the HTTP proxy must stay unset for it to be reachable.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY || true

# --- FSDP env (the parallelism that worked for the SANA variant; SANA+DeepSpeed
# misaligned at batch>2, FSDP is clean). SHARD_GRAD_OP keeps params resident.
if [[ "${ACCEL_CFG:-}" == *fsdp* ]]; then
  export ACCELERATE_USE_FSDP=true
  export FSDP_SHARDING_STRATEGY=${FSDP_SHARD:-SHARD_GRAD_OP}
  export FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP
  export FSDP_TRANSFORMER_CLS_TO_WRAP=Block
  export FSDP_USE_ORIG_PARAMS=true
  export FSDP_BACKWARD_PREFETCH=BACKWARD_PRE
  export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT
  export FSDP_SYNC_MODULE_STATES=true
  export FSDP_CPU_RAM_EFFICIENT_LOADING=false
fi

# --- intranet (on-prem) Weights & Biases ---
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.12.1.245:8080}"
WANDB_ENTITY_="${WANDB_ENTITY:-jjho1314}"
WANDB_PROJECT_="${WANDB_PROJECT:-fastwam-cosmos-libero}"
WANDB_NAME_="cosmos_libero_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"

NPROC=$(nvidia-smi -L | wc -l)
echo "[sbatch] node=$(hostname) gpus=$NPROC python=$COSMOS_PY wandb=${WANDB_BASE_URL}"

# HPC3 absolute paths (robust to config edits / rsync).
W=/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B
CKPT=$W/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt

# 512 global batch = per-GPU 2 x 8 GPUs x grad-accum 32 (heavy 4B model; per-GPU
# batch kept at 2 for memory, larger global batch via accumulation). 10 epochs.
"$COSMOS_PY" -m accelerate.commands.launch \
  --config_file "${ACCEL_CFG:-scripts/accelerate_configs/accelerate_fsdp_sana.yaml}" \
  --num_processes "$NPROC" \
  scripts/train.py task=libero_cosmos_2cam224 \
  "model.video_dit_pretrained_path=${CKPT}" \
  "model.vae.vae_pth=${W}/tokenizer.pth" \
  wandb.enabled=true wandb.mode=online \
  "wandb.workspace=${WANDB_ENTITY_}" "wandb.project=${WANDB_PROJECT_}" "wandb.name=${WANDB_NAME_}" \
  batch_size=2 gradient_accumulation_steps=32 num_epochs=10 \
  ${EXTRA:-}
