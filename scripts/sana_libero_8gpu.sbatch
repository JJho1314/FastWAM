#!/usr/bin/env bash
#SBATCH -J sana_libero
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=0
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
# NOTE: do NOT set expandable_segments:True — it causes CUDA 'misaligned address'
# with the SANA kernels at per-GPU batch > 2 (the fault location moves around,
# a tell-tale of allocator-level misalignment rather than an op bug).
unset PYTORCH_CUDA_ALLOC_CONF || true
# Intranet wandb is on 10.x; the HTTP proxy must stay unset for it to be reachable.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY || true

# --- FSDP env (mirrors SANA's set_fsdp_env). Only when launching with an FSDP
# accelerate config; the yaml fsdp_* keys aren't always honored, but these env
# vars are. FSDP_USE_ORIG_PARAMS=true is essential — without it FSDP flattens
# the patch-embed Conv3d weight ("weight should have at least three dimensions").
if [[ "${ACCEL_CFG:-}" == *fsdp* ]]; then
  export ACCELERATE_USE_FSDP=true
  # FULL_SHARD (=ZeRO-3) shards params too -> per-layer all-gather every micro-step,
  # the dominant comm cost. SHARD_GRAD_OP (=ZeRO-2) keeps params resident (no forward
  # all-gather, grad-accum doesn't re-gather) -> much faster; the 3.1B model fits.
  # Override with FSDP_SHARD=SHARD_GRAD_OP.
  export FSDP_SHARDING_STRATEGY=${FSDP_SHARD:-FULL_SHARD}
  export FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP
  export FSDP_TRANSFORMER_CLS_TO_WRAP=SanaVideoMSBlock,DiTBlock
  export FSDP_USE_ORIG_PARAMS=true
  export FSDP_BACKWARD_PREFETCH=BACKWARD_PRE
  export FSDP_FORWARD_PREFETCH=false
  export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT
  export FSDP_SYNC_MODULE_STATES=true
  export FSDP_CPU_RAM_EFFICIENT_LOADING=false
fi

# --- intranet (on-prem) Weights & Biases ---
# Auth comes from ~/.netrc (machine 10.12.1.245:8080). Override via env if needed.
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.12.1.245:8080}"
WANDB_ENTITY_="${WANDB_ENTITY:-jjho1314}"
WANDB_PROJECT_="${WANDB_PROJECT:-fastwam-sana-libero}"
WANDB_NAME_="sana_libero_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"

NPROC=$(nvidia-smi -L | wc -l)
echo "[sbatch] node=$(hostname) gpus=$NPROC env=fastwam_sana wandb=${WANDB_BASE_URL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# HPC3 absolute paths as overrides so they're robust to config edits/rsync.
SANA_REPO=/data/user/jhe724/workspace/Sana
SANA_W=/data/user/jhe724/weights/SANA-Video_2B_480p

accelerate launch \
  --config_file "${ACCEL_CFG:-scripts/accelerate_configs/accelerate_zero1_sana.yaml}" \
  --num_processes "$NPROC" \
  scripts/train.py task=libero_sana_2cam224_1e-4 \
  "model.sana_repo_path=${SANA_REPO}" \
  "model.video_dit_pretrained_path=${SANA_W}/checkpoints/SANA_Video_2B_480p.pth" \
  "model.vae.vae_pth=${SANA_W}/vae/Wan2.1_VAE.pth" \
  "model.action_dit_pretrained_path=/data/user/jhe724/workspace/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  wandb.enabled=true wandb.mode=online \
  "wandb.workspace=${WANDB_ENTITY_}" "wandb.project=${WANDB_PROJECT_}" "wandb.name=${WANDB_NAME_}" \
  ${EXTRA:-}
