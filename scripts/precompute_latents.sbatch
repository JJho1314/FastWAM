#!/usr/bin/env bash
#SBATCH -J sana_vae_cache
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=0
#SBATCH -t 0-04:00:00
#SBATCH -o /data/user/jhe724/workspace/FastWAM_sana/runs/_slurm/%x_%j.out
#SBATCH -e /data/user/jhe724/workspace/FastWAM_sana/runs/_slurm/%x_%j.err
#
# Precompute WanVAE latents for the SANA-Video FastWAM (one-time, ~20-30 min).
# Output dir overridable via $CACHE_DIR.
set -euo pipefail

REPO=/data/user/jhe724/workspace/FastWAM_sana
mkdir -p "$REPO/runs/_slurm"
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam_sana

export HF_HOME=/data/user/jhe724/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
unset PYTORCH_CUDA_ALLOC_CONF || true
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY || true

SANA_REPO=/data/user/jhe724/workspace/Sana
SANA_W=/data/user/jhe724/weights/SANA-Video_2B_480p
CACHE_DIR="${CACHE_DIR:-/data/user/jhe724/vae_latent_cache/libero_sana}"
NPROC=$(nvidia-smi -L | wc -l)
echo "[sbatch] node=$(hostname) gpus=$NPROC cache_dir=$CACHE_DIR"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_ddp_sana.yaml \
  --num_processes "$NPROC" \
  scripts/precompute_vae_latents_sana.py task=libero_sana_2cam224_1e-4 \
  "model.sana_repo_path=${SANA_REPO}" \
  "model.video_dit_pretrained_path=${SANA_W}/checkpoints/SANA_Video_2B_480p.pth" \
  "model.vae.vae_pth=${SANA_W}/vae/Wan2.1_VAE.pth" \
  "model.action_dit_pretrained_path=/data/user/jhe724/workspace/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
  "model.vae_latent_cache_path=${CACHE_DIR}" \
  num_workers=8 batch_size=8
