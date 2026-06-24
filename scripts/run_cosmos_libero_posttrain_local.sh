#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean}"
COSMOS_REPO="${COSMOS_REPO:-/data/LFT-W02_data/junjie/cosmos-predict2.5}"
COSMOS_PY="${COSMOS_PY:-${COSMOS_REPO}/.venv/bin/python}"
W="${COSMOS_WEIGHTS:-/data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B}"
CKPT="${CKPT:-${W}/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt}"
TEXT_CACHE="${TEXT_CACHE:-/data/LFT-W02_data/junjie/_ola_stage/libero_qwen}"

cd "$REPO"

export PYTHONPATH="$REPO/src:$COSMOS_REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset PYTORCH_CUDA_ALLOC_CONF || true

# Match the original Cosmos sbatch FSDP setup; this repo trains Cosmos via FSDP,
# while the current Cosmos venv does not include DeepSpeed.
export ACCELERATE_USE_FSDP=true
export FSDP_SHARDING_STRATEGY="${FSDP_SHARD:-SHARD_GRAD_OP}"
export FSDP_AUTO_WRAP_POLICY="${FSDP_AUTO_WRAP_POLICY:-TRANSFORMER_BASED_WRAP}"
export FSDP_TRANSFORMER_CLS_TO_WRAP="${FSDP_TRANSFORMER_CLS_TO_WRAP:-Block}"
export FSDP_USE_ORIG_PARAMS=true
export FSDP_BACKWARD_PREFETCH=BACKWARD_PRE
export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT
export FSDP_SYNC_MODULE_STATES=true
export FSDP_CPU_RAM_EFFICIENT_LOADING=false

RUN_ID="${RUN_ID:-posttrain_$(date +%Y-%m-%d_%H-%M-%S)}"
WANDB_NAME_="${WANDB_NAME:-cosmos_libero_mot_joint_denoise_posttrain_${RUN_ID}}"
NPROC="${NPROC:-2}"

ACCEL_ARGS=(
  --use_fsdp
  --mixed_precision bf16
  --num_processes "$NPROC"
  --num_machines 1
  --fsdp_sharding_strategy "${FSDP_SHARDING_STRATEGY}"
  --fsdp_backward_prefetch "${FSDP_BACKWARD_PREFETCH}"
  --fsdp_state_dict_type "${FSDP_STATE_DICT_TYPE}"
  --fsdp_use_orig_params true
  --fsdp_sync_module_states true
  --fsdp_cpu_ram_efficient_loading false
)

if [[ "${FASTWAM_FSDP_AUTO_WRAP:-1}" != "0" ]]; then
  ACCEL_ARGS+=(
    --fsdp_auto_wrap_policy "${FSDP_AUTO_WRAP_POLICY}"
    --fsdp_transformer_layer_cls_to_wrap "${FSDP_TRANSFORMER_CLS_TO_WRAP}"
  )
fi

"$COSMOS_PY" -m accelerate.commands.launch \
  "${ACCEL_ARGS[@]}" \
  scripts/train.py task=libero_cosmos_2cam224 \
  "output_dir=./runs/libero_cosmos_mot_joint_denoise_posttrain/${RUN_ID}" \
  "model.video_dit_pretrained_path=${CKPT}" \
  "model.vae.vae_pth=${W}/tokenizer.pth" \
  "data.train.text_embedding_cache_dir=${TEXT_CACHE}" \
  wandb.enabled=true wandb.mode=online \
  "wandb.workspace=${WANDB_ENTITY:-jjho1314}" \
  "wandb.project=${WANDB_PROJECT:-fastwam-cosmos-libero}" \
  "wandb.name=${WANDB_NAME_}" \
  batch_size=2 gradient_accumulation_steps=32 num_epochs=10 \
  "${@}"
