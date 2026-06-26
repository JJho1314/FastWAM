#!/usr/bin/env bash
# Parallel LIBERO-Plus eval: shard all non-excluded (suite:tid) pairs across NPROC
# procs (5/GPU x 8), each runs cosmos_eval_libero_plus.py on its shard.
VENV=/data/users/junjie/cosmos-predict2.5-fw/.venv; PY=$VENV/bin/python
for d in $VENV/lib/python3.10/site-packages/nvidia/*/lib; do [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# ImageMagick (micromamba im_env) so wand's motion_blur works for the Sensor Noise dim
export MAGICK_HOME=${MAGICK_HOME:-/data/users/junjie/im_env}
export LD_LIBRARY_PATH=/data/users/junjie/im_env/lib:${LD_LIBRARY_PATH:-}
# cap CPU threads per proc -> avoid torch/OpenMP oversubscription when packing many procs/GPU
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /data/users/junjie/FastWAM_cosmos
NIS=${NIS:-10}
NPROC=${NPROC:-64}
EXCL="${EXCL:-}"
CPL=${CPL:-agra}
RDIR=${RDIR:-/data/users/junjie/FastWAM_cosmos/runs/train/2026-06-17_17-00-14}
STEP=${STEP:-21700}
CKPT=${CKPT:-}
ACTION_HIDDEN_DIM=${ACTION_HIDDEN_DIM:-1024}
ACTION_FFN_DIM=${ACTION_FFN_DIM:-4096}
ACTION_ATTENTION_HEAD_DIM=${ACTION_ATTENTION_HEAD_DIM:-128}
OUT=${OUT:-/data/users/junjie/FastWAM_cosmos/evaluate_results/cosmos_agra_gr00t_plus}
LIBERO_PLUS_ROOT=${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}
export LIBERO_PLUS_ROOT
mkdir -p $OUT/shards

$PY - "$NPROC" "$EXCL" "$OUT" "$LIBERO_PLUS_ROOT" <<'PYEOF'
import json, sys, os
nproc=int(sys.argv[1]); excl=set(c.strip() for c in sys.argv[2].split(",") if c.strip())
out=sys.argv[3]; root=sys.argv[4]
cls=json.load(open(root+"/libero/libero/benchmark/task_classification.json"))
suites=["libero_spatial","libero_object","libero_goal","libero_10"]
pairs=[]
for s in suites:
    for tid,e in enumerate(cls[s]):
        if e["category"] in excl: continue
        pairs.append(f"{s}:{tid}")
shards=[[] for _ in range(nproc)]
for i,p in enumerate(pairs): shards[i%nproc].append(p)
for i,sh in enumerate(shards):
    open(f"{out}/shards/shard_{i}.txt","w").write(",".join(sh))
print(f"total pairs {len(pairs)} -> {nproc} shards (~{len(pairs)//max(nproc,1)}/shard)")
PYEOF

echo "launching $NPROC procs, NIS=$NIS, excl='$EXCL', run=$RDIR, step=$STEP"
for idx in $(seq 0 $((NPROC-1))); do
  gpu=$((idx % 8))
  PAIRS=$(cat $OUT/shards/shard_${idx}.txt 2>/dev/null)
  [ -z "$PAIRS" ] && continue
  CKPT_ARGS=()
  if [ -n "$CKPT" ]; then
    CKPT_ARGS+=(--ckpt "$CKPT")
  else
    CKPT_ARGS+=(--step "$STEP")
  fi
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY experiments/libero/cosmos_eval_libero_plus.py \
    --pairs "$PAIRS" --tag "$idx" --num_trials 1 --num_inference_steps $NIS \
    --coupling $CPL --run_dir $RDIR "${CKPT_ARGS[@]}" --out_dir $OUT --exclude_categories "$EXCL" \
    --action_hidden_dim "$ACTION_HIDDEN_DIM" --action_ffn_dim "$ACTION_FFN_DIM" \
    --action_attention_head_dim "$ACTION_ATTENTION_HEAD_DIM" --no-save_videos \
    > $OUT/proc${idx}_gpu${gpu}.log 2>&1 &
  sleep 1
done
wait; echo "ALL-PROCS-DONE"
