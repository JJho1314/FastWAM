#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/users/junjie/FastWAM_cosmos}
VENV=${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}
PY=${PY:-$VENV/bin/python}
RUN_DIR=${RUN_DIR:-$REPO/runs/libero_cosmos_agra_gr00t_post_photometric_aug_no_geo/20260620_145810}
STD_OUT=${STD_OUT:?STD_OUT is required}
PLUS_OUT=${PLUS_OUT:?PLUS_OUT is required}
NIS=${NIS:-10}
NPROC=${NPROC:-8}
PLUS_NPROC=${PLUS_NPROC:-64}

cd "$REPO"
for d in "$VENV"/lib/python3.10/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MAGICK_HOME=${MAGICK_HOME:-/data/users/junjie/im_env}
export LD_LIBRARY_PATH=/data/users/junjie/im_env/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$STD_OUT" "$PLUS_OUT" "$STD_OUT/resume_shards"

echo "[resume-standard] start $(date)"
echo "[resume-standard] run_dir=$RUN_DIR"
echo "[resume-standard] standard_out=$STD_OUT"
echo "[resume-standard] plus_out=$PLUS_OUT"

"$PY" - <<'PY'
import glob
import json
import os

out = os.environ["STD_OUT"]
nproc = int(os.environ.get("NPROC", "8"))
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
done = set()

for path in sorted(glob.glob(out + "/results_*.json") + glob.glob(out + "/results_partial_*.json")):
    try:
        data = json.load(open(path))
    except Exception:
        continue
    for row in data.get("by_task", []):
        done.add((row["suite"], int(row["task_id"])))

all_pairs = [(suite, task_id) for suite in suites for task_id in range(10)]
missing = [pair for pair in all_pairs if pair not in done]
shards = [[] for _ in range(nproc)]
for idx, pair in enumerate(missing):
    shards[idx % nproc].append("%s:%d" % pair)

shard_dir = out + "/resume_shards"
os.makedirs(shard_dir, exist_ok=True)
for idx, shard in enumerate(shards):
    open(f"{shard_dir}/shard_{idx}.txt", "w").write(",".join(shard))

print("[resume-standard] completed_tasks", len(done))
print("[resume-standard] missing_tasks", len(missing), ",".join("%s:%d" % p for p in missing))
for idx, shard in enumerate(shards):
    if shard:
        print("[resume-standard] shard_%d %s" % (idx, ",".join(shard)))
PY

echo "[resume-standard] launching missing standard tasks on $NPROC GPUs"
for idx in $(seq 0 $((NPROC - 1))); do
  pairs=$(cat "$STD_OUT/resume_shards/shard_${idx}.txt" 2>/dev/null || true)
  [ -z "$pairs" ] && continue
  gpu=$((idx % 8))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" experiments/libero/cosmos_eval_libero.py \
    --pairs "$pairs" --num_trials 50 --num_inference_steps "$NIS" \
    --coupling agra --run_dir "$RUN_DIR" --step 21700 \
    --out_dir "$STD_OUT" --tag "resume_${idx}" \
    > "$STD_OUT/proc_resume_${idx}_gpu${gpu}.log" 2>&1 &
done
wait

echo "[resume-standard] workers done $(date); aggregating"
"$PY" - <<'PY'
import glob
import json
import os

out = os.environ["STD_OUT"]
paths = []
paths.extend(sorted(p for p in glob.glob(out + "/results_*.json") if "partial" not in os.path.basename(p)))
paths.extend(sorted(glob.glob(out + "/results_partial_*.json")))

seen = set()
by_suite = {}
by_task = []
tot_s = 0
tot_t = 0
for path in paths:
    try:
        data = json.load(open(path))
    except Exception:
        continue
    for row in data.get("by_task", []):
        key = (row["suite"], int(row["task_id"]))
        if key in seen:
            continue
        seen.add(key)
        by_task.append(row)
        successes = int(row["successes"])
        trials = int(row["trials"])
        item = by_suite.setdefault(row["suite"], {"successes": 0, "trials": 0})
        item["successes"] += successes
        item["trials"] += trials
        tot_s += successes
        tot_t += trials

for item in by_suite.values():
    item["rate"] = item["successes"] / max(item["trials"], 1)

summary = {
    "tasks_aggregated": len(seen),
    "by_suite": by_suite,
    "by_task": by_task,
    "overall": {
        "successes": tot_s,
        "trials": tot_t,
        "rate": tot_s / max(tot_t, 1),
    },
}
json.dump(summary, open(out + "/summary_aggregate.json", "w"), indent=2)
for suite in sorted(by_suite):
    item = by_suite[suite]
    print("[standard] %-16s %4d/%4d = %.2f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
print("[standard] tasks %d OVERALL %d/%d = %.2f%%" % (len(seen), tot_s, tot_t, 100 * summary["overall"]["rate"]))
if len(seen) != 40 or tot_t != 2000:
    raise SystemExit("standard aggregation incomplete: tasks=%d trials=%d" % (len(seen), tot_t))
PY

echo "[plus] launching full LIBERO-Plus, no excluded categories $(date)"
NPROC="$PLUS_NPROC" EXCL="" CPL=agra RDIR="$RUN_DIR" OUT="$PLUS_OUT" \
  LIBERO_PLUS_ROOT=/data/users/junjie/LIBERO-plus \
  bash experiments/libero/run_cosmos_eval_plus_par.sh

echo "[plus] workers done $(date); aggregating"
"$PY" experiments/libero/combine_plus.py "$PLUS_OUT" | tee "$PLUS_OUT/summary_aggregate.txt"
"$PY" - <<'PY'
import glob
import json
import os

out = os.environ["PLUS_OUT"]
seen = set()
by_suite = {}
by_category = {}
by_task = []
tot_s = 0
tot_t = 0
for path in sorted(glob.glob(out + "/results_partial_*.json")):
    data = json.load(open(path))
    for row in data.get("by_task", []):
        key = (row["suite"], int(row["task_id"]))
        if key in seen:
            continue
        seen.add(key)
        by_task.append(row)
        successes = int(row["successes"])
        trials = int(row["trials"])
        tot_s += successes
        tot_t += trials
        suite_item = by_suite.setdefault(row["suite"], {"successes": 0, "trials": 0})
        suite_item["successes"] += successes
        suite_item["trials"] += trials
        cat_item = by_category.setdefault(row.get("category"), {"successes": 0, "trials": 0})
        cat_item["successes"] += successes
        cat_item["trials"] += trials

for group in (by_suite, by_category):
    for item in group.values():
        item["rate"] = item["successes"] / max(item["trials"], 1)

summary = {
    "tasks_aggregated": len(seen),
    "by_suite": by_suite,
    "by_category": by_category,
    "by_task": by_task,
    "overall": {
        "successes": tot_s,
        "trials": tot_t,
        "rate": tot_s / max(tot_t, 1),
    },
}
json.dump(summary, open(out + "/summary_aggregate.json", "w"), indent=2)
print("[plus-json] tasks %d OVERALL %d/%d = %.2f%%" % (len(seen), tot_s, tot_t, 100 * summary["overall"]["rate"]))
PY

echo "[eval] ALL DONE $(date)"
