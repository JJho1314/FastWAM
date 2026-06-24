#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/users/junjie/FastWAM_cosmos}
VENV=${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}
PY=${PY:-$VENV/bin/python}
RUN_DIR=${RUN_DIR:-$REPO/runs/libero_cosmos_agra_gr00t_post_photometric_aug_no_geo/20260620_145810}
STD_OUT=${STD_OUT:?STD_OUT is required}
PLUS_OUT=${PLUS_OUT:?PLUS_OUT is required}
NIS=${NIS:-10}
CHUNK=${CHUNK:-10}
MAX_PROCS=${MAX_PROCS:-64}
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

mkdir -p "$STD_OUT/trial_shards" "$PLUS_OUT"

echo "[trial-shards] start $(date)"
echo "[trial-shards] standard_out=$STD_OUT"
echo "[trial-shards] plus_out=$PLUS_OUT"

"$PY" - <<'PY'
import glob
import json
import os

out = os.environ["STD_OUT"]
chunk = int(os.environ.get("CHUNK", "10"))
max_procs = int(os.environ.get("MAX_PROCS", "64"))
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
covered = {(suite, task_id): set() for suite in suites for task_id in range(10)}

paths = []
paths.extend(sorted(p for p in glob.glob(out + "/results_*.json") if "partial" not in os.path.basename(p)))
paths.extend(sorted(glob.glob(out + "/results_partial_*.json")))
for path in paths:
    try:
        data = json.load(open(path))
    except Exception:
        continue
    for row in data.get("by_task", []):
        suite = row["suite"]
        task_id = int(row["task_id"])
        trials = int(row["trials"])
        start = int(row.get("trial_start", 0))
        if trials >= 50 and "trial_start" not in row:
            start = 0
        covered.setdefault((suite, task_id), set()).update(range(start, min(start + trials, 50)))

jobs = []
for suite in suites:
    for task_id in range(10):
        missing = sorted(set(range(50)) - covered[(suite, task_id)])
        start = None
        prev = None
        spans = []
        for trial in missing:
            if start is None:
                start = prev = trial
            elif trial == prev + 1:
                prev = trial
            else:
                spans.append((start, prev + 1))
                start = prev = trial
        if start is not None:
            spans.append((start, prev + 1))
        for lo, hi in spans:
            cur = lo
            while cur < hi:
                end = min(cur + chunk, hi)
                jobs.append((suite, task_id, cur, end - cur))
                cur = end

shard_dir = out + "/trial_shards"
os.makedirs(shard_dir, exist_ok=True)
for idx, job in enumerate(jobs):
    suite, task_id, trial_start, num_trials = job
    open(f"{shard_dir}/job_{idx:04d}.txt", "w").write(f"{suite}:{task_id}:{trial_start}:{num_trials}")

print("[trial-shards] existing_trial_units", sum(len(v) for v in covered.values()))
print("[trial-shards] remaining_jobs", len(jobs), "remaining_trials", sum(j[3] for j in jobs))
print("[trial-shards] launching_jobs", min(len(jobs), max_procs))
PY

mapfile -t jobs < <(find "$STD_OUT/trial_shards" -maxdepth 1 -type f -name 'job_*.txt' | sort)
running=0
idx=0
for job_file in "${jobs[@]}"; do
  spec=$(cat "$job_file")
  IFS=: read -r suite task_id trial_start num_trials <<<"$spec"
  gpu=$((idx % 8))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" experiments/libero/cosmos_eval_libero.py \
    --pairs "${suite}:${task_id}" --num_trials "$num_trials" --trial_start "$trial_start" \
    --num_inference_steps "$NIS" --coupling agra --run_dir "$RUN_DIR" --step 21700 \
    --out_dir "$STD_OUT" --tag "trial_${idx}_${suite}_${task_id}_${trial_start}" --no-save_videos \
    > "$STD_OUT/proc_trial_${idx}_gpu${gpu}.log" 2>&1 &
  idx=$((idx + 1))
  running=$((running + 1))
  if [ "$running" -ge "$MAX_PROCS" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

echo "[trial-shards] workers done $(date); aggregating"
"$PY" - <<'PY'
import glob
import json
import os

out = os.environ["STD_OUT"]
paths = []
paths.extend(sorted(p for p in glob.glob(out + "/results_*.json") if "partial" not in os.path.basename(p)))
paths.extend(sorted(glob.glob(out + "/results_partial_*.json")))
by_task = {}
covered = {}
for path in paths:
    try:
        data = json.load(open(path))
    except Exception:
        continue
    for row in data.get("by_task", []):
        key = (row["suite"], int(row["task_id"]))
        trials = int(row["trials"])
        start = int(row.get("trial_start", 0))
        if trials >= 50 and "trial_start" not in row:
            start = 0
        trial_ids = set(range(start, min(start + trials, 50)))
        seen_trials = covered.setdefault(key, set())
        if trial_ids and trial_ids.issubset(seen_trials):
            continue
        if trial_ids & seen_trials:
            raise SystemExit(f"overlapping partial trial interval for {key}: {sorted(trial_ids & seen_trials)}")
        seen_trials.update(trial_ids)
        item = by_task.setdefault(key, {
            "suite": row["suite"],
            "task_id": int(row["task_id"]),
            "successes": 0,
            "trials": 0,
            "desc": row.get("desc"),
        })
        item["successes"] += int(row["successes"])
        item["trials"] += trials

suite_totals = {}
tot_s = 0
tot_t = 0
rows = []
for key in sorted(by_task):
    item = by_task[key]
    if item["trials"] != 50:
        raise SystemExit(f"incomplete task {key}: trials={item['trials']}")
    item["rate"] = item["successes"] / item["trials"]
    rows.append(item)
    suite_item = suite_totals.setdefault(item["suite"], {"successes": 0, "trials": 0})
    suite_item["successes"] += item["successes"]
    suite_item["trials"] += item["trials"]
    tot_s += item["successes"]
    tot_t += item["trials"]

for item in suite_totals.values():
    item["rate"] = item["successes"] / max(item["trials"], 1)

summary = {
    "tasks_aggregated": len(rows),
    "by_suite": suite_totals,
    "by_task": rows,
    "overall": {
        "successes": tot_s,
        "trials": tot_t,
        "rate": tot_s / max(tot_t, 1),
    },
}
json.dump(summary, open(out + "/summary_aggregate.json", "w"), indent=2)
for suite in sorted(suite_totals):
    item = suite_totals[suite]
    print("[standard] %-16s %4d/%4d = %.2f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
print("[standard] tasks %d OVERALL %d/%d = %.2f%%" % (len(rows), tot_s, tot_t, 100 * summary["overall"]["rate"]))
if len(rows) != 40 or tot_t != 2000:
    raise SystemExit("standard aggregation incomplete")
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
