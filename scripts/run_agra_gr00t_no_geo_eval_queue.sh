#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/data/users/junjie/FastWAM_cosmos}
VENV=${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}
PY=${PY:-$VENV/bin/python}
RUN_DIR=${RUN_DIR:-$REPO/runs/libero_cosmos_agra_gr00t_post_photometric_aug_no_geo/20260620_145810}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
STD_OUT=${STD_OUT:-$REPO/evaluate_results/agra_gr00t_post_no_geo_standard50_$TS}
PLUS_OUT=${PLUS_OUT:-$REPO/evaluate_results/agra_gr00t_post_no_geo_plus_full10030_$TS}
NIS=${NIS:-10}
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

mkdir -p "$STD_OUT" "$PLUS_OUT"

echo "[eval] start $(date)"
echo "[eval] run_dir=$RUN_DIR"
echo "[eval] standard_out=$STD_OUT"
echo "[eval] plus_out=$PLUS_OUT"

echo "[standard] launching 4 suite workers, num_trials=50"
suites=(libero_spatial libero_object libero_goal libero_10)
for idx in 0 1 2 3; do
  suite=${suites[$idx]}
  CUDA_VISIBLE_DEVICES=$idx "$PY" experiments/libero/cosmos_eval_libero.py \
    --suites "$suite" --task_ids all --num_trials 50 --num_inference_steps "$NIS" \
    --coupling agra --run_dir "$RUN_DIR" --step 21700 \
    --out_dir "$STD_OUT" --tag "$suite" \
    > "$STD_OUT/proc_${suite}_gpu${idx}.log" 2>&1 &
done
wait

echo "[standard] workers done $(date); aggregating"
"$PY" - <<'PY'
import glob
import json
import os

out = os.environ["STD_OUT"]
files = sorted(
    f for f in glob.glob(out + "/results_*.json")
    if "partial" not in os.path.basename(f)
)
by_suite = {}
by_task = []
tot_s = 0
tot_t = 0
for path in files:
    data = json.load(open(path))
    by_task.extend(data.get("by_task", []))
    for suite, item in data.get("by_suite", {}).items():
        accum = by_suite.setdefault(suite, {"successes": 0, "trials": 0})
        accum["successes"] += int(item["successes"])
        accum["trials"] += int(item["trials"])

for item in by_suite.values():
    item["rate"] = item["successes"] / max(item["trials"], 1)
    tot_s += item["successes"]
    tot_t += item["trials"]

summary = {
    "files": files,
    "by_suite": by_suite,
    "by_task": by_task,
    "overall": {
        "successes": tot_s,
        "trials": tot_t,
        "rate": tot_s / max(tot_t, 1),
    },
}
json.dump(summary, open(out + "/summary_aggregate.json", "w"), indent=2)
print("[standard] files", len(files))
for suite in sorted(by_suite):
    item = by_suite[suite]
    print(
        "[standard] %-16s %4d/%4d = %.2f%%"
        % (suite, item["successes"], item["trials"], 100 * item["rate"])
    )
print("[standard] OVERALL %d/%d = %.2f%%" % (tot_s, tot_t, 100 * summary["overall"]["rate"]))
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
by_cat = {}
by_suite = {}
seen = set()
tasks = []
tot_s = 0
tot_t = 0
for path in sorted(glob.glob(out + "/results_partial_*.json")):
    data = json.load(open(path))
    for row in data.get("by_task", []):
        key = (row["suite"], row["task_id"])
        if key in seen:
            continue
        seen.add(key)
        tasks.append(row)
        successes = int(row["successes"])
        trials = int(row["trials"])
        tot_s += successes
        tot_t += trials
        suite_item = by_suite.setdefault(row["suite"], {"successes": 0, "trials": 0})
        suite_item["successes"] += successes
        suite_item["trials"] += trials
        cat_item = by_cat.setdefault(row.get("category"), {"successes": 0, "trials": 0})
        cat_item["successes"] += successes
        cat_item["trials"] += trials

for group in (by_suite, by_cat):
    for item in group.values():
        item["rate"] = item["successes"] / max(item["trials"], 1)

summary = {
    "tasks_aggregated": len(tasks),
    "by_suite": by_suite,
    "by_category": by_cat,
    "overall": {
        "successes": tot_s,
        "trials": tot_t,
        "rate": tot_s / max(tot_t, 1),
    },
    "by_task": tasks,
}
json.dump(summary, open(out + "/summary_aggregate.json", "w"), indent=2)
print("[plus-json] tasks", len(tasks), "overall %.2f%%" % (100 * summary["overall"]["rate"]))
PY

echo "[eval] ALL DONE $(date)"
