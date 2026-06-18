"""Aggregate LIBERO-Plus parallel-eval shards -> per-dimension + per-suite + overall."""
import glob, json, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/users/junjie/FastWAM_cosmos/evaluate_results/cosmos_agra_gr00t_plus"

by_cat, by_suite = {}, {}
tot_s = tot_t = 0
ntasks = 0
seen = set()
for f in sorted(glob.glob(OUT + "/results_partial_*.json")):
    d = json.load(open(f))
    for r in d.get("by_task", []):
        key = (r["suite"], r["task_id"])
        if key in seen:  # guard against overlap
            continue
        seen.add(key)
        ntasks += 1
        s, t = int(r["successes"]), int(r["trials"])
        tot_s += s; tot_t += t
        c = by_cat.setdefault(r.get("category"), [0, 0]); c[0] += s; c[1] += t
        b = by_suite.setdefault(r["suite"], [0, 0]); b[0] += s; b[1] += t

print("tasks aggregated:", ntasks)
print("\n== per dimension ==")
for cat in sorted(by_cat, key=lambda k: (k is None, k)):
    s, t = by_cat[cat]
    print("  %-22s %5d/%5d = %5.1f%%" % (cat, s, t, 100 * s / max(t, 1)))
print("\n== per suite ==")
for su in sorted(by_suite):
    s, t = by_suite[su]
    print("  %-16s %5d/%5d = %5.1f%%" % (su, s, t, 100 * s / max(t, 1)))
print("\nOVERALL %d/%d = %.2f%%" % (tot_s, tot_t, 100 * tot_s / max(tot_t, 1)))
