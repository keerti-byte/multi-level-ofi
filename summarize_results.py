#!/usr/bin/env python3
"""Summarise several train.py runs (one results_*.json per seed, or per test day with --test-day).

    python summarize_results.py                 # reads results_*.json in the current folder
    python summarize_results.py results_1.json results_2.json
Shows, per horizon, the out-of-sample R^2 of each level setting and the gain of the deepest setting
over the shallowest, across runs (mean, std, how many runs had a positive gain, and the per-run gains).
"""
import glob
import json
import sys

import numpy as np

files = sys.argv[1:] or sorted(glob.glob("results_*.json"))
if not files:
    sys.exit("No results_*.json files found. Run train.py with --out results_<seed>.json first.")

runs = []
for f in files:
    d = json.load(open(f))
    runs.append({"file": f, "test_day": d["args"].get("test_day"), "horizons": d["args"]["horizons"], "res": {r["n_levels"]: r for r in d["results"]}})

horizons = runs[0]["horizons"]
levels = sorted(runs[0]["res"])
for key, label in (("lstm_r2", "LSTM"), ("linear_r2", "Linear baseline")):
    if not all(key in r["res"][lv] for r in runs for lv in levels):
        continue
    print(f"\n=== {label}: out-of-sample R^2 (test day), {len(runs)} run(s) ===")
    print(f"{'':<14}" + "".join(f"{str(h) + '-tick':>14}" for h in horizons))
    for lv in levels:
        arr = np.array([r["res"][lv][key] for r in runs])
        print(f"{'L' + str(lv) + ' mean':<14}" + "".join(f"{m:>14.5f}" for m in arr.mean(0)))
        if len(runs) > 1:
            print(f"{'  std':<14}" + "".join(f"{s:>14.5f}" for s in arr.std(0, ddof=1)))
    if len(levels) > 1:
        lo, hi = levels[0], levels[-1]
        gain = np.array([np.array(r["res"][hi][key]) - np.array(r["res"][lo][key]) for r in runs])
        print(f"{'gain L' + str(hi) + '-L' + str(lo):<14}" + "".join(f"{m:>+14.5f}" for m in gain.mean(0)))
        if len(runs) > 1:
            print(f"{'  gain std':<14}" + "".join(f"{s:>14.5f}" for s in gain.std(0, ddof=1)))
            print(f"{'  runs >0':<14}" + "".join(f"{int((gain[:, j] > 0).sum()):>10d}/{len(runs):<3d}" for j in range(len(horizons))))
            print("  per-run gain:")
            for r, g in zip(runs, gain):
                tag = f"test_day={r['test_day']}" if r["test_day"] else r["file"]
                print(f"    {tag:<22}" + "".join(f"{x:>+14.5f}" for x in g))
print()
