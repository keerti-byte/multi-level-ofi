#!/usr/bin/env python3
"""Compare two train.py result files for the SAME test day (e.g. 8 epochs vs 20 epochs).

    python compare_runs.py results_levels.json results_long_day5.json
    python compare_runs.py results_day2.json results_long_day2.json --key mlp_r2

Matches level settings present in both files; shows epochs run, R^2 per horizon in each file, the change,
and the level-5-vs-level-1 gain in each file. Default key: lstm_r2.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("a"), ap.add_argument("b")
ap.add_argument("--key", default="lstm_r2")
args = ap.parse_args()

A, B = (json.load(open(f)) for f in (args.a, args.b))
ra = {r["n_levels"]: r for r in A["results"]}
rb = {r["n_levels"]: r for r in B["results"]}
hz = B["args"]["horizons"]
common = sorted(set(ra) & set(rb))
if not common:
    raise SystemExit("No level setting appears in both files.")
for f, d in ((args.a, A), (args.b, B)):
    a = d["args"]
    print(f"{f}: epochs cap {a['epochs']}, seed {a['seed']}, test_day {a.get('test_day')}, "
          f"train_stride {a.get('train_stride')}")
print(f"\n{'levels':<8}{'':<10}" + "".join(f"{str(h) + '-tick':>12}" for h in hz) + f"{'epochs run':>14}{'best val':>11}")
for lv in common:
    x, y = ra[lv], rb[lv]
    print(f"{lv:<8}{'file A':<10}" + "".join(f"{v:>12.5f}" for v in x[args.key]) + f"{x.get('epochs_run', ''):>14}{x.get('best_val_loss', 0):>11.4f}")
    print(f"{'':<8}{'file B':<10}" + "".join(f"{v:>12.5f}" for v in y[args.key]) + f"{y.get('epochs_run', ''):>14}{y.get('best_val_loss', 0):>11.4f}")
    print(f"{'':<8}{'change':<10}" + "".join(f"{q - p:>+12.5f}" for p, q in zip(x[args.key], y[args.key])))
if 1 in common and 5 in common:
    print("\ngain of 5 levels over 1 level:")
    for name, r in (("file A", ra), ("file B", rb)):
        print(f"  {name:<8}" + "".join(f"{a5 - a1:>+12.5f}" for a1, a5 in zip(r[1][args.key], r[5][args.key])))
