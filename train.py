#!/usr/bin/env python3
"""
Multi-level OFI -> mid-price change forecasting (LOBSTER-format NASDAQ data).

Question: do deeper book levels (OFI levels 1..5) improve out-of-sample R^2 over
level-1 OFI alone, at 10/20/50-tick horizons?  Single-layer LSTM, Adam/MSE,
early stopping (patience 5).

    python train.py --data-dir data --day-pattern "day_{}.csv" --levels-sweep 1,5 --epochs 10 --batch-size 512
    python train.py --data-dir data --n-levels 5 --max-rows 200000          # quick look
    python train.py --data-dir data --levels-sweep 1,5 --skip-lstm          # linear baseline only (no TF)
"""
import argparse
import json
import time

import numpy as np

from data_pipeline import load_all_days
from windowing import Scaler, WindowDataset, make_splits


def r2_score(y_true, y_pred):
    """Out-of-sample R^2 per column: 1 - SSE / sum((y - mean(y_test))^2)."""
    sse = ((y_true - y_pred) ** 2).sum(axis=0)
    sst = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - sse / np.maximum(sst, 1e-12)


def linear_baseline(ds_tr, ds_te, y_std, y_mean):
    """OLS of horizon targets on window-summed OFI (Kolm-style cumulative OFI). Cheap diagnostic."""
    def design(ds):
        S = ds.window_sums()
        return np.hstack([S, np.ones((len(S), 1))])
    Xtr, Xte = design(ds_tr), design(ds_te)
    mu, sd = Xtr[:, :-1].mean(0), np.maximum(Xtr[:, :-1].std(0), 1e-8)
    Xtr[:, :-1] = (Xtr[:, :-1] - mu) / sd
    Xte[:, :-1] = (Xte[:, :-1] - mu) / sd
    beta, *_ = np.linalg.lstsq(Xtr, ds_tr.all_targets(), rcond=None)
    return r2_score(ds_te.all_targets(), Xte @ beta)


def run_one(days, k, args, horizons):
    sub = [{"features": d["features"][:, :k], "targets": d["targets"]} for d in days]
    tr, va, te, desc = make_splits(sub)
    print(f"\n=== n_levels={k} | {desc}")
    sc = Scaler(clip=args.clip).fit(tr)
    max_h = max(horizons)
    tr_s, va_s, te_s = sc.apply(tr), sc.apply(va), sc.apply(te)
    ds_tr_full = WindowDataset(tr_s, args.lookback, max_h)          # linear baseline always uses all windows
    ds_tr = WindowDataset(tr_s, args.lookback, max_h, stride=args.train_stride)
    ds_va = WindowDataset(va_s, args.lookback, max_h, stride=args.val_stride)
    ds_te = WindowDataset(te_s, args.lookback, max_h)               # test is never subsampled
    print(f"  samples: train {len(ds_tr):,} | val {len(ds_va):,} | test {len(ds_te):,}"
          + (f"  (LSTM trains on every {args.train_stride}th window)" if args.train_stride > 1 else ""))
    if min(len(ds_tr), len(ds_va), len(ds_te)) == 0:
        raise SystemExit("A split has zero samples - use more rows/days or a smaller --lookback.")

    out = {"n_levels": k, "samples": [len(ds_tr), len(ds_va), len(ds_te)]}
    out["linear_r2"] = linear_baseline(ds_tr_full, ds_te, sc.y_std, sc.y_mean).tolist()
    print("  linear baseline R^2:", "  ".join(f"{h}t={r:.5f}" for h, r in zip(horizons, out["linear_r2"])))

    if args.skip_lstm:
        return out
    backend = args.backend
    if backend == "auto":
        try:
            import torch  # noqa: F401
            backend = "torch"
        except ImportError:
            try:
                import tensorflow  # noqa: F401
                backend = "tf"
            except ImportError:
                raise SystemExit("No deep-learning backend installed. Run:  pip install torch\n"
                                 "(TensorFlow has no wheels for Python 3.14.)  "
                                 "Or use --skip-lstm for the linear baseline only.")
    t0 = time.time()
    if backend == "torch":
        from model_torch import fit_predict
        res = fit_predict(ds_tr, ds_va, ds_te, k, len(horizons), units=args.lstm_units, lr=args.lr,
                          batch_size=args.batch_size, epochs=args.epochs, seed=args.seed)
        pred, epochs_run, best_val = res["pred"], res["epochs_run"], res["best_val_loss"]
    else:
        import tensorflow as tf
        from model import build_model, make_sequence
        tf.keras.utils.set_random_seed(args.seed)
        model = build_model(args.lookback, k, len(horizons), units=args.lstm_units, lr=args.lr)
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        hist = model.fit(make_sequence(ds_tr, args.batch_size, True, args.seed), epochs=args.epochs,
                         validation_data=make_sequence(ds_va, args.batch_size, False),
                         callbacks=cb, verbose=2)
        pred = model.predict(make_sequence(ds_te, args.batch_size, False), verbose=0)
        epochs_run, best_val = len(hist.history["loss"]), float(min(hist.history["val_loss"]))
    r2 = r2_score(ds_te.all_targets(), pred)
    out.update(lstm_r2=r2.tolist(), epochs_run=epochs_run, best_val_loss=best_val,
               minutes=(time.time() - t0) / 60, backend=backend)
    print(f"  Out-of-sample R^2 (test, levels 1-{k}, {backend}): " + "  ".join(f"{h}t={r:.5f}" for h, r in zip(horizons, r2)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--day-pattern", default="day_{}.csv")
    ap.add_argument("--n-levels", type=int, default=5)
    ap.add_argument("--levels-sweep", default=None, help="e.g. 1,5 -> run each and compare")
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--lookback", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lstm-units", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=10.0, help="clip z-scored OFI at +/-clip (0 = off)")
    ap.add_argument("--train-stride", type=int, default=1, help="LSTM trains on every Nth window (windows overlap ~fully; 5-10 is ~N x faster)")
    ap.add_argument("--val-stride", type=int, default=1, help="same for the validation set (test set is never subsampled)")
    ap.add_argument("--max-rows", type=int, default=None, help="read only first N rows of each day (quick tests)")
    ap.add_argument("--backend", choices=["auto", "torch", "tf"], default="auto",
                    help="auto = PyTorch if installed, else TensorFlow (TF has no Python 3.14 wheels)")
    ap.add_argument("--skip-lstm", action="store_true", help="linear baseline only (no TensorFlow needed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels_sweep.split(",")] if args.levels_sweep else [args.n_levels]
    kmax = max(levels)
    print(f"Loading days from {args.data_dir} (OFI levels 1..{kmax}, horizons {args.horizons})")
    days = load_all_days(args.data_dir, args.day_pattern, kmax, args.horizons, args.max_rows)

    results = [run_one(days, k, args, args.horizons) for k in levels]

    key = "lstm_r2" if not args.skip_lstm else "linear_r2"
    print(f"\n{'levels':<8}" + "".join(f"{str(h) + '-tick':>12}" for h in args.horizons) + f"   ({key})")
    for r in results:
        print(f"{r['n_levels']:<8}" + "".join(f"{v:>12.5f}" for v in r[key]))
    if len(results) > 1:
        base = results[0]
        for r in results[1:]:
            d = np.array(r[key]) - np.array(base[key])
            print(f"gain L{r['n_levels']} vs L{base['n_levels']}: " + "  ".join(f"{x:+.5f}" for x in d))
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
