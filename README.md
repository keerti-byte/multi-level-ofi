# Multi-level order flow imbalance for mid-price forecasting

Does order-flow imbalance (OFI) from deeper levels of the limit order book improve short-horizon
mid-price forecasts beyond level-1 OFI? This repo reconstructs NASDAQ order books from raw
TotalView-ITCH 5.0 data, computes per-level OFI (Cont, Kukanov & Stoikov), and compares a single-layer LSTM
trained on level-1 OFI against one trained on levels 1-5, at 10 / 20 / 50-tick horizons.

## Preliminary results

AAPL, five NASDAQ sample days (2019-01-30, 03-27, 07-30, 10-30, 12-30), chronological whole-day split:
train = first three days, validation = 10-30, **test = 12-30 only**. Out-of-sample R^2 on the test day for an LSTM
(32 units) using OFI from the first K book levels (seed 1; epochs capped at 8, every 10th window used for training):

| OFI levels | 10-tick | 20-tick | 50-tick |
|---|---|---|---|
| 1 | 0.0600 | 0.0588 | 0.0512 |
| 2 | 0.0848 | 0.0804 | 0.0637 |
| 3 | 0.0919 | 0.0884 | 0.0688 |
| 5 | 0.0962 | 0.0919 | 0.0724 |
| 10 | 0.0963 | 0.0920 | 0.0723 |

Levels 2 and 3 capture roughly 60-68% and 84-89% of the total gain over level 1; **the gain saturates around five
levels** and ten adds nothing. With 3 seeds (levels 1 vs 5) the gain was +0.0359 / +0.0326 / +0.0197 R^2 (std over
seeds 0.002 or less) and positive in every seed at every horizon. A linear baseline (OLS on window-summed OFI) does
not benefit from depth (R^2 falls from 0.0176 at level 1 to 0.0155 at level 10 at the 10-tick horizon).

### Does it hold on other days? (leave-one-day-out)

Same setup (levels 1 vs 5, seed 1), with each day in turn as the unseen test day (`--test-day K`; the validation day
is the previous day, and the model trains on the remaining days, which for early test days includes later days).
Gain in out-of-sample R^2 of 5 levels over 1 level, LSTM:

| Test day | 10-tick | 20-tick | 50-tick |
|---|---|---|---|
| 2019-01-30 | +0.018 | +0.020 | +0.013 |
| 2019-03-27 | +0.035 | +0.032 | +0.023 |
| 2019-07-30 | +0.031 | +0.021 | +0.000 |
| 2019-10-30 | +0.021 | +0.019 | +0.016 |
| 2019-12-30 | +0.036 | +0.033 | +0.021 |
| **mean** | **+0.028** | **+0.025** | **+0.014** |

The gain is positive on every test day at every horizon, but its size varies by day and fades with the horizon
(about zero at 50 ticks on 2019-07-30). Absolute R^2 also varies a lot between days (mean level-1 R^2 over the four
extra test days: 0.037 at 10 ticks with a day-to-day std of about 0.02). The linear baseline has essentially no skill
on those four days (R^2 about 0 at both level 1 and level 5).

**Limitations:** a single stock; one training seed per test day; all runs used the 8-epoch cap, so training may not
have converged; early test days train on later days (blocked cross-validation, not walk-forward). Treat this as a
first result, not a general finding.

## Pipeline

```
download_itch.py   NASDAQ ITCH 5.0 .gz  (emi.nasdaq.com)  ->  itch_raw/
itch2lobster.cpp   fast C++ single-symbol ITCH -> LOBSTER-format order book
build_days.py      runs the reconstructor per day          ->  data/day_N.csv + manifest.json
validate_days.py   format / price scale / crossed books / duplicate-day checks
train.py           OFI -> windows -> LSTM (+ linear baseline), level sweep, R^2 table
mlp_baseline.py    NumPy MLP on window-summed OFI (nonlinear, no sequence order) - baseline for the LSTM
summarize_results.py   mean/std of R^2 and of the level-5-vs-level-1 gain across runs (seeds or test days)
```
Modules: `ofi.py` (OFI per level) | `data_pipeline.py` (load, drop, OFI, targets) |
`windowing.py` (day-safe windows, scaler, splits) | `model_torch.py` / `model.py` (PyTorch / TensorFlow LSTM) |
`tests/` (pytest; includes a byte-for-byte check of the C++ reconstructor against a Python reference).

## Quick start

```bash
pip install -r requirements.txt            # numpy pandas torch pytest
python -m pytest -q tests

# data (about 4-5 GB per day; only ~7 free full-day files exist on NASDAQ's server)
python download_itch.py --list
python download_itch.py --out-dir itch_raw --days 01302019 03272019 07302019 10302019 12302019

# reconstruct AAPL order books (build the C++ engine once; needs g++ and zlib headers)
g++ -O3 -march=native -std=c++17 -o itch2lobster itch2lobster.cpp -lz
python build_days.py --jobs 3
python validate_days.py --data-dir data

# experiments
python train.py --data-dir data --levels-sweep 1,5 --skip-lstm            # linear baseline, seconds
for s in 1 2 3; do
  python train.py --data-dir data --levels-sweep 1,5 --epochs 8 --batch-size 1024 \
      --train-stride 10 --val-stride 10 --seed $s --out results_$s.json
done
python summarize_results.py

# does it hold on other test days? (leave-one-day-out; --test-day K makes day_K the unseen test day)
for d in 1 2 3 4; do
  python train.py --data-dir data --levels-sweep 1,5 --epochs 8 --batch-size 1024 \
      --train-stride 10 --val-stride 10 --seed 1 --test-day $d --out results_day$d.json
done
python summarize_results.py results_day*.json

# is the LSTM's edge from sequence structure or just nonlinearity? (MLP sees only window-summed OFI; no LSTM needed, fast)
for d in 1 2 3 4 5; do
  python train.py --data-dir data --levels-sweep 1,5 --skip-lstm --baseline-mlp \
      --train-stride 10 --val-stride 10 --seed 1 --test-day $d --out results_mlp_day$d.json
done
python summarize_results.py results_mlp_day*.json
```
Quick smoke test without any real data: `python make_synthetic_days.py --out-dir data_synth` then
`python train.py --data-dir data_synth --levels-sweep 1,5 --epochs 1 --max-rows 8000`.

`--backend torch|tf|auto` picks the LSTM implementation (TensorFlow has no wheels for Python 3.14).

## Design decisions
- **Data:** LOBSTER layout `AskP,AskV,BidP,BidV` x levels; price = dollars x 10,000; empty levels are placeholders
  (|price| >= 999,999,000, size 0).
- **OFI:** event-conditional Cont-Kukanov-Stoikov per level; an empty level on a side contributes 0.
  Rows with an empty level 1 are dropped (mid-price undefined).
- **Target:** mid-price change over h ticks, in cents; standardised with train statistics (R^2 is scale-invariant).
- **Features:** per-tick OFI levels 1..K, z-scored with train-only statistics, clipped at +/-10 sigma.
- **Windows:** 50-tick lookback; a window and its target horizon never cross a trading day.
- **Split:** chronological whole days (~70/10/20; 5 days -> 3/1/1); fewer than 3 days -> row split inside each day.
- **Model:** one LSTM(32), no dropout, Dense(3), Adam 1e-3, MSE, early stopping (patience 5).
- **Metric:** out-of-sample R^2 = 1 - SSE / sum((y - mean(y_test))^2), per horizon.
- **C++ reconstructor:** rows written for 09:30-16:00 ET by default (`--all-day` for everything); an ITCH replace is
  one atomic row; a truncated/corrupt `.gz` aborts with exit code 4.

## Data
The raw ITCH files come from NASDAQ's public sample server and are **not included** in this repository
(they are several GB each). Check NASDAQ's terms before redistributing any data derived from them.

## Next steps
- Test on other days (leave-one-day-out via `--test-day`; note early test days train on later days) and other stocks.
- Longer training (epochs > 8) to check the level ranking holds once converged.
- Compare the LSTM with the MLP baseline per test day (`--baseline-mlp`) to see whether sequence order matters.
- Other stocks (`itch2lobster --symbol`) to check the result is not Apple-specific.
