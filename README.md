# Multi-level order flow imbalance for mid-price forecasting

Does order-flow imbalance (OFI) from deeper levels of the limit order book improve short-horizon
mid-price forecasts beyond level-1 OFI? This repo reconstructs NASDAQ order books from raw
TotalView-ITCH 5.0 data, computes per-level OFI (Cont, Kukanov & Stoikov), and compares a single-layer LSTM
trained on level-1 OFI against one trained on levels 1-5, at 10 / 20 / 50-tick horizons.

## Preliminary results

AAPL, five NASDAQ sample days (2019-01-30, 03-27, 07-30, 10-30, 12-30), chronological whole-day split:
train = first three days, validation = 10-30, **test = 12-30 only**. Out-of-sample R^2 on the test day,
LSTM (32 units), mean over 3 training seeds:

| OFI levels | 10-tick | 20-tick | 50-tick |
|---|---|---|---|
| 1 | 0.0597 | 0.0585 | 0.0509 |
| 1-5 | 0.0956 | 0.0911 | 0.0705 |
| gain (std over seeds) | +0.0359 (0.0024) | +0.0326 (0.0021) | +0.0197 (0.0014) |

The gain was positive in all 3 seeds at every horizon. A linear baseline (OLS on window-summed OFI) shows no
such gain (-0.002). Training used every 10th window (`--train-stride 10`); the test set is never subsampled.

**Limitations:** a single stock, a single test day, and seeds vary only training randomness, not the data.
Treat this as a first result, not a general finding; see "Next steps".

## Pipeline

```
download_itch.py   NASDAQ ITCH 5.0 .gz  (emi.nasdaq.com)  ->  itch_raw/
itch2lobster.cpp   fast C++ single-symbol ITCH -> LOBSTER-format order book
build_days.py      runs the reconstructor per day          ->  data/day_N.csv + manifest.json
validate_days.py   format / price scale / crossed books / duplicate-day checks
train.py           OFI -> windows -> LSTM (+ linear baseline), level sweep, R^2 table
summarize_results.py   mean/std of R^2 and of the level-5-vs-level-1 gain across seeds
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
- Test on more days (walk-forward / leave-one-day-out) and on other stocks.
- Sweep depth (levels 1, 2, 3, 5, 10) to see where the gain saturates.
- Stronger baselines (e.g. a small MLP on window-summed OFI) to separate "nonlinearity" from "depth".
