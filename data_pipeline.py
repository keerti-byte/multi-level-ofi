"""
Load LOBSTER order-book days and turn each into (features, targets).

features : (M, n_levels)  per-tick OFI for levels 1..n_levels (shares)
targets  : (M, H)         mid-price change over each horizon h, in TICKS (cents):
                          (mid[t+h] - mid[t]) / 0.01      (NaN where t+h runs off the day)

Rows whose TOP OF BOOK (level 1, either side) is an empty-level placeholder are dropped,
because mid-price would be nonsense. Empty deeper levels are kept (their OFI is 0).
Each day is processed separately so nothing ever crosses a day boundary.
"""
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ofi import SENTINEL_ABS, is_real, multi_level_ofi

PRICE_SCALE = 10_000.0   # LOBSTER price = dollars x 10,000
TICK = 0.01              # US equity tick size in dollars


def load_day(path: str, n_levels: int, max_rows: int = None) -> np.ndarray:
    ncols = 4 * n_levels
    try:
        df = pd.read_csv(path, header=None, usecols=range(ncols), dtype=np.float64, nrows=max_rows)
    except ValueError as e:
        raise ValueError(
            f"{path}: could not read {ncols} numeric columns ({e}). "
            "Expected a header-less LOBSTER orderbook CSV."
        ) from e
    return df.to_numpy()


def build_day(raw: np.ndarray, n_levels: int, horizons: Sequence[int]) -> Dict:
    finite = np.isfinite(raw).all(axis=1)
    top_ok = is_real(raw[:, 0]) & is_real(raw[:, 2])
    keep = finite & top_ok
    n_dropped = int((~keep).sum())
    raw = raw[keep]
    if len(raw) < 2:
        raise ValueError("day has fewer than 2 usable rows")

    ofi = multi_level_ofi(raw, n_levels)[1:]                    # row 0 has no predecessor
    mid = ((raw[:, 0] + raw[:, 2]) / 2.0 / PRICE_SCALE)[1:]     # dollars, aligned with ofi
    M = len(mid)

    targets = np.full((M, len(horizons)), np.nan, dtype=np.float64)
    for j, h in enumerate(horizons):
        if h < M:
            targets[: M - h, j] = (mid[h:] - mid[: M - h]) / TICK
    return {
        "features": ofi.astype(np.float32),
        "targets": targets.astype(np.float32),
        "n_dropped": n_dropped,
        "n_rows": M,
    }


def find_day_files(data_dir: str, pattern: str = "day_{}.csv") -> List[str]:
    files, i = [], 1
    while True:
        p = os.path.join(data_dir, pattern.format(i))
        if not os.path.exists(p):
            break
        files.append(p)
        i += 1
    if not files:
        raise FileNotFoundError(f"No files like {os.path.join(data_dir, pattern.format(1))} found")
    return files


def load_all_days(data_dir, pattern, n_levels, horizons, max_rows=None, verbose=True) -> List[Dict]:
    days = []
    for p in find_day_files(data_dir, pattern):
        d = build_day(load_day(p, n_levels, max_rows), n_levels, horizons)
        d["name"] = os.path.basename(p)
        if verbose:
            print(f"  {d['name']}: {d['n_rows']:,} rows kept, "
                  f"{d['n_dropped']:,} dropped (level-1 empty/non-finite)")
        days.append(d)
    return days
