#!/usr/bin/env python3
"""Generate synthetic LOBSTER-format day files (day_1.csv ...) to smoke-test the pipeline.
Book imbalance is persistent and nudges the mid-price, so OFI has real (weak) predictive signal."""
import argparse
import os

import numpy as np

SENT = 9_999_999_999


def make_book(n, levels=10, seed=0, p_empty_deep=0.05):
    rng = np.random.default_rng(seed)
    mid = 1_500_000 + rng.integers(-300_000, 300_000)
    mid = mid // 100 * 100
    imb = 0.0
    rows = np.empty((n, 4 * levels), dtype=np.int64)
    for i in range(n):
        imb = 0.97 * imb + rng.normal(0, 0.3)
        p_up = 1 / (1 + np.exp(-2 * imb))
        if rng.random() < 0.15:
            mid += 100 * (1 if rng.random() < p_up else -1)
        for L in range(levels):
            a, b = mid + 100 * (L + 1), mid - 100 * (L + 1)
            qa = max(1, int(rng.exponential(200) * (1 - 0.4 * np.tanh(imb))))
            qb = max(1, int(rng.exponential(200) * (1 + 0.4 * np.tanh(imb))))
            if L >= 2 and rng.random() < p_empty_deep:
                rows[i, 4 * L:4 * L + 4] = [-SENT, 0, SENT, 0]     # LOBSTER-style placeholders
            else:
                rows[i, 4 * L:4 * L + 4] = [a, qa, b, qb]
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_synth")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--rows", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for d in range(1, a.days + 1):
        np.savetxt(os.path.join(a.out_dir, f"day_{d}.csv"), make_book(a.rows, seed=d), fmt="%d", delimiter=",")
    print(f"Wrote {a.days} files to {a.out_dir}")
