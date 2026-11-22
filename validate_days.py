#!/usr/bin/env python3
"""
validate_days.py - sanity-check day_N.csv order-book files before training.

Checks per file (LOBSTER layout: AskP1, AskV1, BidP1, BidV1, AskP2, ...):
  * column count == 4 * levels, every cell numeric (catches headers / blank lines)
  * prices are integers in dollars x 10,000 (tick-aligned: multiples of 100)
  * level-1 sentinel rate (top of book should almost never be empty)
  * crossed books (best ask < best bid) - should be ~0
  * no negative volumes
  * every day is a DIFFERENT day (fingerprint of the first rows) - this catches
    "one day split ten ways" or the same file copied twice
Exit code 1 if any file FAILS.

    python validate_days.py --data-dir data
    python validate_days.py --data-dir data --levels 10 --pattern "day_*.csv"
"""
import argparse
import glob
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd

SENTINEL_ABS = 999_999_000  # |raw price| above this = empty-level placeholder


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]


def check_file(path, levels, chunksize=500_000):
    ncols = 4 * levels
    res = {"file": os.path.basename(path), "status": "PASS", "notes": []}

    def warn(msg):
        if res["status"] == "PASS":
            res["status"] = "WARN"
        res["notes"].append(msg)

    def fail(msg):
        res["status"] = "FAIL"
        res["notes"].append(msg)

    with open(path, "rb") as f:
        head = [f.readline() for _ in range(1000)]
    first = head[0].decode("utf-8", "replace").strip()
    if not first:
        fail("first line is blank")
        return res
    found_cols = first.count(",") + 1
    if found_cols != ncols:
        fail(f"{found_cols} columns, expected {ncols} (levels={levels})")
        return res
    res["fingerprint"] = hashlib.md5(b"".join(head)).hexdigest()

    rows = crossed = locked = l1_sent = tick_bad = tick_chk = neg_vol = 0
    bid_min, bid_max = np.inf, -np.inf
    try:
        reader = pd.read_csv(path, header=None, dtype=np.float64, chunksize=chunksize, sep=",")
        for ch in reader:
            a = ch.to_numpy()
            rows += len(a)
            ask1, bid1 = a[:, 0], a[:, 2]
            ask_real = np.abs(ask1) < SENTINEL_ABS
            bid_real = np.abs(bid1) < SENTINEL_ABS
            l1_sent += int((~ask_real | ~bid_real).sum())
            both = ask_real & bid_real
            crossed += int((ask1[both] < bid1[both]).sum())
            locked += int((ask1[both] == bid1[both]).sum())
            if bid_real.any():
                bid_min = min(bid_min, float(bid1[bid_real].min()))
                bid_max = max(bid_max, float(bid1[bid_real].max()))
            prices = a[:, 0::4][:, :].ravel()
            prices = np.concatenate([prices, a[:, 2::4].ravel()])
            real = np.abs(prices) < SENTINEL_ABS
            pr = prices[real]
            tick_chk += len(pr)
            tick_bad += int((np.mod(pr, 100) != 0).sum())
            vols = np.concatenate([a[:, 1::4].ravel(), a[:, 3::4].ravel()])
            neg_vol += int((vols < 0).sum())
    except ValueError as e:
        fail(f"non-numeric content ({e}). Header row, blank line or decimal prices?")
        return res

    res.update(rows=rows, bid_min=bid_min, bid_max=bid_max)
    if rows < 100_000:
        warn(f"only {rows:,} rows - short for a full AAPL day (smoke-test output?)")
    if tick_chk and tick_bad / tick_chk > 0.001:
        fail(f"{tick_bad / tick_chk:.2%} of prices are not multiples of 100 -> not dollars x 10,000 integers")
    if np.isfinite(bid_max) and not (10_000 <= bid_max <= 100_000_000):
        fail(f"best-bid max {bid_max:,.0f} outside $1-$10,000 x 10,000 range -> wrong price scale")
    if rows and l1_sent / rows > 0.01:
        warn(f"level-1 sentinel in {l1_sent / rows:.2%} of rows")
    if rows and crossed / rows > 0.001:
        fail(f"crossed book (ask < bid) in {crossed / rows:.2%} of rows -> reconstruction bug")
    elif crossed:
        warn(f"{crossed:,} crossed rows")
    if neg_vol:
        fail(f"{neg_vol:,} negative volumes")
    res["locked_pct"] = locked / rows if rows else 0
    res["l1_sent_pct"] = l1_sent / rows if rows else 0
    res["crossed"] = crossed
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pattern", default="day_*.csv")
    ap.add_argument("--levels", type=int, default=10)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data_dir, a.pattern)), key=natural_key)
    if not files:
        sys.exit(f"No files matching {a.pattern} in {a.data_dir}")

    results = []
    print(f"{'file':<14}{'status':<7}{'rows':>12}{'L1 empty':>10}{'crossed':>9}{'best bid $ (min..max)':>26}")
    for p in files:
        r = check_file(p, a.levels)
        results.append(r)
        if "rows" in r:
            span = f"{r['bid_min'] / 1e4:,.2f}..{r['bid_max'] / 1e4:,.2f}"
            print(f"{r['file']:<14}{r['status']:<7}{r['rows']:>12,}{r['l1_sent_pct']:>10.2%}{r['crossed']:>9,}{span:>26}")
        else:
            print(f"{r['file']:<14}{r['status']:<7}")
        for n in r["notes"]:
            print(f"    - {n}")

    seen = {}
    for r in results:
        fp = r.get("fingerprint")
        if fp is None:
            continue
        if fp in seen:
            r["status"] = "FAIL"
            print(f"\nFAIL: {r['file']} starts identically to {seen[fp]} -> same day duplicated, or one day split into pieces.")
        else:
            seen[fp] = r["file"]

    bad = [r for r in results if r["status"] == "FAIL"]
    warn = [r for r in results if r["status"] == "WARN"]
    print(f"\n{len(results) - len(bad) - len(warn)} PASS, {len(warn)} WARN, {len(bad)} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
