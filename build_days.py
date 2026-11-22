#!/usr/bin/env python3
"""
build_days.py - turn downloaded ITCH files into one LOBSTER-format order-book
CSV per trading day, named so train.py can load them.

For every MMDDYYYY.NASDAQ_ITCH50.gz in --itch-dir it reconstructs the --symbol order
book with either the fast C++ engine (itch2lobster, minutes per day) or your Python
reconstruct_lobster_book.py (hours per day), then exposes the results as

    data/raw/AAPL_2019-01-30_orderbook_10.csv     (descriptive name)
    data/day_1.csv, day_2.csv, ...                (chronological, for train.py)
    data/manifest.json                            (day_N -> date, source, rows)

Then train with e.g.
    python train.py --data-dir data --day-pattern "day_{}.csv" --n-levels 5 --epochs 10 --batch-size 512

Safe to re-run: finished days are skipped; partial output is written to a
.part file and only renamed on success.

Build the fast engine once:   g++ -O3 -march=native -std=c++17 -o itch2lobster itch2lobster.cpp -lz
Smoke test one day first (2M messages, written to data/smoke/):
    python build_days.py --days 01302019 --max-messages 2000000
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import time

PAT = re.compile(r"^(\d{2})(\d{2})(\d{4})\.NASDAQ_ITCH50\.gz$")


def find_days(itch_dir, wanted):
    found = []
    for fn in os.listdir(itch_dir):
        m = PAT.match(fn)
        if not m:
            continue
        mm, dd, yyyy = m.groups()
        key = f"{mm}{dd}{yyyy}"
        if wanted and key not in wanted:
            continue
        found.append((f"{yyyy}-{mm}-{dd}", key, os.path.join(itch_dir, fn)))
    return sorted(found)  # ISO date sorts chronologically


def count_rows(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def engine_command(args):
    """Return the argv prefix for the chosen reconstruction engine."""
    cpp = args.cpp_binary
    have_cpp = os.path.isfile(cpp) and os.access(cpp, os.X_OK)
    if args.engine == "cpp" or (args.engine == "auto" and have_cpp):
        if not have_cpp:
            sys.exit(f"C++ engine not found at {cpp}. Build it:\n"
                     "  g++ -O3 -march=native -std=c++17 -o itch2lobster itch2lobster.cpp -lz")
        return "cpp", [cpp]
    if not os.path.isfile(args.reconstruct_script):
        sys.exit(f"No engine available: {cpp} not built and {args.reconstruct_script} not found.")
    return "python", [sys.executable, args.reconstruct_script]


def run_one(iso_date, src, args, out_dir, log_dir):
    name = f"{args.symbol}_{iso_date}_orderbook_{args.levels}.csv"
    final = os.path.join(out_dir, name)
    part = final + ".part"
    if os.path.exists(final) and os.path.getsize(final) > 0:
        return iso_date, final, "skipped (already built)"
    os.makedirs(out_dir, exist_ok=True)
    kind, prefix = engine_command(args)
    cmd = prefix + [
        "--input", src, "--output", part,
        "--symbol", args.symbol, "--levels", str(args.levels),
    ]
    if args.max_messages:
        cmd += ["--max-messages", str(args.max_messages)]
        if kind == "cpp":
            cmd += ["--all-day"]   # early messages are pre-market; don't filter to 09:30-16:00
    if args.progress_every:
        cmd += ["--progress-every", str(args.progress_every)]
    log = os.path.join(log_dir, f"{iso_date}.log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(part) or os.path.getsize(part) == 0:
        raise RuntimeError(f"{iso_date}: reconstruct failed (exit {rc}); see {log}")
    os.replace(part, final)
    return iso_date, final, f"built in {(time.time() - t0) / 60:.1f} min"


def link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--itch-dir", default="itch_raw")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--levels", type=int, default=10, help="book depth to write (>= --n-levels used in training)")
    ap.add_argument("--engine", choices=["auto", "cpp", "python"], default="auto",
                    help="auto = C++ if ./itch2lobster exists, else the Python script")
    ap.add_argument("--cpp-binary", default="./itch2lobster")
    ap.add_argument("--reconstruct-script", default="reconstruct_lobster_book.py")
    ap.add_argument("--days", nargs="*", help="only these MMDDYYYY dates")
    ap.add_argument("--jobs", type=int, default=1, help="parallel days (each needs 1 core; RAM is modest)")
    ap.add_argument("--max-messages", type=int, default=0, help="smoke-test: stop after N messages (output goes to <out-dir>/smoke/)")
    ap.add_argument("--progress-every", type=int, default=0)
    args = ap.parse_args()

    kind, _ = engine_command(args)
    print(f"Engine: {kind}")
    if not os.path.isdir(args.itch_dir):
        sys.exit(f"Missing {args.itch_dir}. Run download_itch.py first.")
    days = find_days(args.itch_dir, set(args.days) if args.days else None)
    if not days:
        sys.exit(f"No MMDDYYYY.NASDAQ_ITCH50.gz files found in {args.itch_dir}")

    smoke = bool(args.max_messages)
    raw_dir = os.path.join(args.out_dir, "smoke" if smoke else "raw")
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"{len(days)} day(s): " + ", ".join(d for d, _, _ in days))

    results, failures = {}, []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(run_one, d, p, args, raw_dir, log_dir): d for d, _, p in days}
        for fut in cf.as_completed(futs):
            d = futs[fut]
            try:
                iso_date, path, msg = fut.result()
                results[iso_date] = path
                print(f"  {iso_date}: {msg}")
            except Exception as e:
                failures.append(d)
                print(f"  {d}: FAILED - {e}")

    if smoke:
        print(f"\nSmoke output in {raw_dir}. Inspect it (first rows, price scale), then run without --max-messages.")
        return 1 if failures else 0

    # Chronological day_N.csv over everything that exists in raw/ (not just this run),
    # so numbering stays consistent if you add days later.
    all_raw = sorted(
        f for f in os.listdir(raw_dir)
        if re.match(rf"^{re.escape(args.symbol)}_\d{{4}}-\d{{2}}-\d{{2}}_orderbook_{args.levels}\.csv$", f)
    )
    manifest = {}
    for i, fn in enumerate(all_raw, start=1):
        dst = os.path.join(args.out_dir, f"day_{i}.csv")
        link_or_copy(os.path.join(raw_dir, fn), dst)
        date = fn.split("_")[1]
        manifest[f"day_{i}.csv"] = {"date": date, "source": fn, "rows": count_rows(dst)}
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nManifest:")
    for k, v in manifest.items():
        print(f"  {k:<11} {v['date']}  {v['rows']:>10,} rows")
    if failures:
        print("\nFailed:", ", ".join(failures), "- re-run to retry only those.")
        return 1
    print(f"\nNext: python validate_days.py --data-dir {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
