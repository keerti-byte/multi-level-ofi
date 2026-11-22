#!/usr/bin/env python3
"""
download_itch.py - fetch full-day NASDAQ TotalView-ITCH 5.0 sample files.

Source: NASDAQ's public sample server
        https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/
Files are named MMDDYYYY.NASDAQ_ITCH50.gz (3.5-5.6 GB each, ALL symbols) and
have a sibling <file>.md5sum used here for integrity checking.

Features: HTTP Range resume (safe to Ctrl-C and re-run), retry with backoff,
md5 verification, disk-space check, dry-run.

Examples
    python download_itch.py --list
    python download_itch.py --list --live          # ask the server what exists now
    python download_itch.py --dry-run
    python download_itch.py --days 01302019 03272019 --out-dir itch_raw
    python download_itch.py                         # all 7 default days (~31 GB)
"""
import argparse
import hashlib
import http.client
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
SUFFIX = ".NASDAQ_ITCH50.gz"

# Full-day files seen in the server's directory listing (size in bytes).
# Use --live to refresh this against the server.
CATALOG = {
    "01302019": 4_764_426_091,
    "03272019": 5_510_131_732,
    "07302019": 3_662_140_094,
    "08302019": 4_075_649_457,
    "10302019": 3_872_931_242,
    "12302019": 3_524_013_057,
    "01302020": 5_597_158_940,
}

CHUNK = 1 << 20  # 1 MiB
UA = {"User-Agent": "Mozilla/5.0 (itch-downloader)"}


def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024


def iso(day):
    return f"{day[4:]}-{day[:2]}-{day[2:4]}"


def valid_day(day):
    return bool(re.fullmatch(r"\d{8}", day))


def _open(url, headers=None, method="GET", timeout=60):
    h = dict(UA)
    if headers:
        h.update(headers)
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=h, method=method), timeout=timeout
    )


def remote_size(url, fallback=None):
    """Content-Length via HEAD; falls back to a 1-byte ranged GET, then to `fallback`."""
    try:
        with _open(url, method="HEAD") as r:
            cl = r.headers.get("Content-Length")
            if cl:
                return int(cl)
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        pass
    try:
        with _open(url, headers={"Range": "bytes=0-0"}) as r:
            cr = r.headers.get("Content-Range", "")
            m = re.search(r"/(\d+)$", cr)
            if m:
                return int(m.group(1))
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        pass
    if fallback:
        return fallback
    raise RuntimeError(f"Could not determine size of {url}")


def live_catalog():
    """Best-effort scrape of the server's directory listing -> {MMDDYYYY: size}."""
    with _open(BASE_URL) as r:
        html = r.read().decode("utf-8", "replace")
    days = sorted(set(re.findall(r"(\d{8})\.NASDAQ_ITCH50\.gz(?!\.md5sum)", html)))
    out = {}
    for d in days:
        try:
            size = remote_size(BASE_URL + d + SUFFIX)
        except Exception:
            continue
        if size > 1_000_000_000:  # skip md5-only leftovers
            out[d] = size
    return out


def expected_md5(url):
    try:
        with _open(url + ".md5sum") as r:
            token = r.read().decode("ascii", "replace").split()[0].lower()
        return token if re.fullmatch(r"[0-9a-f]{32}", token) else None
    except Exception:
        return None


def md5_file(path):
    h = hashlib.md5()
    done, total, t0 = 0, os.path.getsize(path), time.time()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK * 8)
            if not b:
                break
            h.update(b)
            done += len(b)
            if time.time() - t0 > 2:
                print(f"\r  verifying md5 ... {done / total:6.1%}", end="", flush=True)
                t0 = time.time()
    print("\r" + " " * 40 + "\r", end="")
    return h.hexdigest()


def _stream(url, part, have, total, label):
    headers = {"Range": f"bytes={have}-"} if have else None
    with _open(url, headers=headers) as r:
        if have and r.status != 206:  # server ignored Range -> restart cleanly
            have = 0
        mode = "ab" if have else "wb"
        t0, last, got0 = time.time(), 0.0, have
        with open(part, mode) as f:
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                now = time.time()
                if now - last > 1.5:
                    rate = (have - got0) / max(now - t0, 1e-9)
                    eta = (total - have) / rate if rate else 0
                    print(
                        f"\r  {label}: {have / total:6.1%}  {fmt_bytes(have)} / "
                        f"{fmt_bytes(total)}  {fmt_bytes(rate)}/s  ETA {eta / 60:5.1f} min",
                        end="",
                        flush=True,
                    )
                    last = now
    print()


def download_one(day, out_dir, retries=8, skip_md5=False):
    name = day + SUFFIX
    url = BASE_URL + name
    dest = os.path.join(out_dir, name)
    part = dest + ".part"
    total = remote_size(url, fallback=CATALOG.get(day))

    if os.path.exists(dest) and os.path.getsize(dest) == total:
        print(f"[{iso(day)}] already present ({fmt_bytes(total)})")
    else:
        for attempt in range(1, retries + 1):
            try:
                have = os.path.getsize(part) if os.path.exists(part) else 0
                if have > total:
                    os.remove(part)
                    have = 0
                if have < total:
                    print(f"[{iso(day)}] downloading (resume at {fmt_bytes(have)})")
                    _stream(url, part, have, total, iso(day))
                if os.path.getsize(part) == total:
                    break
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
                wait = min(60, 2 ** attempt)
                print(f"\n  attempt {attempt}/{retries} failed: {e!r}; retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"{name}: gave up after {retries} attempts (re-run to resume)")
        os.replace(part, dest)

    if skip_md5:
        return dest
    want = expected_md5(url)
    if want is None:
        print(f"[{iso(day)}] no usable .md5sum on server - skipping integrity check")
        return dest
    got = md5_file(dest)
    if got != want:
        bad = dest + ".corrupt"
        os.replace(dest, bad)
        raise RuntimeError(f"{name}: md5 mismatch (expected {want}, got {got}); moved to {bad}")
    print(f"[{iso(day)}] md5 OK ({got})")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="itch_raw", help="where to store the .gz files")
    ap.add_argument("--days", nargs="*", help="MMDDYYYY dates (default: every day in the catalog)")
    ap.add_argument("--list", action="store_true", help="list available days and exit")
    ap.add_argument("--live", action="store_true", help="refresh the catalog from the server")
    ap.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    ap.add_argument("--skip-md5", action="store_true")
    ap.add_argument("--retries", type=int, default=8)
    a = ap.parse_args()

    catalog = dict(CATALOG)
    if a.live:
        try:
            live = live_catalog()
            if live:
                catalog = live
            else:
                print("Live listing returned nothing usable; using built-in catalog.")
        except Exception as e:
            print(f"Live listing failed ({e!r}); using built-in catalog.")

    if a.list:
        for d, s in sorted(catalog.items(), key=lambda kv: kv[0][4:] + kv[0][:4]):
            print(f"{d}  {iso(d)}  {fmt_bytes(s):>10}")
        print(f"total: {fmt_bytes(sum(catalog.values()))}")
        return 0

    days = a.days if a.days else sorted(catalog, key=lambda d: d[4:] + d[:4])
    for d in days:
        if not valid_day(d):
            sys.exit(f"Bad day '{d}': use MMDDYYYY, e.g. 01302019")
        if d not in catalog:
            print(f"Note: {d} is not in the known catalog; will try it anyway.")

    os.makedirs(a.out_dir, exist_ok=True)
    need = 0
    for d in days:
        size = catalog.get(d, 0)
        dest = os.path.join(a.out_dir, d + SUFFIX)
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        part = dest + ".part"
        have = max(have, os.path.getsize(part) if os.path.exists(part) else 0)
        need += max(size - have, 0)
    free = shutil.disk_usage(a.out_dir).free
    print(f"To download: {fmt_bytes(need)}   free space: {fmt_bytes(free)}")
    if a.dry_run:
        for d in days:
            print(f"  would fetch {BASE_URL}{d}{SUFFIX}")
        return 0
    if need > free * 0.95:
        sys.exit("Not enough free disk space. Free some space or pick fewer --days.")

    failures = []
    for d in days:
        try:
            download_one(d, a.out_dir, retries=a.retries, skip_md5=a.skip_md5)
        except Exception as e:
            print(f"[{iso(d)}] FAILED: {e}")
            failures.append(d)
    if failures:
        print("\nFailed days:", " ".join(failures), "(re-run the same command to resume)")
        return 1
    print("\nAll requested files downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
