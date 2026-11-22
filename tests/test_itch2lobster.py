import os, shutil, subprocess, sys
import pytest
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import itch_synth as S

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")

@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("bin") / "itch2lobster")
    r = subprocess.run(["g++", "-O2", "-std=c++17", "-o", out, os.path.join(ROOT, "itch2lobster.cpp"), "-lz"], capture_output=True, text=True)
    if r.returncode: pytest.skip("zlib headers missing: " + r.stderr[:200])
    return out

def run(binary, path, out, *extra):
    return subprocess.run([binary, "--input", path, "--output", out, "--symbol", "AAPL", *extra], capture_output=True, text=True)

@pytest.mark.parametrize("gz", [True, False])
def test_matches_python_reference(binary, tmp_path, gz):
    p = str(tmp_path / ("x.gz" if gz else "x.bin")); blob = S.generate(p, n_events=15000, seed=5, gz=gz)
    out = str(tmp_path / "o.csv"); r = run(binary, p, out); assert r.returncode == 0, r.stderr
    got = open(out).read().splitlines()
    exp = S.reference_rows(blob)
    assert len(got) == len(exp) > 1000
    assert got == exp

def test_all_day_and_levels_and_max_messages(binary, tmp_path):
    p = str(tmp_path / "x.gz"); blob = S.generate(p, n_events=6000, seed=9, t0_sec=10000, t1_sec=70000)
    out = str(tmp_path / "o.csv")
    assert run(binary, p, out, "--levels", "3", "--all-day").returncode == 0
    assert open(out).read().splitlines() == S.reference_rows(blob, levels=3, all_day=True)
    assert run(binary, p, out, "--levels", "3").returncode == 0                     # window filter
    assert open(out).read().splitlines() == S.reference_rows(blob, levels=3)
    assert run(binary, p, out, "--all-day", "--max-messages", "500").returncode == 0
    assert open(out).read().splitlines() == S.reference_rows(blob, all_day=True, max_messages=500)

def test_stdin_and_missing_symbol(binary, tmp_path):
    p = str(tmp_path / "x.gz"); S.generate(p, n_events=2000, seed=2)
    out = str(tmp_path / "o.csv")
    r = subprocess.run(f"zcat {p} | {binary} --input - --output {out} --symbol AAPL", shell=True, capture_output=True, text=True)
    assert r.returncode == 0 and os.path.getsize(out) > 0
    r = subprocess.run([binary, "--input", p, "--output", out, "--symbol", "NOPE"], capture_output=True, text=True)
    assert r.returncode == 2 and "not found" in r.stderr

def test_truncated_gz_is_reported(binary, tmp_path):
    p = str(tmp_path / "x.gz"); S.generate(p, n_events=8000, seed=4)
    data = open(p, "rb").read(); open(p, "wb").write(data[: len(data) // 2])
    r = run(binary, p, str(tmp_path / "o.csv"))
    assert r.returncode == 4 and "truncated" in r.stderr
