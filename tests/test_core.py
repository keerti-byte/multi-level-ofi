import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ofi import level_ofi, multi_level_ofi
from data_pipeline import build_day
from windowing import WindowDataset, Scaler, split_days, make_splits, n_valid
from make_synthetic_days import make_book

S = 9_999_999_999


def test_level_ofi_hand_worked():
    # t0 -> t1: bid price up (qty 10 new)        -> +10 ; ask unchanged 5 -> 5 => e_ask=0   => OFI +10
    # t1 -> t2: bid same 10 -> 4 (-6); ask down to new price qty 7 => e_ask=+7 => OFI -6-7 = -13
    # t2 -> t3: bid down (old qty 4 leaves)  -> -4 ; ask up (old qty 7 leaves) e_ask=-7 => OFI -4+7 = +3
    ask_p = np.array([101, 101, 100, 102.]); ask_v = np.array([5, 5, 7, 9.])
    bid_p = np.array([99, 100, 100, 99.]);   bid_v = np.array([3, 10, 4, 8.])
    out = level_ofi(ask_p, ask_v, bid_p, bid_v)
    assert out.tolist() == [0.0, 10.0, -13.0, 3.0]


@pytest.mark.parametrize("ask_sent,bid_sent", [(-S, S), (S, -S)])  # either placeholder convention
def test_empty_level_transitions_are_zero(ask_sent, bid_sent):
    ask_p = np.array([ask_sent, 100.0, ask_sent, ask_sent]); ask_v = np.array([0, 5, 0, 0.])
    bid_p = np.array([bid_sent, 99.0, bid_sent, bid_sent]);  bid_v = np.array([0, 5, 0, 0.])
    out = level_ofi(ask_p, ask_v, bid_p, bid_v)
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0]   # empty->occupied, occupied->empty, empty->empty


def test_multi_level_shape_and_level_independence():
    raw = make_book(500, levels=10, seed=3).astype(float)
    m5 = multi_level_ofi(raw, 5); m1 = multi_level_ofi(raw, 1)
    assert m5.shape == (500, 5)
    assert np.allclose(m5[:, 0], m1[:, 0])


def test_build_day_drops_only_level1_sentinels_and_targets_align():
    raw = make_book(400, seed=1).astype(float)
    raw[10, 0:4] = [-S, 0, S, 0]                     # level-1 empty -> row dropped
    d = build_day(raw, 5, [10, 20])
    assert d["n_dropped"] == 1 and d["n_rows"] == 400 - 1 - 1   # 1 dropped, 1 lost to OFI predecessor
    mid = ((np.delete(raw, 10, axis=0)[:, 0] + np.delete(raw, 10, axis=0)[:, 2]) / 2 / 1e4)[1:]
    assert d["targets"][0, 0] == pytest.approx((mid[10] - mid[0]) / 0.01, abs=1e-4)
    assert np.isnan(d["targets"][-10:, 0]).all() and not np.isnan(d["targets"][-11, 0])


def _seg(M, nf=2, nh=3, base=0):
    f = (np.arange(M)[:, None] + base) * np.ones((1, nf), np.float32)
    t = (np.arange(M)[:, None] + base) * np.ones((1, nh), np.float32)
    return {"features": f.astype(np.float32), "targets": t.astype(np.float32)}


def test_windows_never_cross_segment_boundaries():
    L, H = 5, 4
    segs = [_seg(30, base=0), _seg(20, base=1000), _seg(12, base=5000)]
    ds = WindowDataset(segs, L, H)
    assert len(ds) == sum(n_valid(len(s["features"]), L, H) for s in segs)
    for i in range(len(ds)):
        X, y = ds.batch(np.array([i]))
        vals = X[0, :, 0]
        assert np.all(np.diff(vals) == 1)                       # contiguous rows -> single segment
        assert vals.max() // 1000 == vals.min() // 1000        # same segment block
        assert y[0, 0] == vals[-1]                              # target aligned with window end


def test_target_horizon_never_runs_past_segment_end():
    L, H = 4, 6
    segs = [_seg(25), _seg(25, base=1000)]
    ds = WindowDataset(segs, L, H)
    assert not np.isnan(ds.all_targets()).any()
    last_rows = ds.idx[ds.idx < 25].max()
    assert last_rows + H <= 24


def test_scaler_fit_on_train_only_and_clip():
    tr = [{"features": np.random.default_rng(0).normal(5, 2, (1000, 2)).astype(np.float32),
           "targets": np.random.default_rng(1).normal(0, 3, (1000, 3)).astype(np.float32)}]
    te = [{"features": np.full((10, 2), 1e6, np.float32), "targets": np.zeros((10, 3), np.float32)}]
    sc = Scaler(clip=10).fit(tr)
    assert abs(sc.f_mean[0] - 5) < 0.3 and abs(sc.f_std[0] - 2) < 0.3
    assert sc.apply(te)[0]["features"].max() == 10.0            # no refit on test, clipped


def test_splits():
    assert split_days(10) == ([0, 1, 2, 3, 4, 5, 6], [7], [8, 9])
    assert split_days(5) == ([0, 1, 2], [3], [4])
    assert split_days(7) == ([0, 1, 2, 3, 4], [5], [6])
    days = [_seg(100)]
    tr, va, te, desc = make_splits(days)
    assert len(tr) == len(va) == len(te) == 1 and "row split" in desc
    assert len(tr[0]["features"]) + len(va[0]["features"]) + len(te[0]["features"]) == 100


def test_end_to_end_linear_baseline(tmp_path):
    from make_synthetic_days import make_book as mb
    from train import linear_baseline, r2_score
    days = [build_day(mb(6000, seed=s).astype(float), 5, [10, 20, 50]) for s in range(1, 6)]
    tr, va, te, _ = make_splits(days)
    sc = Scaler().fit(tr)
    dtr, dte = (WindowDataset(sc.apply(x), 50, 50) for x in (tr, te))
    r2 = linear_baseline(dtr, dte, sc.y_std, sc.y_mean)
    assert r2.shape == (3,) and np.isfinite(r2).all()
    assert r2_score(np.array([[1.0], [2.0], [3.0]]), np.array([[1.0], [2.0], [3.0]]))[0] == 1.0


def test_stride_subsamples_windows_but_keeps_validity():
    L, H = 5, 4
    segs = [_seg(60), _seg(40, base=1000)]
    full, st = WindowDataset(segs, L, H), WindowDataset(segs, L, H, stride=7)
    assert np.array_equal(st.idx, full.idx[::7]) and len(st) < len(full)
    X, y = st.batch(np.arange(len(st)))
    assert X.shape == (len(st), L, 2) and np.isfinite(y).all()
