"""
Day-safe windowing without materialising every window.

A sample at row t uses features[t-L+1 .. t] and target[t] (all horizons), and is valid only if
t-L+1 >= 0 and t+max_h <= M-1 *inside its own segment* (a segment = one trading day, or one
row-slice of a day). Segments are concatenated, but valid indices never span two of them.
"""
from typing import List, Sequence

import numpy as np


def n_valid(M: int, lookback: int, max_h: int) -> int:
    return max(0, M - max_h - lookback + 1)


class WindowDataset:
    def __init__(self, segments: List[dict], lookback: int, max_h: int, stride: int = 1):
        self.lookback = lookback
        feats, targs, idx, off = [], [], [], 0
        for s in segments:
            M = len(s["features"])
            feats.append(s["features"])
            targs.append(s["targets"])
            k = n_valid(M, lookback, max_h)
            if k:
                idx.append(off + np.arange(lookback - 1, lookback - 1 + k))
            off += M
        self.F = np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)
        self.Y = np.concatenate(targs) if targs else np.zeros((0, 1), np.float32)
        self.idx = np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)
        if stride > 1:                       # keep every stride-th window (adjacent windows overlap ~fully)
            self.idx = self.idx[::stride]
        self._offs = np.arange(-lookback + 1, 1)

    def __len__(self):
        return len(self.idx)

    def batch(self, sel: np.ndarray):
        """sel: positions into self.idx. Returns X (B, L, F), y (B, H)."""
        t = self.idx[sel]
        X = self.F[t[:, None] + self._offs[None, :]]
        return X, self.Y[t]

    def all_targets(self) -> np.ndarray:
        return self.Y[self.idx]

    def window_sums(self, scale: int = None) -> np.ndarray:
        """Sum of features over the last `scale` ticks (default: the whole lookback) of each sample."""
        L = scale or self.lookback
        if not 1 <= L <= self.lookback:
            raise ValueError(f"scale must be between 1 and the lookback ({self.lookback})")
        if getattr(self, "_cs", None) is None:
            self._cs = np.concatenate([np.zeros((1, self.F.shape[1])), np.cumsum(self.F.astype(np.float64), axis=0)])
        t = self.idx
        return self._cs[t + 1] - self._cs[t + 1 - L]


class Scaler:
    """Per-feature z-score (+ clip) and per-horizon target standardisation, fit on TRAIN only."""

    def __init__(self, clip: float = 10.0):
        self.clip = clip

    def fit(self, train_segments: Sequence[dict]):
        F = np.concatenate([s["features"] for s in train_segments]).astype(np.float64)
        Y = np.concatenate([s["targets"] for s in train_segments]).astype(np.float64)
        self.f_mean, self.f_std = F.mean(0), np.maximum(F.std(0), 1e-8)
        self.y_mean = np.nanmean(Y, axis=0)
        self.y_std = np.maximum(np.nanstd(Y, axis=0), 1e-8)
        return self

    def apply(self, segments: Sequence[dict]) -> List[dict]:
        out = []
        for s in segments:
            f = (s["features"] - self.f_mean) / self.f_std
            if self.clip:
                f = np.clip(f, -self.clip, self.clip)
            y = (s["targets"] - self.y_mean) / self.y_std
            out.append({"features": f.astype(np.float32), "targets": y.astype(np.float32)})
        return out

    def inverse_y(self, y):
        return y * self.y_std + self.y_mean


def split_days(n_days: int):
    """Chronological whole-day split: ~70/10/20 (10 days -> 7/1/2). Needs n_days >= 3."""
    n_test = max(1, int(round(0.2 * n_days)))
    n_val = max(1, int(round(0.1 * n_days)))
    n_train = n_days - n_val - n_test
    if n_train < 1:
        raise ValueError("need at least 3 days for a whole-day split")
    ids = list(range(n_days))
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def split_days_loo(n_days: int, test_day: int):
    """Leave-one-day-out: test = day `test_day` (1-based); validation = the previous day (the next day
    if test_day is 1); train = every other day. test_day = n_days reproduces the default split for 5 days.
    NOTE: for early test days the model trains on LATER days (blocked cross-validation, not walk-forward)."""
    if n_days < 3:
        raise ValueError("leave-one-day-out needs at least 3 days")
    if not 1 <= test_day <= n_days:
        raise ValueError(f"--test-day must be between 1 and {n_days}")
    t = test_day - 1
    v = t - 1 if t > 0 else 1
    return [i for i in range(n_days) if i not in (t, v)], [v], [t]


def split_rows(days: List[dict], fracs=(0.7, 0.1, 0.2)):
    """<3 days: chronological row split of each day into isolated train/val/test segments."""
    tr, va, te = [], [], []
    for d in days:
        M = len(d["features"])
        a, b = int(M * fracs[0]), int(M * (fracs[0] + fracs[1]))
        cut = lambda lo, hi: {"features": d["features"][lo:hi], "targets": d["targets"][lo:hi]}
        tr.append(cut(0, a)); va.append(cut(a, b)); te.append(cut(b, M))
    return tr, va, te


def make_splits(days: List[dict], test_day: int = None):
    if test_day is not None:
        a, b, c = split_days_loo(len(days), test_day)
        pick = lambda ids: [days[i] for i in ids]
        names = lambda ids: ",".join(f"day_{i + 1}" for i in ids)
        return (pick(a), pick(b), pick(c),
                f"leave-one-day-out: test {names(c)} | val {names(b)} | train {names(a)}")
    if len(days) >= 3:
        a, b, c = split_days(len(days))
        pick = lambda ids: [days[i] for i in ids]
        return pick(a), pick(b), pick(c), f"whole days: train {a} / val {b} / test {c}"
    tr, va, te = split_rows(days)
    return tr, va, te, f"{len(days)} day(s) -> chronological row split 70/10/20 inside each day"
