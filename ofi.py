"""
Order Flow Imbalance (OFI), event-conditional Cont-Kukanov-Stoikov form, per book level.

LOBSTER orderbook layout (per row, 4 columns per level):
    AskPrice_k, AskSize_k, BidPrice_k, BidSize_k      for k = 1..K

For level k and consecutive snapshots n-1 -> n:
    e_bid = 1[Pb_n >= Pb_{n-1}] * qb_n  -  1[Pb_n <= Pb_{n-1}] * qb_{n-1}
    e_ask = 1[Pa_n <= Pa_{n-1}] * qa_n  -  1[Pa_n >= Pa_{n-1}] * qa_{n-1}
    OFI_k(n) = e_bid - e_ask
Positive OFI = net buying pressure at that level.

Empty levels: LOBSTER fills missing depth with huge placeholder prices (|p| ~ 1e10)
and size 0. Any transition into / out of / within an empty level on a given side
contributes exactly 0 for that side (done explicitly, so it does not depend on the
sign convention of the placeholder).
"""
import numpy as np

SENTINEL_ABS = 999_999_000  # |raw price| >= this means "empty level" (raw = dollars x 10,000)


def is_real(price):
    return np.abs(price) < SENTINEL_ABS


def level_ofi(ask_p, ask_v, bid_p, bid_v):
    """OFI for one level. Inputs are 1-D arrays of equal length N. out[0] = 0."""
    n = len(ask_p)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    ap, ap0, aq, aq0 = ask_p[1:], ask_p[:-1], ask_v[1:], ask_v[:-1]
    bp, bp0, bq, bq0 = bid_p[1:], bid_p[:-1], bid_v[1:], bid_v[:-1]

    e_bid = np.where(bp >= bp0, bq, 0.0) - np.where(bp <= bp0, bq0, 0.0)
    e_ask = np.where(ap <= ap0, aq, 0.0) - np.where(ap >= ap0, aq0, 0.0)

    bid_ok = is_real(bp) & is_real(bp0)
    ask_ok = is_real(ap) & is_real(ap0)
    out[1:] = np.where(bid_ok, e_bid, 0.0) - np.where(ask_ok, e_ask, 0.0)
    return out


def multi_level_ofi(raw, n_levels):
    """raw: (N, >=4*n_levels) LOBSTER array. Returns (N, n_levels) OFI, row 0 = 0."""
    if raw.shape[1] < 4 * n_levels:
        raise ValueError(f"need >= {4 * n_levels} columns for {n_levels} levels, got {raw.shape[1]}")
    cols = []
    for k in range(n_levels):
        c = 4 * k
        cols.append(level_ofi(raw[:, c], raw[:, c + 1], raw[:, c + 2], raw[:, c + 3]))
    return np.stack(cols, axis=1)
