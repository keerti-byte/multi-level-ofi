"""Synthetic ITCH 5.0 stream generator + independent pure-Python reference reconstructor (for testing itch2lobster)."""
import gzip, random, struct

SENT_ASK, SENT_BID = -9999999999, 9999999999


def _hdr(t, loc, ts):
    return t + struct.pack(">HH", loc, 0) + ts.to_bytes(6, "big")


def msg_R(loc, sym, ts=0):
    return _hdr(b"R", loc, ts) + sym.ljust(8).encode() + b"Q" + b"N" + struct.pack(">I", 100) + b"N" + b"C" + b"  " + b"P" + b"N" + b" " + b"1" + b"N" + struct.pack(">I", 0) + b"N"

def msg_A(loc, ts, ref, side, sh, sym, px, mpid=False):
    b = _hdr(b"F" if mpid else b"A", loc, ts) + struct.pack(">QcI", ref, side, sh) + sym.ljust(8).encode() + struct.pack(">I", px)
    return b + (b"NSDQ" if mpid else b"")

def msg_E(loc, ts, ref, sh):  return _hdr(b"E", loc, ts) + struct.pack(">QIQ", ref, sh, 1)
def msg_C(loc, ts, ref, sh, px): return _hdr(b"C", loc, ts) + struct.pack(">QIQcI", ref, sh, 1, b"Y", px)
def msg_X(loc, ts, ref, sh):  return _hdr(b"X", loc, ts) + struct.pack(">QI", ref, sh)
def msg_D(loc, ts, ref):      return _hdr(b"D", loc, ts) + struct.pack(">Q", ref)
def msg_U(loc, ts, old, new, sh, px): return _hdr(b"U", loc, ts) + struct.pack(">QQII", old, new, sh, px)
def msg_P(loc, ts, ref, sh, sym, px): return _hdr(b"P", loc, ts) + struct.pack(">QcI", ref, b"B", sh) + sym.ljust(8).encode() + struct.pack(">II", px, 0) + b"\0" * 4
def msg_S(ts): return _hdr(b"S", 0, ts) + b"O"


def generate(path, n_events=20000, seed=1, target="AAPL", others=("MSFT", "ZZZZ"), t0_sec=33000, t1_sec=58000, gz=True, drift=True):
    """Writes an ITCH stream; returns list of (message_type, payload) needed by the reference."""
    rnd = random.Random(seed)
    syms = {target: 7, **{s: 20 + i for i, s in enumerate(others)}}
    out = [msg_S(0)] + [msg_R(l, s) for s, l in syms.items()]
    live = {s: {} for s in syms}          # sym -> {ref: (side, px, shares)}
    ref = 1000
    mid = {s: 1_500_000 + 10000 * i for i, s in enumerate(syms)}
    for i in range(n_events):
        ts = int((t0_sec + (t1_sec - t0_sec) * i / n_events) * 1e9)
        s = target if rnd.random() < 0.6 else rnd.choice(others)
        loc, book = syms[s], live[s]
        r = rnd.random()
        if r < 0.45 or not book:
            ref += 1
            side = b"B" if rnd.random() < 0.5 else b"S"
            off = 100 * rnd.randint(1, 12)
            px = mid[s] - off if side == b"B" else mid[s] + off
            sh = rnd.randint(1, 500)
            out.append(msg_A(loc, ts, ref, side, sh, s, px, mpid=rnd.random() < 0.2))
            book[ref] = [side, px, sh]
        else:
            k = rnd.choice(list(book))
            side, px, sh = book[k]
            if r < 0.60:
                n = rnd.randint(1, sh); out.append(msg_E(loc, ts, k, n)); sh -= n
            elif r < 0.68:
                n = rnd.randint(1, sh); out.append(msg_C(loc, ts, k, n, px)); sh -= n
            elif r < 0.78:
                n = rnd.randint(1, sh); out.append(msg_X(loc, ts, k, n)); sh -= n
            elif r < 0.90:
                out.append(msg_D(loc, ts, k)); sh = 0
            elif r < 0.97:
                ref += 1; o2 = 100 * rnd.randint(1, 12); npx = mid[s] - o2 if side == b"B" else mid[s] + o2; nsh = rnd.randint(1, 500)
                out.append(msg_U(loc, ts, k, ref, nsh, npx)); del book[k]; book[ref] = [side, npx, nsh]; continue
            else:
                out.append(msg_P(loc, ts, 5, 10, s, px)); continue
            if sh == 0: del book[k]
            else: book[k][2] = sh
        if drift and rnd.random() < 0.05: mid[s] += 100 * rnd.choice([-1, 1])
    blob = b"".join(struct.pack(">H", len(m)) + m for m in out)
    with (gzip.open if gz else open)(path, "wb") as fh:
        fh.write(blob)
    return blob


def reference_rows(blob, symbol="AAPL", levels=10, window=(34200e9, 57600e9), all_day=False, max_messages=0):
    """Independent, deliberately simple reconstruction straight from the spec."""
    pos, n, loc, orders, rows = 0, 0, None, {}, []
    while pos + 2 <= len(blob):
        ln = struct.unpack(">H", blob[pos:pos + 2])[0]; m = blob[pos + 2: pos + 2 + ln]; pos += 2 + ln; n += 1
        if max_messages and n > max_messages: break
        t = chr(m[0])
        if t == "R":
            if m[11:19].decode().strip() == symbol: loc = struct.unpack(">H", m[1:3])[0]
            continue
        if loc is None or struct.unpack(">H", m[1:3])[0] != loc: continue
        ts = int.from_bytes(m[5:11], "big"); changed = False
        if t in "AF":
            r, side, sh = struct.unpack(">QcI", m[11:24]); px = struct.unpack(">I", m[32:36])[0]
            orders[r] = [side, px, sh]; changed = True
        elif t in "ECX":
            r, sh = struct.unpack(">QI", m[11:23])
            if r in orders: orders[r][2] -= min(sh, orders[r][2]); changed = True
        elif t == "D":
            r = struct.unpack(">Q", m[11:19])[0]
            if r in orders: orders[r][2] = 0; changed = True
        elif t == "U":
            o, nw, sh, px = struct.unpack(">QQII", m[11:35])
            if o in orders:
                side = orders[o][0]; orders[o][2] = 0; orders[nw] = [side, px, sh]; changed = True
        if not changed: continue
        orders = {k: v for k, v in orders.items() if v[2] > 0}
        if all_day or window[0] <= ts <= window[1]:
            bids, asks = {}, {}
            for side, px, sh in orders.values():
                d = bids if side == b"B" else asks
                d[px] = d.get(px, 0) + sh
            row = []
            a = sorted(asks.items()); b = sorted(bids.items(), reverse=True)
            for l in range(levels):
                row += list(a[l]) if l < len(a) else [SENT_ASK, 0]
                row += list(b[l]) if l < len(b) else [SENT_BID, 0]
            rows.append(",".join(map(str, row)))
    return rows
