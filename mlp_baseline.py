"""
Small feed-forward network (MLP) baseline on window-summed OFI -- pure NumPy, no deep-learning library.

Purpose: separate "nonlinearity" from "sequence structure". The MLP sees only the OFI summed over the last
`scale` ticks (per level) -- no tick-by-tick order -- but can model nonlinear relationships. If the LSTM beats
it clearly, the temporal pattern inside the window matters; if the MLP matches the LSTM, nonlinearity alone
explains the gain over the linear model.

Same recipe as the LSTM: Adam, MSE, early stopping on validation loss (patience 5), best weights restored.
"""
import copy
import time

import numpy as np


class MLP:
    def __init__(self, n_in, hidden=(64, 32), n_out=3, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)).astype(dtype) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=dtype) for b in sizes[1:]]

    def forward(self, X, cache=False):
        acts, h = [X], X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = z if i == last else np.maximum(z, 0.0)
            acts.append(h)
        return (h, acts) if cache else h

    def backward(self, acts, y):
        """Gradients of mean-squared error (mean over every output element, like torch's MSELoss)."""
        d = 2.0 * (acts[-1] - y) / y.size
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    def state(self):
        return copy.deepcopy((self.W, self.b))

    def load(self, st):
        self.W, self.b = copy.deepcopy(st[0]), copy.deepcopy(st[1])


def fit_predict_mlp(Xtr, ytr, Xva, yva, Xte, hidden=(64, 32), lr=1e-3, batch_size=1024,
                    epochs=30, patience=5, seed=42, verbose=True):
    """Standardise inputs with train statistics, train with Adam, return test predictions."""
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-8)
    Xtr, Xva, Xte = [((x - mu) / sd).astype(np.float32) for x in (Xtr, Xva, Xte)]
    ytr, yva = ytr.astype(np.float32), yva.astype(np.float32)
    net = MLP(Xtr.shape[1], hidden, ytr.shape[1], seed)
    rng = np.random.default_rng(seed)
    m = [np.zeros_like(p) for p in net.W + net.b]
    v = [np.zeros_like(p) for p in net.W + net.b]
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    best, best_state, bad, ran = np.inf, None, 0, 0
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(n)
        run = 0.0
        for i in range(0, n, batch_size):
            sel = order[i:i + batch_size]
            out, acts = net.forward(Xtr[sel], cache=True)
            run += float(((out - ytr[sel]) ** 2).mean()) * len(sel)
            gW, gb = net.backward(acts, ytr[sel])
            step += 1
            params, grads = net.W + net.b, gW + gb
            for k, (p, g) in enumerate(zip(params, grads)):
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                p -= lr * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)
        val = float(((net.forward(Xva) - yva) ** 2).mean())
        ran = ep
        if verbose:
            print(f"    mlp epoch {ep}/{epochs}  loss {run / n:.5f}  val_loss {val:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        if val < best - 1e-9:
            best, best_state, bad = val, net.state(), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    mlp early stopping (no val improvement for {patience} epochs)")
                break
    if best_state is None:
        raise RuntimeError("MLP validation loss never became finite - check the data")
    net.load(best_state)
    return net.forward(Xte), ran
