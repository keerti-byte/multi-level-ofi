"""PyTorch backend: single-layer LSTM, no dropout -> one output per horizon.
Same recipe as the TensorFlow backend (Adam, MSE, early stopping patience 5, best weights restored).
Works on Python 3.14 (TensorFlow has no wheels there)."""
import copy
import time

import numpy as np


def fit_predict(ds_tr, ds_va, ds_te, n_features, n_horizons, units=32, lr=1e-3,
                batch_size=512, epochs=10, patience=5, seed=42, verbose=True):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, units, batch_first=True)   # single layer, no dropout
            self.head = nn.Linear(units, n_horizons)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def batches(ds, shuffle):
        order = np.arange(len(ds))
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            X, y = ds.batch(order[i:i + batch_size])
            yield torch.from_numpy(X), torch.from_numpy(y)

    def predict(ds):
        net.eval()
        with torch.no_grad():
            return np.concatenate([net(X).numpy() for X, _ in batches(ds, False)])

    y_va = ds_va.all_targets()
    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        run, n = 0.0, 0
        for X, y in batches(ds_tr, True):
            opt.zero_grad()
            loss = loss_fn(net(X), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(X)
            n += len(X)
        val = float(((predict(ds_va) - y_va) ** 2).mean())
        history.append({"loss": run / n, "val_loss": val})
        if verbose:
            print(f"  epoch {ep}/{epochs}  loss {run / n:.5f}  val_loss {val:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        if val < best - 1e-9:
            best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stopping (no val improvement for {patience} epochs)")
                break
    if best_state is None:
        raise RuntimeError("validation loss never became finite - check the data (NaN/inf in features or targets)")
    net.load_state_dict(best_state)
    return {"pred": predict(ds_te), "epochs_run": len(history), "best_val_loss": best}
