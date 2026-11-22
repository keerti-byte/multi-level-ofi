"""Single-layer LSTM regressor (no dropout) -> one output per horizon. TensorFlow is imported lazily."""
import numpy as np


def build_model(lookback, n_features, n_horizons, units=32, lr=1e-3):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, n_features)),
        tf.keras.layers.LSTM(units),                 # single layer, no dropout
        tf.keras.layers.Dense(n_horizons),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return m


def make_sequence(ds, batch_size, shuffle, seed=0):
    """Keras Sequence that builds each batch of windows on the fly from a WindowDataset."""
    import tensorflow as tf

    class WindowSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.order = np.arange(len(ds))
            self.rng = np.random.default_rng(seed)
            if shuffle:
                self.rng.shuffle(self.order)

        def __len__(self):
            return int(np.ceil(len(ds) / batch_size))

        def __getitem__(self, i):
            sel = self.order[i * batch_size:(i + 1) * batch_size]
            return ds.batch(sel)

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.order)

    return WindowSequence()
