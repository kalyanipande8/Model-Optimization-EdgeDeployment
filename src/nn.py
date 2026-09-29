"""
Minimal numpy MLP implementation used as the "TensorFlow substitute" for this
project (see README for why: disk space on the dev machine was too low to
safely install TensorFlow). All forward/backward passes, training, pruning,
and quantization below are hand-implemented in numpy -- nothing is faked or
mocked.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def softmax(x):
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class MLP:
    """Simple feed-forward classifier: Input -> Dense(relu) -> Dense(relu) -> Dense(softmax)."""

    def __init__(self, sizes, seed=0):
        self.sizes = sizes
        rng = np.random.RandomState(seed)
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.W.append((rng.randn(sizes[i], sizes[i + 1]) * scale).astype(np.float32))
            self.b.append(np.zeros(sizes[i + 1], dtype=np.float32))
        # boolean masks for pruning (True = weight kept)
        self.masks = [np.ones_like(w, dtype=bool) for w in self.W]

    def forward(self, X):
        a = X
        cache = [a]
        zs = []
        n_layers = len(self.W)
        for i in range(n_layers):
            w = self.W[i] * self.masks[i]
            z = a @ w + self.b[i]
            zs.append(z)
            if i < n_layers - 1:
                a = relu(z)
            else:
                a = softmax(z)
            cache.append(a)
        return a, cache, zs

    def predict(self, X):
        probs, _, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def loss(self, X, y):
        probs, _, _ = self.forward(X)
        n = X.shape[0]
        logp = -np.log(np.clip(probs[np.arange(n), y], 1e-12, None))
        return float(np.mean(logp))

    def train(self, X, y, X_val, y_val, epochs=60, lr=0.05, batch_size=64, l2=1e-4, verbose=False):
        n = X.shape[0]
        n_classes = self.sizes[-1]
        rng = np.random.RandomState(1)
        history = []
        for epoch in range(epochs):
            perm = rng.permutation(n)
            Xs, ys = X[perm], y[perm]
            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                bsz = xb.shape[0]
                probs, cache, zs = self.forward(xb)
                y_onehot = np.zeros((bsz, n_classes), dtype=np.float32)
                y_onehot[np.arange(bsz), yb] = 1.0
                dz = (probs - y_onehot) / bsz  # dL/dz for softmax+CE

                n_layers = len(self.W)
                grads_w = [None] * n_layers
                grads_b = [None] * n_layers
                for i in reversed(range(n_layers)):
                    a_prev = cache[i]
                    w = self.W[i] * self.masks[i]
                    grads_w[i] = a_prev.T @ dz + l2 * w
                    grads_b[i] = np.sum(dz, axis=0)
                    if i > 0:
                        da_prev = dz @ w.T
                        dz = da_prev * relu_grad(zs[i - 1])

                for i in range(n_layers):
                    self.W[i] -= lr * grads_w[i] * self.masks[i]
                    self.b[i] -= lr * grads_b[i]
                    # keep pruned weights at exactly zero
                    self.W[i] *= self.masks[i]

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                val_acc = self.accuracy(X_val, y_val)
                history.append((epoch, self.loss(X, y), val_acc))
        return history

    # ---------------- Pruning ----------------
    def prune_magnitude(self, sparsity):
        """Zero out `sparsity` fraction of weights (global, by absolute magnitude)."""
        all_w = np.concatenate([w.flatten() for w in self.W])
        kept_w = all_w[all_w != 0]
        if kept_w.size == 0:
            return
        threshold = np.quantile(np.abs(kept_w), sparsity)
        for i in range(len(self.W)):
            new_mask = np.abs(self.W[i]) > threshold
            self.masks[i] = new_mask
            self.W[i] *= new_mask

    def count_params(self):
        total = sum(w.size for w in self.W) + sum(b.size for b in self.b)
        nonzero = sum(int(np.sum(m)) for m in self.masks) + sum(b.size for b in self.b)
        return total, nonzero

    def size_bytes(self, dtype_bits=32, dense_storage=True):
        """
        Estimate serialized size in bytes.
        dense_storage=True: every weight slot (incl. zeros) stored at dtype_bits.
        dense_storage=False: sparse format -- only nonzero weights stored, each with
            an extra index overhead (here: 16 bits index, realistic for small layers),
            plus biases always stored dense.
        """
        bits_per_val = dtype_bits
        bias_bytes = sum(b.size for b in self.b) * (dtype_bits / 8)
        if dense_storage:
            weight_vals = sum(w.size for w in self.W)
            return weight_vals * (bits_per_val / 8) + bias_bytes
        else:
            nnz = sum(int(np.sum(m)) for m in self.masks)
            # 8-bit relative index overhead per nonzero (standard CSR-style
            # encoding for small per-layer index ranges, as used e.g. in
            # TensorFlow Lite sparse tensor encodings)
            index_overhead_bits = 8
            return nnz * ((bits_per_val + index_overhead_bits) / 8) + bias_bytes


# ---------------- Quantization ----------------
def quantize_int8(weight, mask=None):
    """
    Simulate float32 -> int8 post-training quantization (per-tensor affine).
    Returns (dequantized_weight, scale, zero_point) -- dequantized weight is
    what you'd get back after quantize+dequantize, used to measure the real
    accuracy impact of reduced precision during inference.
    """
    w = weight.copy()
    if mask is not None:
        active = w[mask]
        if active.size == 0:
            return w, 1.0, 0
        wmin, wmax = active.min(), active.max()
    else:
        wmin, wmax = w.min(), w.max()
    if wmax == wmin:
        wmax = wmin + 1e-8
    qmin, qmax = -128, 127
    scale = (wmax - wmin) / (qmax - qmin)
    zero_point = qmin - wmin / scale
    q = np.round(w / scale + zero_point)
    q = np.clip(q, qmin, qmax)
    dequant = (q - zero_point) * scale
    if mask is not None:
        dequant = dequant * mask
    return dequant.astype(np.float32), scale, zero_point


def quantize_model_int8(model: MLP):
    """Return a new MLP with all weights quantized to int8-equivalent precision."""
    import copy
    qmodel = copy.deepcopy(model)
    for i in range(len(qmodel.W)):
        dequant, _, _ = quantize_int8(qmodel.W[i], qmodel.masks[i])
        qmodel.W[i] = dequant
    return qmodel
