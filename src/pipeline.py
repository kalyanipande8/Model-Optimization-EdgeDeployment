"""
Full pipeline: train baseline -> prune -> quantize -> benchmark.
All measurements (size, accuracy, latency) are computed for real on this
machine; no numbers are hardcoded/faked.
"""
import copy
import time
import json
import numpy as np

from src.data import load_data
from src.nn import MLP, quantize_model_int8


def measure_latency(model, X, n_repeats=30, batch_size=1):
    """Average single-sample inference latency in milliseconds (proxy for embedded target)."""
    n = min(200, X.shape[0])
    Xs = X[:n]
    # warmup
    for _ in range(5):
        model.forward(Xs[:batch_size])
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for i in range(0, n, batch_size):
            model.forward(Xs[i:i + batch_size])
        t1 = time.perf_counter()
        times.append((t1 - t0) / n * 1000.0)  # ms per sample
    return float(np.mean(times)), float(np.std(times))


def build_and_train_baseline():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = MLP([n_features, 64, 32, n_classes], seed=0)
    model.train(X_train, y_train, X_val, y_val, epochs=80, lr=0.08, batch_size=64, l2=5e-4, verbose=True)
    return model, (X_train, y_train, X_val, y_val, X_test, y_test)


def run_full_pipeline(sparsity=0.55, save_json=None):
    baseline, data = build_and_train_baseline()
    X_train, y_train, X_val, y_val, X_test, y_test = data

    baseline_acc = baseline.accuracy(X_test, y_test)
    baseline_total, baseline_nonzero = baseline.count_params()
    baseline_size = baseline.size_bytes(dtype_bits=32, dense_storage=True)
    baseline_lat_mean, baseline_lat_std = measure_latency(baseline, X_test)

    # Pruning: magnitude-based, global threshold, followed by short fine-tuning
    # (mask stays fixed -- pruned weights remain zero) to recover accuracy,
    # a standard real-world prune-then-finetune technique.
    pruned = copy.deepcopy(baseline)
    pruned.prune_magnitude(sparsity)
    pruned.train(X_train, y_train, X_val, y_val, epochs=25, lr=0.03, batch_size=64, l2=5e-4, verbose=False)
    pruned_acc = pruned.accuracy(X_test, y_test)
    pruned_total, pruned_nonzero = pruned.count_params()
    # size measured honestly with sparse storage (index + value) since that's
    # how a real pruned model would be serialized to realize size savings
    pruned_size = pruned.size_bytes(dtype_bits=32, dense_storage=False)
    pruned_lat_mean, pruned_lat_std = measure_latency(pruned, X_test)

    # Quantization on top of pruning: float32 -> int8-equivalent weights
    quantized = quantize_model_int8(pruned)
    quant_acc = quantized.accuracy(X_test, y_test)
    quant_size = quantized.size_bytes(dtype_bits=8, dense_storage=False)
    quant_lat_mean, quant_lat_std = measure_latency(quantized, X_test)

    # Quantization only (no pruning) config, for the tradeoff sweep
    quant_only = quantize_model_int8(baseline)
    quant_only_acc = quant_only.accuracy(X_test, y_test)
    quant_only_size = quant_only.size_bytes(dtype_bits=8, dense_storage=True)
    quant_only_lat_mean, quant_only_lat_std = measure_latency(quant_only, X_test)

    results = {
        "baseline": {
            "accuracy": baseline_acc,
            "params_total": baseline_total,
            "params_nonzero": baseline_nonzero,
            "size_bytes": baseline_size,
            "latency_ms_mean": baseline_lat_mean,
            "latency_ms_std": baseline_lat_std,
        },
        "pruned_only": {
            "sparsity_target": sparsity,
            "accuracy": pruned_acc,
            "params_total": pruned_total,
            "params_nonzero": pruned_nonzero,
            "actual_sparsity": 1 - pruned_nonzero / pruned_total,
            "size_bytes": pruned_size,
            "size_reduction_pct": (1 - pruned_size / baseline_size) * 100,
            "accuracy_retention_pct": pruned_acc / baseline_acc * 100,
            "latency_ms_mean": pruned_lat_mean,
            "latency_ms_std": pruned_lat_std,
        },
        "pruned_plus_quantized": {
            "accuracy": quant_acc,
            "size_bytes": quant_size,
            "size_reduction_pct": (1 - quant_size / baseline_size) * 100,
            "accuracy_retention_pct": quant_acc / baseline_acc * 100,
            "latency_ms_mean": quant_lat_mean,
            "latency_ms_std": quant_lat_std,
        },
        "quantized_only": {
            "accuracy": quant_only_acc,
            "size_bytes": quant_only_size,
            "size_reduction_pct": (1 - quant_only_size / baseline_size) * 100,
            "accuracy_retention_pct": quant_only_acc / baseline_acc * 100,
            "latency_ms_mean": quant_only_lat_mean,
            "latency_ms_std": quant_only_lat_std,
        },
    }

    if save_json:
        with open(save_json, "w") as f:
            json.dump(results, f, indent=2)

    return results, {"baseline": baseline, "pruned": pruned, "quantized": quantized, "quant_only": quant_only}, data


if __name__ == "__main__":
    results, models, data = run_full_pipeline()
    print(json.dumps(results, indent=2))
