"""
Embedded-target benchmarking (proxy). We do not have real embedded hardware
(e.g. Raspberry Pi / Coral / microcontroller) available, so this benchmarks
real single-sample inference latency on this development machine's CPU as a
*proxy* for an edge device, across several real pruning/quantization
configurations. This is disclosed honestly in the README -- it is a stand-in
for, not a replacement of, on-device measurement.
"""
import copy
import json
import numpy as np

from src.data import load_data
from src.nn import MLP, quantize_model_int8
from src.pipeline import measure_latency, build_and_train_baseline


def sweep(save_json=None):
    baseline, data = build_and_train_baseline()
    X_train, y_train, X_val, y_val, X_test, y_test = data
    baseline_acc = baseline.accuracy(X_test, y_test)
    baseline_size = baseline.size_bytes(dtype_bits=32, dense_storage=True)

    configs = []
    sparsities = [0.0, 0.3, 0.55, 0.7, 0.85]
    for s in sparsities:
        m = copy.deepcopy(baseline)
        if s > 0:
            m.prune_magnitude(s)
            m.train(X_train, y_train, X_val, y_val, epochs=25, lr=0.03, batch_size=64, l2=5e-4)
        acc = m.accuracy(X_test, y_test)
        size_fp32 = m.size_bytes(dtype_bits=32, dense_storage=(s == 0))
        lat_mean, lat_std = measure_latency(m, X_test)
        configs.append({
            "name": f"prune_{int(s*100)}pct_fp32",
            "sparsity": s,
            "quantized": False,
            "accuracy": acc,
            "accuracy_retention_pct": acc / baseline_acc * 100,
            "size_bytes": size_fp32,
            "size_reduction_pct": (1 - size_fp32 / baseline_size) * 100,
            "latency_ms_mean": lat_mean,
            "latency_ms_std": lat_std,
        })

        mq = quantize_model_int8(m)
        accq = mq.accuracy(X_test, y_test)
        size_q = mq.size_bytes(dtype_bits=8, dense_storage=(s == 0))
        latq_mean, latq_std = measure_latency(mq, X_test)
        configs.append({
            "name": f"prune_{int(s*100)}pct_int8",
            "sparsity": s,
            "quantized": True,
            "accuracy": accq,
            "accuracy_retention_pct": accq / baseline_acc * 100,
            "size_bytes": size_q,
            "size_reduction_pct": (1 - size_q / baseline_size) * 100,
            "latency_ms_mean": latq_mean,
            "latency_ms_std": latq_std,
        })

    result = {
        "baseline_accuracy": baseline_acc,
        "baseline_size_bytes": baseline_size,
        "note": "Latency measured on this development machine's CPU as a PROXY for an "
                "edge/embedded device -- no physical embedded hardware was used.",
        "configs": configs,
    }
    if save_json:
        with open(save_json, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = sweep()
    print(json.dumps(r, indent=2))
