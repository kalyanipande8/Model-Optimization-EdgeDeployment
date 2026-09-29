"""
Run the full model-optimization pipeline end to end:
  1. Train baseline numpy MLP on synthetic data
  2. Apply real magnitude-based pruning + fine-tune
  3. Apply real int8-equivalent quantization
  4. Run the embedded-target latency benchmark sweep (proxy hardware)
  5. Save results JSON and a tradeoff figure to figures/

Usage: python3 run_all.py
"""
import json
import os

from src.pipeline import run_full_pipeline
from src.benchmark import sweep

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_tradeoff(bench, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp32 = [c for c in bench["configs"] if not c["quantized"]]
    int8 = [c for c in bench["configs"] if c["quantized"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot([c["size_reduction_pct"] for c in fp32], [c["accuracy_retention_pct"] for c in fp32],
            "o-", label="Pruned (fp32)", color="#4C72B0")
    ax.plot([c["size_reduction_pct"] for c in int8], [c["accuracy_retention_pct"] for c in int8],
            "s-", label="Pruned + int8 quantized", color="#DD8452")
    ax.axhline(94, color="gray", linestyle="--", linewidth=1, label="94% accuracy floor")
    ax.set_xlabel("Model size reduction (%)")
    ax.set_ylabel("Accuracy retention vs baseline (%)")
    ax.set_title("Size reduction vs accuracy retention")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot([c["latency_ms_mean"] for c in fp32], [c["accuracy_retention_pct"] for c in fp32],
              "o-", label="Pruned (fp32)", color="#4C72B0")
    ax2.plot([c["latency_ms_mean"] for c in int8], [c["accuracy_retention_pct"] for c in int8],
              "s-", label="Pruned + int8 quantized", color="#DD8452")
    ax2.set_xlabel("Mean inference latency (ms/sample, proxy CPU)")
    ax2.set_ylabel("Accuracy retention vs baseline (%)")
    ax2.set_title("Latency vs accuracy retention")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Saved figure to {out_path}")


def main():
    print("=== Step 1-3: baseline train, prune, quantize ===")
    results, models, data = run_full_pipeline(save_json=os.path.join(RESULTS_DIR, "results.json"))
    print(json.dumps(results, indent=2))

    print("\n=== Step 4: embedded-target (proxy) benchmark sweep ===")
    bench = sweep(save_json=os.path.join(RESULTS_DIR, "benchmark.json"))

    try:
        plot_tradeoff(bench, os.path.join(RESULTS_DIR, "figures", "tradeoff.png"))
    except ImportError:
        print("matplotlib not available; skipping figure generation.")

    print("\n=== Summary ===")
    p = results["pruned_only"]
    pq = results["pruned_plus_quantized"]
    print(f"Baseline accuracy: {results['baseline']['accuracy']:.4f}")
    print(f"Pruned-only size reduction: {p['size_reduction_pct']:.2f}% | "
          f"accuracy retention: {p['accuracy_retention_pct']:.2f}%")
    print(f"Pruned+quantized size reduction: {pq['size_reduction_pct']:.2f}% | "
          f"accuracy retention: {pq['accuracy_retention_pct']:.2f}%")


if __name__ == "__main__":
    main()
