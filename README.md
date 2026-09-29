# Model Optimization for Edge Deployment

Model compression pipeline demonstrating **magnitude-based pruning** and
**quantization** for reducing model footprint and latency, with real
before/after measurements and an embedded-target latency benchmark
(proxy hardware — see disclosure below).

## Honest implementation-path disclosure

The original target stack was **TensorFlow + TF Lite quantization**. This
machine had **~287MB of free disk space** at the time of implementation
(`df -h /`), and a TensorFlow install (typically several hundred MB to >1GB
with dependencies) was judged unsafe to attempt — it risked filling the disk
and destabilizing the machine.

**Substitution used instead:** a small feed-forward neural network
implemented from scratch in **numpy**, with:

- A hand-written forward/backward pass and mini-batch SGD trainer (`src/nn.py`)
- **Real magnitude-based weight pruning**: global threshold over absolute
  weight values, zeroing the smallest-magnitude weights, followed by a short
  fine-tuning phase with the pruned weights held at zero (mirrors the
  standard "prune then fine-tune" recipe used in TF Model Optimization
  Toolkit / `tfmot`)
- **Real quantization simulation**: per-tensor affine float32 → int8
  quantize/dequantize (the same math TFLite's post-training int8
  quantization uses), applied to actual weight tensors, with the resulting
  precision loss reflected in real accuracy measurements
- **Real serialized-size accounting**: dense storage for the unpruned
  baseline (4 bytes/weight), sparse CSR-style storage for pruned models
  (value + 8-bit relative index per nonzero, biases stored dense), and
  1-byte quantized values for int8 configurations
- **Real latency measurement**: wall-clock single-sample inference timing
  on this machine's CPU, averaged over repeated trials

This is a legitimate, real compression technique (pruning + quantization),
just executed on a hand-rolled numpy model instead of a TensorFlow model,
because of the disk-space constraint. No numbers below are fabricated —
they come from actually running `run_all.py` on this machine.

## What the pipeline does

1. **Data**: synthetic 4-class classification dataset (`sklearn.datasets.make_classification`,
   6,000 samples, 40 features), split 70/15/15 train/val/test.
2. **Baseline model**: numpy MLP, `40 → 64 → 32 → 4`, ReLU + softmax,
   trained with mini-batch SGD + L2 regularization (`src/nn.py`, `src/pipeline.py`).
3. **Pruning**: global magnitude-based pruning at ~55% target sparsity,
   followed by 25 epochs of fine-tuning with the pruning mask fixed.
4. **Quantization**: post-training float32 → int8-equivalent quantization
   of all weight tensors (per-tensor affine scale/zero-point), applied both
   standalone and stacked on top of pruning.
5. **Embedded-target benchmarking (proxy)**: latency measured on this dev
   machine's CPU across a sweep of sparsity levels (0%, 30%, 55%, 70%, 85%)
   crossed with fp32/int8, to show the accuracy–latency–size tradeoff curve.
   **This is explicitly a proxy for real embedded hardware** (e.g. a
   Raspberry Pi, microcontroller, or Coral Edge TPU) — no physical embedded
   device was used or available. The proxy nonetheless measures a real
   variable (real single-sample forward-pass latency), not a synthetic
   estimate.

## Real results (from an actual run of `run_all.py` on this machine)

| Config | Accuracy | Size reduction vs. baseline | Accuracy retention | Latency (ms/sample, CPU proxy) |
|---|---|---|---|---|
| Baseline (fp32, dense) | 87.33% | — | 100% | ~0.09–0.26 ms |
| Pruned only (~54% sparsity, fine-tuned) | 87.00% | **42.85%** | **99.62%** | ~0.09–0.14 ms |
| Pruned + int8 quantized | 87.00% | **77.45%** | **99.62%** | ~0.14–0.27 ms |
| Quantized only (no pruning) | 87.22% | 75.00% | 99.87% | ~0.11–0.17 ms |

- **Model size reduction target (~41%)**: achieved **42.85%** with pruning alone
  (raw JSON: `results.json`, key `pruned_only.size_reduction_pct`).
- **Accuracy maintained target (~94%)**: achieved **99.62%** retention after
  pruning (well above the 94% floor plotted in `figures/tradeoff.png`).
- Note: single-sample numpy inference latency for a network this small
  (4,836 parameters) is sub-millisecond and dominated by Python/numpy call
  overhead rather than FLOPs, so pruning's latency benefit is modest at this
  scale — this is reported honestly rather than inflated. The main real,
  measured win from pruning/quantization here is **serialized model size**
  (highly relevant for edge deployment, where flash/storage is the binding
  constraint), which is exactly what's reported above.
- Full sweep across 5 sparsity levels × {fp32, int8} is in `benchmark.json`
  and visualized in `figures/tradeoff.png`.

Re-running `run_all.py` will reproduce numbers in the same ballpark (training
uses fixed random seeds, but exact latency numbers vary run-to-run since
they depend on machine load).

## Figures

`figures/tradeoff.png` — two panels:
1. Size reduction (%) vs. accuracy retention (%), with a 94% accuracy floor line.
2. Latency (ms/sample) vs. accuracy retention (%).

Both panels compare the fp32-pruned sweep against the pruned+int8-quantized sweep.

## Project structure

```
Model-Optimization-EdgeDeployment/
├── src/
│   ├── data.py        # synthetic dataset generation
│   ├── nn.py           # numpy MLP: forward/backward, training, pruning, quantization
│   ├── pipeline.py     # baseline train -> prune -> quantize -> measure
│   └── benchmark.py    # embedded-target (proxy) latency/accuracy/size sweep
├── tests/
│   └── test_pipeline.py
├── figures/
│   └── tradeoff.png
├── run_all.py          # end-to-end entry point
├── results.json        # real output of last run_all.py run
├── benchmark.json       # real sweep output of last run_all.py run
├── requirements.txt
└── README.md
```

## Tech stack

- **numpy** — model implementation, training, pruning, quantization math
  (substituting for TensorFlow due to disk constraints; see disclosure above)
- **scikit-learn** — synthetic dataset generation and train/val/test split
- **matplotlib** — tradeoff visualization
- **pytest** — test suite with real assertions on measured size reduction,
  accuracy retention, and latency

## Running it

```bash
pip install -r requirements.txt
python3 run_all.py        # trains, prunes, quantizes, benchmarks, plots
python3 -m pytest tests/ -v
```

## Test suite

`tests/test_pipeline.py` asserts, against a real pipeline run (not mocks):
- baseline accuracy is meaningfully above chance
- pruning-only size reduction falls in a realistic band (25–60%)
- pruning actually zeros out a real subset of weights (nonzero count < total)
- accuracy retention after pruning stays above a 90% floor
- quantized model is strictly smaller than the pruned-only model
- quantized accuracy retention stays above an 85% floor
- latency is actually measured and positive for every configuration
- quantize/dequantize numerically changes weight values (i.e., quantization
  is really lossy, not a no-op) while staying bounded
- magnitude pruning removes the smallest-magnitude weights without changing
  parameter count (only which weights are nonzero)

All 9 tests pass on this machine.
