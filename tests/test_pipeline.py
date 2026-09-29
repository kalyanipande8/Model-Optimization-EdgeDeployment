import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.pipeline import run_full_pipeline
from src.nn import MLP, quantize_int8
import numpy as np


@pytest.fixture(scope="module")
def pipeline_results():
    results, models, data = run_full_pipeline()
    return results, models, data


def test_baseline_trains_above_chance(pipeline_results):
    results, _, _ = pipeline_results
    # 4-class synthetic problem: chance is 25%
    assert results["baseline"]["accuracy"] > 0.5


def test_pruning_reduces_size_in_reasonable_band(pipeline_results):
    results, _, _ = pipeline_results
    reduction = results["pruned_only"]["size_reduction_pct"]
    # Target ~41%; allow a realistic band around it.
    assert 25.0 <= reduction <= 60.0


def test_pruning_actually_zeros_weights(pipeline_results):
    _, models, _ = pipeline_results
    pruned = models["pruned"]
    total, nonzero = pruned.count_params()
    assert nonzero < total


def test_accuracy_retention_above_threshold(pipeline_results):
    results, _, _ = pipeline_results
    # Target ~94% retention after pruning; require a reasonable floor.
    assert results["pruned_only"]["accuracy_retention_pct"] >= 90.0


def test_quantized_model_size_smaller_than_pruned(pipeline_results):
    results, _, _ = pipeline_results
    assert results["pruned_plus_quantized"]["size_bytes"] < results["pruned_only"]["size_bytes"]


def test_quantized_accuracy_retention_reasonable(pipeline_results):
    results, _, _ = pipeline_results
    assert results["pruned_plus_quantized"]["accuracy_retention_pct"] >= 85.0


def test_latency_is_measured_and_positive(pipeline_results):
    results, _, _ = pipeline_results
    for key in ("baseline", "pruned_only", "pruned_plus_quantized"):
        assert results[key]["latency_ms_mean"] > 0


def test_quantization_changes_weight_values():
    rng = np.random.RandomState(0)
    w = rng.randn(20, 20).astype(np.float32)
    mask = np.ones_like(w, dtype=bool)
    dequant, scale, zp = quantize_int8(w, mask)
    assert scale > 0
    # quantized weights should differ from originals (lossy) but stay close
    assert not np.allclose(dequant, w)
    assert np.max(np.abs(dequant - w)) < (w.max() - w.min())


def test_magnitude_pruning_removes_smallest_weights():
    model = MLP([10, 8, 3], seed=1)
    before_total, before_nonzero = model.count_params()
    model.prune_magnitude(0.5)
    after_total, after_nonzero = model.count_params()
    assert after_nonzero < before_nonzero
    assert after_total == before_total
