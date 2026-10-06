import json
import os
from unittest.mock import MagicMock

from scripts.benchmark_latency import measure_inference_latency, run_benchmarks


def test_measure_inference_latency():
    """Verify latency measuring function computes statistics properly."""
    dummy_predict = MagicMock(return_value="dummy_result")
    metrics = measure_inference_latency(dummy_predict, "sample", warmup=5, runs=20)

    assert "mean_ms" in metrics
    assert "std_ms" in metrics
    assert "p50_ms" in metrics
    assert "p95_ms" in metrics
    assert "p99_ms" in metrics
    assert "throughput_samples_per_sec" in metrics
    assert metrics["mean_ms"] >= 0.0
    assert dummy_predict.call_count == 25


def test_run_benchmarks_custom_file(tmp_path):
    """Verify run_benchmarks writes valid JSON to custom output path."""
    out_file = str(tmp_path / "test_metrics.json")
    results = run_benchmarks(output_file=out_file)

    assert os.path.exists(out_file)
    assert "service_single_latency" in results
    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)

    assert "service_single_latency" in data
    assert "service_batch8_latency" in data
    assert data["service_single_latency"]["p50_ms"] >= 0.0
