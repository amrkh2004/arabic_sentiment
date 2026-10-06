import json
import os
import time
from typing import Any

import numpy as np


def measure_inference_latency(
    predict_fn: Any,
    sample_input: Any,
    warmup: int = 15,
    runs: int = 100,
) -> dict[str, float]:
    """
    Measure inference latency across warm-up and evaluation iterations.
    Calculates Mean, Std, P50, P95, P99 (ms) and Throughput.
    """
    # 1. Warm-up iterations
    for _ in range(warmup):
        _ = predict_fn(sample_input)

    # 2. Benchmark iterations
    latencies: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = predict_fn(sample_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    arr = np.array(latencies)
    mean_val = float(np.mean(arr))
    return {
        "mean_ms": round(mean_val, 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "throughput_samples_per_sec": round(1000.0 / mean_val, 2) if mean_val > 0 else 0.0,
    }


def run_benchmarks(output_file: str = "metrics.json") -> dict[str, Any]:
    """Execute latency benchmark suites and persist results into metrics.json."""
    sample_text = "المنتج ممتاز جدا وخامته رائعة وسريع التوصيل"
    results: dict[str, Any] = {}

    print("=" * 60)
    print("Running Arabic Sentiment Latency & Throughput Benchmarks")
    print("=" * 60)

    # 1. API Inference Service Benchmark
    try:
        from arabic_sentiment.api.service import PredictionService

        service = PredictionService()

        # Single review (Batch size = 1)
        single_metrics = measure_inference_latency(service.predict_single, sample_text)
        results["service_single_latency"] = single_metrics
        print(
            f"[*] Service Single Latency (Batch=1): P50 = {single_metrics['p50_ms']} ms | P95 = {single_metrics['p95_ms']} ms"
        )

        # Batch reviews (Batch size = 8)
        batch_input = [sample_text] * 8
        batch_metrics = measure_inference_latency(service.predict_batch, batch_input)
        batch_metrics["throughput_samples_per_sec"] = (
            round(float(8 * 1000.0 / batch_metrics["mean_ms"]), 2)
            if batch_metrics["mean_ms"] > 0
            else 0.0
        )
        results["service_batch8_latency"] = batch_metrics
        print(
            f"[*] Service Batch Latency (Batch=8):  P50 = {batch_metrics['p50_ms']} ms | Throughput = {batch_metrics['throughput_samples_per_sec']} samples/s"
        )

    except (ImportError, RuntimeError, OSError) as e:
        print(f"[!] Service benchmarking skipped: {e}")

    # 2. ONNX Runtime Model-level Benchmark (if model artifact exists)
    onnx_candidates = [
        "artifacts/student_int8.onnx",
        "artifacts/models/student_model.onnx",
    ]
    onnx_path = next((p for p in onnx_candidates if os.path.exists(p)), None)

    if onnx_path:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            sample_ids = np.ones((1, 64), dtype=np.int64)
            sample_mask = np.ones((1, 64), dtype=np.int64)
            onnx_inputs = {"input_ids": sample_ids, "attention_mask": sample_mask}

            onnx_metrics = measure_inference_latency(
                lambda inputs: session.run(None, inputs),
                onnx_inputs,
            )
            results["onnx_int8_model_latency"] = onnx_metrics
            print(
                f"[*] ONNX INT8 Model Latency:        P50 = {onnx_metrics['p50_ms']} ms | P95 = {onnx_metrics['p95_ms']} ms"
            )
        except (ImportError, RuntimeError, OSError) as e:
            print(f"[!] ONNX model benchmark skipped: {e}")

    # 3. Save to metrics.json
    existing_metrics: dict[str, Any] = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_metrics = {}

    existing_metrics.update(results)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Benchmark metrics successfully updated in {output_file}")
    return results


if __name__ == "__main__":
    run_benchmarks()
