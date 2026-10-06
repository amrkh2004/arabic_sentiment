from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import torch
except ImportError:
    torch = None


def compute_classification_metrics(
    y_true: np.ndarray | list, y_pred: np.ndarray | list
) -> dict[str, float]:
    """Calculates accuracy, Macro-F1, Precision, and Recall."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def get_file_size_mb(file_path: str) -> float:
    """Returns file size in megabytes."""
    if not os.path.exists(file_path):
        return 0.0
    return os.path.getsize(file_path) / 1e6


def count_model_parameters_m(model: Any) -> float:
    """Returns parameter count in millions."""
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters()) / 1e6
    return 0.0


def measure_latency_ms(
    predict_fn: Callable[[Any, Any], Any],
    input_ids: Any,
    attention_mask: Any,
    warmup: int = 10,
    iters: int = 50,
    sync_cuda: bool = False,
) -> tuple[float, float]:
    """
    Measures p50 (median) and p95 latency in milliseconds.
    """
    for _ in range(warmup):
        predict_fn(input_ids, attention_mask)

    times = []
    for _ in range(iters):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        predict_fn(input_ids, attention_mask)

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    return p50, p95
