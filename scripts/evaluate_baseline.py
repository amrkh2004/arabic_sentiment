"""
Evaluation stage for DVC pipeline.
Evaluates test split and outputs standard metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from arabic_sentiment.api.service import SentimentInferenceService
from arabic_sentiment.utils.metrics import compute_classification_metrics


def evaluate_test_set(
    test_path: str = "data/processed/test.csv",
    metrics_path: str = "metrics.json",
    sample_limit: int = 500,
) -> dict:
    test_file = Path(test_path)
    if not test_file.exists():
        raise FileNotFoundError(f"Test split not found at {test_file}")

    df = pd.read_csv(test_file)
    if sample_limit and len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)

    print(f"Evaluating {len(df)} test samples...")
    service = SentimentInferenceService()
    predictions = service.predict(df["clean_text"].tolist())

    y_pred = [p.label for p in predictions]
    y_true = df["label"].tolist()

    # Map labels to integer ids for metrics calculation
    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    y_true_ids = [label_map.get(lbl, 1) for lbl in y_true]
    y_pred_ids = [label_map.get(lbl, 1) for lbl in y_pred]

    metrics = compute_classification_metrics(y_true_ids, y_pred_ids)

    # Save to metrics.json
    out_metrics = {
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "macro_precision": round(metrics["macro_precision"], 4),
        "macro_recall": round(metrics["macro_recall"], 4),
        "test_samples": len(df),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}: {out_metrics}")
    return out_metrics


if __name__ == "__main__":
    evaluate_test_set()
