"""
High-Throughput Offline Batch Scoring Pipeline for Arabic Sentiment Analysis.
Processes at least 1,000 samples and saves output to /data/scoring/output/.
Computes feature metadata (text_length, confidence_score) for downstream drift detection.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from arabic_sentiment.api.service import SentimentInferenceService


def run_batch_scoring(
    input_path: str = "data/raw/reviews.csv",
    output_dir: str = "data/scoring/output",
    num_samples: int = 1000,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.csv"

    print("=" * 60)
    print("      OFFLINE BATCH SCORING PIPELINE (1,000+ SAMPLES)")
    print("=" * 60)

    df = pd.read_csv(input_path).head(num_samples)
    text_col = "review" if "review" in df.columns else df.columns[0]
    texts = df[text_col].astype(str).tolist()

    print(f"Loaded {len(texts)} samples from {input_path}")
    print("Initializing inference service...")
    service = SentimentInferenceService()

    t0 = time.perf_counter()
    results = service.predict(texts)
    elapsed_s = time.perf_counter() - t0

    df["predicted_label"] = [r.label for r in results]
    df["confidence_score"] = [r.confidence for r in results]
    df["text_length"] = [len(t) for t in texts]

    df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(
        f"Batch Scoring completed in {elapsed_s:.2f} seconds ({len(texts) / elapsed_s:.1f} samples/sec)"
    )
    print(f"Results saved successfully to: {out_file}")
    print("=" * 60)
    return out_file


if __name__ == "__main__":
    run_batch_scoring()
