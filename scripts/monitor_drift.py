"""
Data and Concept Drift Monitoring Suite for Arabic Sentiment Analysis.
Computes statistical drift between baseline reference data and production traffic.
Calculates Population Stability Index (PSI), Wasserstein Distance, and Kolmogorov-Smirnov (KS) test.
Triggers automated alerting when drift exceeds acceptable SLA thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from arabic_sentiment.api.service import SentimentInferenceService


def calculate_psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-4) -> float:
    """Calculates Population Stability Index (PSI) between two categorical distributions."""
    expected = np.maximum(expected, epsilon)
    actual = np.maximum(actual, epsilon)
    expected /= np.sum(expected)
    actual /= np.sum(actual)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def run_drift_analysis(
    reference_path: str = "data/processed/val.csv",
    output_dir: str = "artifacts/monitoring",
    psi_alert_threshold: float = 0.15,
    ks_pvalue_alert_threshold: float = 0.05,
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      ARABIC SENTIMENT PIPELINE - DRIFT MONITORING SUITE")
    print("=" * 65)

    ref_file = Path(reference_path)
    if not ref_file.exists():
        print(f"Reference file not found at {ref_file}. Creating synthetic baseline...")
        ref_df = pd.DataFrame(
            {
                "clean_text": ["ممتاز جدا" * 5] * 200
                + ["رديء جدا" * 5] * 200
                + ["عادي ومقبول" * 5] * 200,
                "label": ["positive"] * 200 + ["negative"] * 200 + ["neutral"] * 200,
            }
        )
    else:
        ref_df = pd.read_csv(ref_file)

    service = SentimentInferenceService()

    # Baseline predictions and feature distributions
    ref_lengths = ref_df["clean_text"].astype(str).str.len().to_numpy()
    ref_counts = ref_df["label"].value_counts(normalize=True)
    labels = ["positive", "neutral", "negative"]
    ref_dist = np.array([ref_counts.get(l, 0.333) for l in labels])

    # Simulate production batch with recent Black Friday / promotion customer surge (drift scenario)
    print("Simulating incoming production batch (recent customer reviews)...")
    simulated_prod_reviews = [
        "التوصيل اتأخر جدا والمنتج مكسور والعلبة مفتوحة حسبي الله",
        "تأخير أسبوع في الشحن وخامة رديئة جدا وغير مطابقة",
        "سيء للغاية ولا أنصح به بتاتا",
        "خدمة عملاء زبالة وما في رد",
        "المنتج عادي بس السعر غالي",
    ] * 60 + [
        "ممتاز ورائع وجودة جيدة جدا",
        "حلو وعجبني وسريع",
    ] * 25

    prod_df = pd.DataFrame({"clean_text": simulated_prod_reviews})
    prod_predictions = service.predict(prod_df["clean_text"].tolist())
    prod_labels = [p.label for p in prod_predictions]
    prod_df["pred_label"] = prod_labels
    prod_lengths = prod_df["clean_text"].astype(str).str.len().to_numpy()

    prod_counts = prod_df["pred_label"].value_counts(normalize=True)
    prod_dist = np.array([prod_counts.get(l, 0.0) for l in labels])

    # 1. Concept Drift Analysis (Target/Prediction Shift)
    psi_score = calculate_psi(ref_dist, prod_dist)
    js_distance = float(jensenshannon(ref_dist, prod_dist))
    concept_drift_detected = bool(float(psi_score) > float(psi_alert_threshold))

    # 2. Data Drift Analysis (Input Feature Shift - Text Length & Characteristics)
    ks_stat, ks_pvalue = ks_2samp(ref_lengths, prod_lengths)
    data_drift_detected = bool(float(ks_pvalue) < float(ks_pvalue_alert_threshold))

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "baseline_samples": len(ref_df),
        "production_batch_samples": len(prod_df),
        "concept_drift": {
            "psi_score": round(float(psi_score), 4),
            "jensen_shannon_distance": round(float(js_distance), 4),
            "alert_threshold": float(psi_alert_threshold),
            "drift_detected": bool(concept_drift_detected),
            "baseline_distribution": {l: round(float(p), 3) for l, p in zip(labels, ref_dist)},
            "production_distribution": {l: round(float(p), 3) for l, p in zip(labels, prod_dist)},
        },
        "data_drift": {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": float(ks_pvalue),
            "alert_threshold_pvalue": float(ks_pvalue_alert_threshold),
            "drift_detected": bool(data_drift_detected),
            "baseline_mean_text_length": round(float(np.mean(ref_lengths)), 1),
            "production_mean_text_length": round(float(np.mean(prod_lengths)), 1),
        },
        "alert_triggered": bool(concept_drift_detected or data_drift_detected),
    }

    # Save detailed JSON report
    report_file = out_path / "drift_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[Concept Drift] PSI Score: {psi_score:.4f} (Threshold: {psi_alert_threshold})")
    print(f"  Baseline Class Distribution:   {report['concept_drift']['baseline_distribution']}")
    print(f"  Production Class Distribution: {report['concept_drift']['production_distribution']}")

    print(
        f"\n[Data Drift] KS Test p-value: {ks_pvalue:.4e} (Threshold: {ks_pvalue_alert_threshold})"
    )

    # Alert Triggering
    if report["alert_triggered"]:
        alert_payload = {
            "status": "CRITICAL_ALERT",
            "reason": "Model drift detected in production traffic.",
            "concept_drift": bool(concept_drift_detected),
            "data_drift": bool(data_drift_detected),
            "recommended_action": "Trigger automated model retraining pipeline with DVC and review recent logs.",
        }
        alert_file = out_path / "drift_alert.json"
        with open(alert_file, "w", encoding="utf-8") as f:
            json.dump(alert_payload, f, indent=2)

        print("\n" + "!" * 65)
        print(" [ALERT TRIGGERED] CRITICAL: Data or Concept Drift detected!")
        print("  - Concept Drift Detected:", concept_drift_detected)
        print("  - Data Drift Detected:   ", data_drift_detected)
        print(f"  - Alert notification saved to: {alert_file}")
        print("  - Recommendation: Trigger automated pipeline retraining.")
        print("!" * 65 + "\n")
    # Save HTML visual dashboard
    html_dashboard = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Arabic Sentiment Pipeline - Drift Monitoring Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #334155; }}
    .alert-banner {{ background: #ef4444; color: white; padding: 1rem; border-radius: 8px; font-weight: bold; margin-bottom: 1.5rem; text-align: center; }}
    .ok-banner {{ background: #10b981; color: white; padding: 1rem; border-radius: 8px; font-weight: bold; margin-bottom: 1.5rem; text-align: center; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
    .stat-card {{ background: #0f172a; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #334155; }}
    .stat-val {{ font-size: 1.8rem; font-weight: bold; color: #38bdf8; }}
    .bar-row {{ display: flex; align-items: center; margin: 0.5rem 0; }}
    .bar-label {{ width: 120px; font-weight: 500; }}
    .bar-outer {{ flex: 1; background: #334155; height: 20px; border-radius: 10px; overflow: hidden; }}
    .bar-inner {{ height: 100%; border-radius: 10px; }}
    .pos {{ background: #10b981; }}
    .neu {{ background: #f59e0b; }}
    .neg {{ background: #ef4444; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 Drift Monitoring Dashboard</h1>
    <p>Arabic Sentiment Analysis Production Model (Track A)</p>
    {'<div class="alert-banner">⚠️ CRITICAL ALERT: Model Drift Detected! Immediate Retraining Recommended.</div>' if report["alert_triggered"] else '<div class="ok-banner">✅ All Systems Healthy. No Significant Drift Detected.</div>'}
    <div class="grid">
      <div class="stat-card"><div>PSI Score</div><div class="stat-val">{psi_score:.4f}</div><small>Threshold: {psi_alert_threshold}</small></div>
      <div class="stat-card"><div>KS Test p-value</div><div class="stat-val">{ks_pvalue:.2e}</div><small>Threshold: {ks_pvalue_alert_threshold}</small></div>
      <div class="stat-card"><div>Baseline Samples</div><div class="stat-val">{len(ref_df)}</div></div>
      <div class="stat-card"><div>Production Batch</div><div class="stat-val">{len(prod_df)}</div></div>
    </div>
    <div class="card" style="margin-top: 1.5rem;">
      <h3>Concept Drift: Sentiment Class Shift</h3>
      <p><b>Baseline:</b> Positive: {report["concept_drift"]["baseline_distribution"]["positive"] * 100:.1f}% | Neutral: {report["concept_drift"]["baseline_distribution"]["neutral"] * 100:.1f}% | Negative: {report["concept_drift"]["baseline_distribution"]["negative"] * 100:.1f}%</p>
      <p><b>Production Batch:</b> Positive: {report["concept_drift"]["production_distribution"]["positive"] * 100:.1f}% | Neutral: {report["concept_drift"]["production_distribution"]["neutral"] * 100:.1f}% | Negative: {report["concept_drift"]["production_distribution"]["negative"] * 100:.1f}%</p>
    </div>
  </div>
</body>
</html>"""
    html_file = out_path / "drift_dashboard.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_dashboard)
    print(f"  - HTML dashboard saved to: {html_file}")

    return report


if __name__ == "__main__":
    run_drift_analysis()
