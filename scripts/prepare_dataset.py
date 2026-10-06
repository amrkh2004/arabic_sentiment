"""
Dataset preparation stage for DVC pipeline.
Reads raw reviews, applies Arabic text preprocessing, and generates stratified train/val/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from arabic_sentiment.data.preprocessor import ArabicTextPreprocessor


def prepare_dataset(
    raw_path: str = "data/raw/reviews.csv",
    output_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict:
    raw_file = Path(raw_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")

    print(f"Loading raw data from {raw_file}...")
    df = pd.read_csv(raw_file)

    # Basic validation
    text_col = "text" if "text" in df.columns else "review"
    label_col = "label" if "label" in df.columns else "sentiment"
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()

    # Preprocessing
    print("Preprocessing Arabic texts with ArabicTextPreprocessor...")
    preprocessor = ArabicTextPreprocessor(normalize_chars=True, remove_tashkeel=True)
    df["clean_text"] = df["text"].apply(preprocessor.clean)
    df = df[df["clean_text"].str.len() > 3].reset_index(drop=True)

    # Stratified Split
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=df["label"],
    )

    test_relative_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_relative_ratio,
        random_state=random_state,
        stratify=temp_df["label"],
    )

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    stats = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "class_distribution": df["label"].value_counts().to_dict(),
    }

    stats_path = out_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Dataset successfully prepared and saved to {out_dir}:")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return stats


if __name__ == "__main__":
    prepare_dataset()
