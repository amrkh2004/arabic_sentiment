import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from arabic_sentiment.core.config import DataConfig


class SentimentDataset(Dataset):
    """PyTorch Dataset for tokenized Arabic sentiment reviews."""

    def __init__(
        self, input_ids: list[list[int]], attention_masks: list[list[int]], labels: list[int]
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SentimentDataModule:
    """
    DataModule handling dataset loading, label encoding, stratified splitting,
    tokenization, class weights calculation, and DataLoader creation.
    """

    def __init__(self, config: DataConfig, tokenizer_name: str, seed: int = 42):
        self.config = config
        self.tokenizer_name = tokenizer_name
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label2id = {name: i for i, name in enumerate(config.labels)}
        self.id2label = {i: name for name, i in self.label2id.items()}

        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.class_weights: torch.Tensor | None = None

    def prepare_data(
        self, df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads data, maps labels, and performs stratified 80/10/10 split."""
        if df is None:
            df = pd.read_csv(self.config.raw_path)

        df = df[[self.config.text_column, self.config.label_column]].dropna()
        df = df.rename(
            columns={self.config.text_column: "text", self.config.label_column: "label_name"}
        )
        df["label"] = df["label_name"].map(self.label2id)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        # Stratified 80 / 10 / 10 split
        test_val_size = self.config.val_ratio + self.config.test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=test_val_size,
            stratify=df["label"],
            random_state=self.seed,
        )
        val_relative_size = self.config.val_ratio / test_val_size
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_relative_size),
            stratify=temp_df["label"],
            random_state=self.seed,
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        if self.config.use_class_weights:
            weights = compute_class_weight(
                "balanced",
                classes=np.arange(len(self.config.labels)),
                y=self.train_df["label"].values,
            )
            self.class_weights = torch.tensor(weights, dtype=torch.float32)

        return self.train_df, self.val_df, self.test_df

    def _create_dataset(self, frame: pd.DataFrame) -> SentimentDataset:
        enc = self.tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=self.config.max_length,
        )
        return SentimentDataset(enc["input_ids"], enc["attention_mask"], frame["label"].tolist())

    def get_dataloaders(self, batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Constructs PyTorch DataLoaders with dynamic padding collator."""
        if self.train_df is None:
            self.prepare_data()

        train_ds = self._create_dataset(self.train_df)
        val_ds = self._create_dataset(self.val_df)
        test_ds = self._create_dataset(self.test_df)

        collator = DataCollatorWithPadding(self.tokenizer)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collator
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collator
        )

        return train_loader, val_loader, test_loader
