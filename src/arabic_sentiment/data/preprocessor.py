import re

import pandas as pd


class ArabicTextPreprocessor:
    """
    Object-Oriented Arabic text preprocessor for cleaning and normalizing
    e-commerce customer product reviews.
    """

    def __init__(self, remove_tashkeel: bool = True, normalize_chars: bool = True):
        self.remove_tashkeel = remove_tashkeel
        self.normalize_chars = normalize_chars
        self._tashkeel_regex = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

    def remove_html_and_urls(self, text: str) -> str:
        """Strips HTML tags, hyperlinks, and web handles."""
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#\w+", " ", text)
        return text

    def normalize_arabic_letters(self, text: str) -> str:
        """Unifies letter variations (Alef forms, Taa Marbuta, Yaa)."""
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        return text

    def remove_tatweel(self, text: str) -> str:
        """Removes elongation character and reduces 3+ repeated characters to 1."""
        # Tatweel unicode
        text = re.sub(r"\u0640", "", text)
        # Reduce 3 or more repeated characters to a single character (e.g. ممتااااز -> ممتاز)
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        return text

    def clean(self, text: str | None) -> str:
        """
        Executes complete cleaning pipeline on a raw text string.
        """
        if not text or not isinstance(text, str):
            return ""

        text = self.remove_html_and_urls(text)

        if self.normalize_chars:
            text = self.normalize_arabic_letters(text)

        text = self.remove_tatweel(text)

        if self.remove_tashkeel:
            text = re.sub(self._tashkeel_regex, "", text)

        # Normalize whitespaces and clean special characters
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"[^\u0600-\u06FF\s0-9.,!?،؟]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def transform(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Applies cleaning to a pandas DataFrame and drops empty/duplicate rows.
        """
        df_clean = df.copy()
        df_clean[text_column] = df_clean[text_column].astype(str).apply(self.clean)
        df_clean = df_clean[df_clean[text_column].str.len() >= 5]
        df_clean = df_clean.drop_duplicates(subset=[text_column]).reset_index(drop=True)
        return df_clean
