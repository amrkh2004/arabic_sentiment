import pandas as pd

from arabic_sentiment.data.preprocessor import ArabicTextPreprocessor


def test_preprocessor_letter_normalization():
    preprocessor = ArabicTextPreprocessor()
    raw = "إعادة تدوير لأحمد ومصطفى"
    cleaned = preprocessor.clean(raw)
    assert "إ" not in cleaned
    assert "ى" not in cleaned
    assert "اعاده تدوير لاحمد ومصطفي" == cleaned


def test_preprocessor_tashkeel_removal():
    preprocessor = ArabicTextPreprocessor()
    raw = "مُنْتَجٌ مُمْتَازٌ جِدًّا"
    cleaned = preprocessor.clean(raw)
    assert cleaned == "منتج ممتاز جدا"


def test_preprocessor_tatweel_and_urls():
    preprocessor = ArabicTextPreprocessor()
    raw = "رووووووعة https://amazon.eg المنتج ممتااااز"
    cleaned = preprocessor.clean(raw)
    assert "https" not in cleaned
    assert "روعه" in cleaned
    assert "ممتاز" in cleaned


def test_preprocessor_dataframe_transform():
    preprocessor = ArabicTextPreprocessor()
    df = pd.DataFrame(
        {
            "text": ["منتج رائع جداً", "سيء للغاية", "https://spam.com", ""],
            "label": ["positive", "negative", "neutral", "neutral"],
        }
    )
    transformed = preprocessor.transform(df)
    assert len(transformed) == 2
    assert "positive" in transformed["label"].values
    assert "negative" in transformed["label"].values
