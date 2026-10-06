"""
Comprehensive tests to boost test coverage across utils, models, CLI, and inference services.
"""

import importlib.util
import sys
from unittest import mock

import numpy as np
import pytest

from arabic_sentiment.api.service import SentimentInferenceService
from arabic_sentiment.utils.metrics import (
    compute_classification_metrics,
    count_model_parameters_m,
    get_file_size_mb,
    measure_latency_ms,
)


def test_compute_classification_metrics():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert metrics["accuracy"] > 0.0


def test_file_size_and_param_count(tmp_path):
    temp_file = tmp_path / "dummy.bin"
    temp_file.write_bytes(b"a" * 1_000_000)

    size = get_file_size_mb(str(temp_file))
    assert 0.95 <= size <= 1.05

    non_existent = get_file_size_mb(str(tmp_path / "absent.bin"))
    assert non_existent == 0.0

    class DummyModel:
        pass

    assert count_model_parameters_m(DummyModel()) == 0.0

    class ModelWithParams:
        def parameters(self):
            class Param:
                def numel(self):
                    return 1_000_000

            return [Param(), Param()]

    assert count_model_parameters_m(ModelWithParams()) == 2.0


def test_config_non_existent_fallback():
    from arabic_sentiment.core.config import AppConfig

    cfg = AppConfig.from_yaml("non_existent_config.yaml")
    assert cfg.seed == 42


def test_schema_batch_validation_errors():
    from pydantic import ValidationError

    from arabic_sentiment.api.schemas import PredictRequest

    with pytest.raises(ValidationError):
        PredictRequest(texts=[])

    with pytest.raises(ValidationError):
        PredictRequest(texts=["valid", "   "])


def test_measure_latency_ms():
    def mock_predict(x, y):
        return x + y

    p50, p95 = measure_latency_ms(mock_predict, 1, 2, warmup=2, iters=5)
    assert p50 >= 0.0
    assert p95 >= p50


def test_compute_distillation_loss():
    try:
        import torch

        from arabic_sentiment.models.student import compute_distillation_loss

        student_logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 1.5, -0.8]])
        teacher_logits = torch.tensor([[1.8, -0.9, 0.4], [0.0, 1.4, -0.6]])
        labels = torch.tensor([0, 1])

        loss = compute_distillation_loss(
            student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5
        )
        assert loss.item() > 0.0
    except ImportError:
        pytest.skip("PyTorch not installed")


def test_cli_help(capsys):
    from arabic_sentiment.cli import main

    with (
        mock.patch.object(sys, "argv", ["arabic-sentiment", "--help"]),
        pytest.raises(SystemExit),
    ):
        main()


def test_cli_command_quantize(capsys):
    from arabic_sentiment.cli import main

    with mock.patch.object(
        sys, "argv", ["arabic-sentiment", "--config", "configs/config.yaml", "quantize"]
    ):
        main()
    captured = capsys.readouterr()
    assert "INT8" in captured.out or "Exporting" in captured.out


def test_service_batch_prediction_edge_cases():
    service = SentimentInferenceService()
    # Test batch prediction
    results = service.predict(["ممتاز جدا", "سيء للغاية"])
    assert len(results) == 2
    assert results[0].label in ["positive", "negative", "neutral"]
    assert results[1].label in ["positive", "negative", "neutral"]

    # Test empty payload
    empty_res = service.predict([])
    assert len(empty_res) == 0

    # Test whitespace fallback
    single_res = service.predict(["   "])
    assert len(single_res) == 1
    assert single_res[0].label in ["positive", "negative", "neutral"]


def test_bento_service():
    if importlib.util.find_spec("bentoml") is None:
        pytest.skip("BentoML not installed")

    from arabic_sentiment.serving.service import ArabicSentimentBentoService

    bento_svc = ArabicSentimentBentoService()
    res = bento_svc.predict("المنتج رائع جدا")
    assert res["label"] in ["positive", "negative", "neutral"]

    batch_res = bento_svc.predict_batch(["منتج ممتاز", "منتج سيء"])
    assert len(batch_res) == 2


def test_sentiment_dataset():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not installed")

    from arabic_sentiment.data.dataset import SentimentDataset

    ds = SentimentDataset([[1, 2, 3]], [[1, 1, 1]], [0])
    assert len(ds) == 1
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item


def test_sentiment_service_with_mock_onnx():
    service = SentimentInferenceService()
    mock_session = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()

    mock_tokenizer.return_value = {
        "input_ids": np.array([[101, 102]], dtype=np.int64),
        "attention_mask": np.array([[1, 1]], dtype=np.int64),
    }
    # Index 2 is max -> positive
    mock_session.run.return_value = [np.array([[-1.0, 0.2, 2.5]])]

    service.is_loaded = True
    service.session = mock_session
    service.tokenizer = mock_tokenizer

    res = service.predict(["منتج رائع جدا"])
    assert len(res) == 1
    assert res[0].label == "positive"
    assert res[0].confidence > 0.5


def test_api_additional_branches(monkeypatch):
    import sys

    from fastapi.testclient import TestClient

    api_mod = sys.modules["arabic_sentiment.api.app"]
    client = TestClient(api_mod.app)

    # 1. Health when loaded
    if api_mod.inference_service is not None:
        monkeypatch.setattr(api_mod.inference_service, "is_loaded", True)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_type"] == "ONNX_INT8"

    # 2. Predict when service is None
    monkeypatch.setattr(api_mod, "inference_service", None)
    resp_err = client.post("/predict", json={"text": "تجربة"})
    assert resp_err.status_code == 503

    # 3. Metrics when Prometheus is not available
    monkeypatch.setattr(api_mod, "PROMETHEUS_AVAILABLE", False)
    resp_501 = client.get("/metrics")
    assert resp_501.status_code == 501


def test_cli_subcommands_coverage():
    from arabic_sentiment.cli import main

    for cmd in ["train-teacher", "distill", "benchmark"]:
        with mock.patch.object(
            sys, "argv", ["arabic-sentiment", "--config", "configs/config.yaml", cmd]
        ):
            main()

    with (
        mock.patch.object(sys, "argv", ["arabic-sentiment"]),
        pytest.raises(SystemExit),
    ):
        main()
