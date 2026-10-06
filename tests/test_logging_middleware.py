import json
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from arabic_sentiment.api.app import app
from arabic_sentiment.api.middleware import JSONLogFormatter

client = TestClient(app)


def test_correlation_id_header_returned():
    """Verify that a Correlation ID is generated and returned if client provides none."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Correlation-ID" in response.headers
    assert len(response.headers["X-Correlation-ID"]) > 0


def test_correlation_id_preserved_when_passed():
    """Verify that client-supplied X-Correlation-ID header is preserved."""
    custom_id = "test-custom-trace-id-12345"
    response = client.get("/health", headers={"X-Correlation-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == custom_id


def test_json_formatter_structure():
    """Verify JSONLogFormatter outputs valid JSON containing all expected fields."""
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Sample structured message",
        args=(),
        exc_info=None,
    )
    record.extra_fields = {"test_metric": 42}

    formatted_str = formatter.format(record)
    parsed = json.loads(formatted_str)

    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Sample structured message"
    assert parsed["test_metric"] == 42
    assert "timestamp" in parsed
    assert "correlation_id" in parsed
    assert parsed["logger"] == "test_logger"


def test_json_formatter_with_exception():
    """Verify JSONLogFormatter properly formats log records with exceptions."""
    formatter = JSONLogFormatter()
    try:
        raise ValueError("Simulated failure for logging")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="error_logger",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )
    formatted_str = formatter.format(record)
    parsed = json.loads(formatted_str)

    assert parsed["level"] == "ERROR"
    assert "exception" in parsed
    assert "Simulated failure for logging" in parsed["exception"]


def test_middleware_unhandled_exception(monkeypatch):
    """Verify middleware logs unhandled exceptions and resets context."""
    import sys

    app_module = sys.modules["arabic_sentiment.api.app"]
    monkeypatch.setattr(
        app_module.inference_service,
        "predict",
        MagicMock(side_effect=RuntimeError("Crash")),
    )

    with pytest.raises(RuntimeError, match="Crash"):
        client.post("/predict", json={"text": "تجربة"})
