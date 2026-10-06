from fastapi.testclient import TestClient

from arabic_sentiment.api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Arabic Sentiment Analysis API"
    assert "docs" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "negative" in data["supported_labels"]
    assert "positive" in data["supported_labels"]
    assert "neutral" in data["supported_labels"]


def test_predict_single_positive():
    payload = {"text": "المنتج ممتاز جدا وخاماته عالية الجودة والتوصيل كان سريع"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["label"] == "positive"
    assert data["results"][0]["confidence"] > 0.5
    assert data["latency_ms"] >= 0.0


def test_predict_single_negative():
    payload = {"text": "المنتج سيء جدا ورديء وتالف ولا يعمل نهائيا"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["label"] == "negative"
    assert data["results"][0]["confidence"] > 0.5


def test_predict_batch():
    payload = {
        "texts": ["المنتج ممتاز ورائع", "خامة رديئة جدا وسيئة", "المنتج عادي ومقبول بالنسبة لسعره"]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert data["results"][0]["label"] == "positive"
    assert data["results"][1]["label"] == "negative"
    assert "probabilities" in data["results"][0]


def test_predict_empty_payload():
    response = client.post("/predict", json={})
    assert response.status_code in (400, 422)


def test_predict_whitespace_text():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code in (400, 422)


def test_metrics_endpoint():
    # Make a prediction to ensure counter increments
    client.post("/predict", json={"text": "رائع وممتاز"})
    response = client.get("/metrics")
    assert response.status_code in (200, 501)
    if response.status_code == 200:
        assert "arabic_sentiment_requests_total" in response.text or "# HELP" in response.text
