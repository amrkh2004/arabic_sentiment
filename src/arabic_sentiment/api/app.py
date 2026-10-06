import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from arabic_sentiment import __version__
from arabic_sentiment.api.middleware import CorrelationIdMiddleware, JSONLogFormatter
from arabic_sentiment.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from arabic_sentiment.api.service import SentimentInferenceService


def setup_structured_logging() -> None:
    """Configure root logger to output structured JSON logs."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONLogFormatter())
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]


setup_structured_logging()

# Eagerly initialize inference service
inference_service: SentimentInferenceService | None = SentimentInferenceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global inference_service
    if inference_service is None:
        inference_service = SentimentInferenceService()
    yield


app = FastAPI(
    title="Arabic Sentiment Analysis API",
    description=(
        "Production REST API for classifying Arabic product reviews "
        "(Positive / Negative / Neutral) using Knowledge-Distilled & Quantized INT8 Models."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Correlation ID and Request/Response structured logging middleware
app.add_middleware(CorrelationIdMiddleware)

# Enable CORS for web apps and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health and Readiness Check",
)
def health_check() -> HealthResponse:
    """Returns the service health status, model readiness, and app version."""
    model_loaded = False
    model_type = "ONNX_INT8_OR_FALLBACK"
    if inference_service is not None:
        model_loaded = inference_service.is_loaded
        if model_loaded:
            model_type = "ONNX_INT8"

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_type=model_type,
        app_version=__version__,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Classify Arabic Sentiment",
    status_code=status.HTTP_200_OK,
)
def predict_sentiment(payload: PredictRequest) -> PredictResponse:
    """
    Classifies the sentiment of one or more Arabic product reviews.
    Accepts either a single string (`text`) or a list of strings (`texts`).
    Returns predicted label, calibrated confidence, probabilities, and latency.
    """
    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not initialized.",
        )

    # Resolve target texts
    raw_texts: list[str] = []
    if payload.text is not None:
        raw_texts.append(payload.text)
    elif payload.texts is not None:
        raw_texts.extend(payload.texts)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or 'texts' field must be provided in the request payload.",
        )

    if not raw_texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Input review text cannot be empty."
        )

    t0 = time.perf_counter()
    results = inference_service.predict(raw_texts)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        LATENCY_HISTOGRAM.labels(endpoint="/predict").observe(latency_ms / 1000.0)
        for r in results:
            SENTIMENT_COUNT.labels(sentiment=r.label).inc()

    return PredictResponse(
        results=results,
        latency_ms=round(latency_ms, 2),
        model_version="student-int8-onnx-v1",
    )


try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True

    REQUEST_COUNT = Counter(
        "arabic_sentiment_requests_total",
        "Total HTTP requests handled by the API",
        ["endpoint", "status"],
    )
    LATENCY_HISTOGRAM = Histogram(
        "arabic_sentiment_latency_seconds",
        "Inference latency in seconds",
        ["endpoint"],
    )
    SENTIMENT_COUNT = Counter(
        "arabic_sentiment_predictions_total",
        "Total sentiment predictions classified by category",
        ["sentiment"],
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False


@app.get("/metrics", tags=["Monitoring"], summary="Prometheus Metrics")
def prometheus_metrics():
    """Exposes Prometheus metrics for Grafana scraping and alert manager."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="prometheus_client is not installed in the environment.",
        )
    from fastapi.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["System"], include_in_schema=False)
def root():
    """Root endpoint with quick link to Swagger docs."""
    return {
        "service": "Arabic Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "version": __version__,
    }
