# ==============================================================================
# Multi-Stage Dockerfile for Arabic Sentiment Analysis API
# Stage 1: Build virtual environment and wheels
# Stage 2: Minimal runtime image with security hardening (non-root user)
# ==============================================================================

# ----------------- Stage 1: Builder -----------------
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package metadata and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the application and core runtime dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# ----------------- Stage 2: Runtime -----------------
FROM python:3.11-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PORT=8000 \
    HOST=0.0.0.0

# Install runtime utilities (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root dedicated application user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application code and configurations
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup pyproject.toml ./

# Switch to non-privileged user
USER appuser

# Expose API port
EXPOSE 8000

# Health check to ensure zero-downtime orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "arabic_sentiment.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
