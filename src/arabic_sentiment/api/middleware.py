import json
import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Context Variable for storing correlation_id per request safely across async tasks
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Retrieve current correlation ID from async context."""
    return correlation_id_ctx.get()


class JSONLogFormatter(logging.Formatter):
    """Format logs as a structured JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "correlation_id": get_correlation_id() or getattr(record, "correlation_id", None),
        }

        # Include extra fields if provided
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            log_record.update(extra_fields)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware for managing Correlation ID and structured Request/Response logging."""

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        corr_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        token = correlation_id_ctx.set(corr_id)

        start_time = time.perf_counter()
        logger = logging.getLogger("arabic_sentiment.api")

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            response.headers["X-Correlation-ID"] = corr_id

            log_data = {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
            logger.info(
                f"{request.method} {request.url.path} completed with {response.status_code}",
                extra={"extra_fields": log_data},
            )
            return response

        except Exception:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            }
            logger.exception(
                f"Unhandled error processing {request.method} {request.url.path}",
                extra={"extra_fields": log_data},
            )
            raise
        finally:
            correlation_id_ctx.reset(token)
