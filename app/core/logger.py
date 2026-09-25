# app/core/logger.py
import logging
import os
import sys
import time
import json
import uuid
import functools
import traceback
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Optional
from pathlib import Path

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create formatters
formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handlers with rotation
def _create_file_handler(filename: str, level: int = logging.DEBUG) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler

# General application log
app_handler = _create_file_handler("app.log", logging.INFO)
# Error log
error_handler = _create_file_handler("errors.log", logging.ERROR)
# Pipeline-specific logs
pipeline_handler = _create_file_handler("pipeline.log", logging.DEBUG)
# IRI computation log
iri_handler = _create_file_handler("iri.log", logging.DEBUG)
# Vision/YOLO log
vision_handler = _create_file_handler("vision.log", logging.DEBUG)
# Spatial/centerline log
spatial_handler = _create_file_handler("spatial.log", logging.DEBUG)
# Fusion/DV log
fusion_handler = _create_file_handler("fusion.log", logging.DEBUG)


class PipelineLogger:
    """Structured logger for pipeline stages with run context."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(f"pipeline.{name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(pipeline_handler)
        self.logger.propagate = True

    def _log(self, level: int, event: str, run_id: str = "", **kwargs: Any) -> None:
        extra: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": time.time(),
            "event": event,
            **kwargs,
        }
        self.logger.log(level, json.dumps(extra, default=str))

    def info(self, event: str, run_id: str = "", **kwargs: Any) -> None:
        self._log(logging.INFO, event, run_id, **kwargs)

    def warning(self, event: str, run_id: str = "", **kwargs: Any) -> None:
        self._log(logging.WARNING, event, run_id, **kwargs)

    def error(self, event: str, run_id: str = "", **kwargs: Any) -> None:
        self._log(logging.ERROR, event, run_id, **kwargs)

    def debug(self, event: str, run_id: str = "", **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, run_id, **kwargs)

    def stage_start(self, stage: str, run_id: str = "", **context: Any) -> None:
        self._log(logging.INFO, f"START: {stage}", run_id, **context)

    def stage_end(self, stage: str, run_id: str = "", duration_ms: float = 0.0, **context: Any) -> None:
        self._log(logging.INFO, f"END: {stage}", run_id, duration_ms=duration_ms, **context)

    def stage_error(self, stage: str, error: Exception, run_id: str = "", **context: Any) -> None:
        self._log(
            logging.ERROR,
            f"ERROR: {stage}: {error}",
            run_id,
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
            **context,
        )

    def metric(self, stage: str, metric_name: str, value: Any, run_id: str = "", **context: Any) -> None:
        self._log(logging.INFO, f"METRIC: {metric_name}={value}", run_id, metric_name=metric_name, value=value, **context)

    def anomaly(self, stage: str, description: str, run_id: str = "", **context: Any) -> None:
        self._log(logging.WARNING, f"ANOMALY: {description}", run_id, **context)


class DomainLogger:
    """Domain-specific logger with structured output."""

    def __init__(self, domain: str, handler: logging.Handler):
        self.logger = logging.getLogger(f"domain.{domain}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def _log(self, level: int, stage: str, message: str, run_id: str = "", **context: Any) -> None:
        extra = {
            "run_id": run_id,
            "timestamp": time.time(),
            "stage": stage,
            "message": message,
            **context,
        }
        self.logger.log(level, json.dumps(extra, default=str))

    def stage_start(self, stage: str, run_id: str = "", **context: Any) -> None:
        self._log(logging.INFO, stage, f"START: {stage}", run_id, **context)

    def stage_end(self, stage: str, run_id: str = "", duration_ms: float = 0.0, **context: Any) -> None:
        self._log(logging.INFO, stage, f"END: {stage}", run_id, duration_ms=duration_ms, **context)

    def stage_error(self, stage: str, error: Exception, run_id: str = "", **context: Any) -> None:
        self._log(
            logging.ERROR,
            stage,
            f"ERROR: {stage}: {error}",
            run_id,
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
            **context,
        )

    def metric(self, stage: str, metric_name: str, value: Any, run_id: str = "", **context: Any) -> None:
        self._log(logging.INFO, stage, f"METRIC: {metric_name}={value}", run_id, metric_name=metric_name, value=value, **context)

    def anomaly(self, stage: str, description: str, run_id: str = "", **context: Any) -> None:
        self._log(logging.WARNING, stage, f"ANOMALY: {description}", run_id, **context)

    def info(self, message: str, run_id: str = "", **context: Any) -> None:
        self._log(logging.INFO, "info", message, run_id, **context)

    def warning(self, message: str, run_id: str = "", **context: Any) -> None:
        self._log(logging.WARNING, "warning", message, run_id, **context)

    def error(self, message: str, run_id: str = "", **context: Any) -> None:
        self._log(logging.ERROR, "error", message, run_id, **context)

    def debug(self, message: str, run_id: str = "", **context: Any) -> None:
        self._log(logging.DEBUG, "debug", message, run_id, **context)


# Domain-specific loggers
iri_logger = DomainLogger("iri", iri_handler)
vision_logger = DomainLogger("vision", vision_handler)
spatial_logger = DomainLogger("spatial", spatial_handler)
fusion_logger = DomainLogger("fusion", fusion_handler)


def get_pipeline_logger(name: str) -> PipelineLogger:
    return PipelineLogger(name)


# =========================================================
# MONITORING DECORATOR
# =========================================================

def monitor(stage: str, domain: Optional[str] = None) -> Callable:
    """Decorator to monitor function execution with timing and error tracking."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run_id = kwargs.get("run_id", str(uuid.uuid4())[:8])
            logger = DomainLogger(domain, pipeline_handler) if domain else get_pipeline_logger(stage)

            logger.stage_start(stage, run_id=run_id)
            start = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.stage_end(stage, run_id=run_id, duration_ms=round(duration_ms, 2))
                logger.info(f"SUCCESS: {stage}", run_id=run_id, duration_ms=round(duration_ms, 2))
                return result
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.stage_error(stage, exc, run_id=run_id, duration_ms=round(duration_ms, 2))
                raise

        return wrapper
    return decorator


# =========================================================
# FRONTEND LOGGING HELPERS (for API responses)
# =========================================================

class FrontendEventLogger:
    """Logger for frontend events - logs to backend via API."""

    def __init__(self, api_base: str = "http://127.0.0.1:8000"):
        self.api_base = api_base
        self._buffer: list[dict] = []
        self._max_buffer = 50

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Queue a frontend event for batch upload."""
        entry = {
            "event_type": event_type,
            "timestamp": time.time(),
            "session_id": payload.get("session_id", str(uuid.uuid4())[:8]),
            "payload": payload,
        }
        self._buffer.append(entry)

        if len(self._buffer) >= self._max_buffer:
            self.flush()

    def flush(self) -> None:
        """Send buffered events to backend."""
        if not self._buffer:
            return

        events = self._buffer[:]
        self._buffer.clear()

        try:
            import requests
            requests.post(
                f"{self.api_base}/api/v1/logs/frontend",
                json={"events": events},
                timeout=2,
            )
        except Exception:
            # Silently drop logs if backend is unavailable
            pass

    def error(self, message: str, **context: Any) -> None:
        self.log("frontend_error", {"message": message, **context})

    def user_action(self, action: str, **context: Any) -> None:
        self.log("user_action", {"action": action, **context})

    def api_call(self, endpoint: str, method: str, status: int, duration_ms: float, **context: Any) -> None:
        self.log("api_call", {
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "duration_ms": duration_ms,
            **context,
        })

    def computation(self, stage: str, duration_ms: float, input_size: Optional[Dict[str, int]] = None, **context: Any) -> None:
        self.log("computation", {
            "stage": stage,
            "duration_ms": duration_ms,
            "input_size": input_size,
            **context,
        })


frontend_logger = FrontendEventLogger()


# =========================================================
# ROOT LOGGER CONFIGURATION
# =========================================================

root_logger = logging.getLogger("roadessy")
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(app_handler)
root_logger.addHandler(error_handler)

# Suppress noisy third-party loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
