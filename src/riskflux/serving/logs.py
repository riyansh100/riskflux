"""Structured JSON logging for the API.

- One JSON object per line on stdout. Cloud Run ships stdout to Cloud Logging, which
  reads the `severity` field natively (so levels show correctly in the console).
- A request ID is bound to every log line of a request (and returned as X-Request-ID),
  so one request can be traced end to end.
- Every prediction is also logged with its inputs and outputs: this is the raw material
  for drift monitoring (Step 9). Optionally mirrored to a JSONL file for local analysis.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog


def _add_severity(logger, method_name: str, event_dict: dict) -> dict:
    event_dict["severity"] = method_name.upper()
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_severity,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


class PredictionSink:
    """Appends one JSON line per scored application to a file (if configured)."""

    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, request_id: str, application: dict, result: dict) -> None:
        if self.path is None:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "model_version": result["model_version"],
            "pd": result["pd"],
            "decision": result["decision"],
            "policy_flags": result["policy_flags"],
            "features": application,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
