"""
Structured logging configuration for Auto-PI.

Provides JSON-formatted log output that can be captured by the API layer
and queried per-run. All agents should use `get_logger(__name__)` instead
of print().
"""

import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emits one JSON object per log line for machine-readable consumption."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        # Allow attaching extra structured fields via `extra={"run_id": ...}`
        for key in ("run_id", "node", "step"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """
    Configure root logger. Call once at application startup.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, use JSON formatter. If False, use human-readable.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Use as: logger = get_logger(__name__)"""
    return logging.getLogger(name)
