"""
Per-run log capture for the Auto-PI API.

Installs a custom Python logging handler that routes log records into
an in-memory store keyed by run_id. The API layer can then query logs
per run via get_logs().
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from api.models import LogEntry

_store: Dict[str, List[LogEntry]] = defaultdict(list)
_lock = threading.Lock()

# Current run context (set/cleared by the API around pipeline execution)
_current_run_id: Optional[str] = None


def set_current_run(run_id: Optional[str]) -> None:
    global _current_run_id
    _current_run_id = run_id


class RunLogHandler(logging.Handler):
    """Routes log records into the in-memory store under the active run_id."""

    def emit(self, record: logging.LogRecord) -> None:
        run_id = getattr(record, "run_id", None) or _current_run_id
        if run_id is None:
            return
        entry = LogEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            logger=record.name,
            msg=record.getMessage(),
            run_id=run_id,
            node=getattr(record, "node", None),
        )
        with _lock:
            _store[run_id].append(entry)


def install_handler() -> None:
    """Install the RunLogHandler on the root logger (call once at startup)."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, RunLogHandler):
            return  # already installed
    root.addHandler(RunLogHandler())


def get_logs(
    run_id: str,
    level: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 500,
) -> List[LogEntry]:
    with _lock:
        entries = list(_store.get(run_id, []))

    if level:
        entries = [e for e in entries if e.level == level.upper()]
    if since:
        since_str = since.isoformat()
        entries = [e for e in entries if e.ts >= since_str]

    return entries[-limit:]


def clear_logs(run_id: str) -> None:
    with _lock:
        _store.pop(run_id, None)
