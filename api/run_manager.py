"""
Run registry for Auto-PI pipeline executions.

Tracks run metadata (status, timing, errors) in memory with optional
SQLite persistence. In cloud deployments the same SQLite file lives on
the persistent volume, so metadata survives container restarts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from api.models import RunListItem, RunStatus
from agents import settings


_lock = threading.Lock()


def _db_path() -> str:
    # Keep run registry global, independent from per-run artifact scope.
    return str(settings.global_output_dir() / "runs.sqlite")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id       TEXT PRIMARY KEY,
                thread_id    TEXT NOT NULL,
                domain_input TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'starting',
                current_node TEXT,
                started_at   TEXT NOT NULL,
                completed_at TEXT,
                error        TEXT
            )
            """
        )
        conn.commit()


# Initialise on import
_init_db()


def create_run(domain_input: str, thread_id: Optional[str] = None) -> RunStatus:
    run_id = str(uuid.uuid4())
    tid = thread_id or run_id
    now = datetime.now(timezone.utc).isoformat()
    settings.run_root(run_id, create=True)

    with _lock, _get_conn() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, thread_id, domain_input, status, started_at) VALUES (?,?,?,?,?)",
            (run_id, tid, domain_input, "starting", now),
        )
        conn.commit()

    return RunStatus(
        run_id=run_id,
        thread_id=tid,
        domain_input=domain_input,
        status="starting",
        started_at=datetime.fromisoformat(now),
    )


def get_run(run_id: str) -> Optional[RunStatus]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return _row_to_status(row)


def list_runs() -> List[RunListItem]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    return [
        RunListItem(
            run_id=r["run_id"],
            domain_input=r["domain_input"],
            status=r["status"],
            started_at=datetime.fromisoformat(r["started_at"]),
            completed_at=datetime.fromisoformat(r["completed_at"]) if r["completed_at"] else None,
        )
        for r in rows
    ]


def update_status(
    run_id: str,
    status: str,
    current_node: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat() if status in ("completed", "failed", "aborted") else None
    with _lock, _get_conn() as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = ?, current_node = ?, completed_at = COALESCE(?, completed_at), error = COALESCE(?, error)
            WHERE run_id = ?
            """,
            (status, current_node, now, error, run_id),
        )
        conn.commit()


def _row_to_status(row: sqlite3.Row) -> RunStatus:
    return RunStatus(
        run_id=row["run_id"],
        thread_id=row["thread_id"],
        domain_input=row["domain_input"],
        status=row["status"],
        current_node=row["current_node"],
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        error=row["error"],
    )
