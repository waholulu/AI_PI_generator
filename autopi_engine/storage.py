from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .contracts import Manifest, PendingAction

_RUN_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_root() -> Path:
    root = Path(os.getenv("AUTOPI_DATA_ROOT", ".")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_root() -> Path:
    root = data_root() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_run_id(run_id: str) -> str:
    if not run_id or any(c not in _RUN_ID_CHARS for c in run_id):
        raise ValueError("run_id must contain only letters, digits, dash, and underscore")
    return run_id


def run_root(run_id: str) -> Path:
    run_id = validate_run_id(run_id)
    root = (runs_root() / run_id).resolve()
    if runs_root() not in root.parents and root != runs_root():
        raise ValueError("run path escaped AUTOPI_DATA_ROOT/runs")
    root.mkdir(parents=True, exist_ok=True)
    for child in ["inputs", "output", "data", "config"]:
        (root / child).mkdir(exist_ok=True)
    return root


def manifest_path(run_id: str) -> Path:
    return run_root(run_id) / "manifest.json"


def input_path(run_id: str, kind: str) -> Path:
    safe_kind = kind.replace("-", "_")
    if any(c not in _RUN_ID_CHARS for c in safe_kind):
        raise ValueError("input kind contains invalid characters")
    return run_root(run_id) / "inputs" / f"{safe_kind}.json"


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@contextlib.contextmanager
def run_lock(run_id: str) -> Iterator[None]:
    path = run_root(run_id) / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as fh:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_manifest(run_id: str) -> Manifest:
    path = manifest_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"unknown run_id: {run_id}")
    return Manifest.model_validate(read_json(path))


def save_manifest(manifest: Manifest) -> Manifest:
    manifest.updated_at = utc_now()
    write_json_atomic(manifest_path(manifest.run_id), manifest.model_dump(mode="json"))
    return manifest


def new_manifest(run_id: str, domain: str) -> Manifest:
    now = utc_now()
    manifest = Manifest(
        run_id=run_id,
        domain=domain,
        created_at=now,
        updated_at=now,
        pending_action=PendingAction(
            kind="search_plan",
            prompt=(
                "Draft a JSON SearchPlan for the initial OpenAlex field scan. "
                "Use purpose, queries, per_query_limit, final_limit, and notes."
            ),
        ),
    )
    save_manifest(manifest)
    return manifest
