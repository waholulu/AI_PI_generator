"""
Lightweight JSON file cache helpers.

Used for query/hash + TTL caching to avoid repeated LLM and OpenAlex calls
across runs while keeping architecture file-based and simple.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import settings
from agents.logging_config import get_logger

logger = get_logger(__name__)


def build_cache_key(prefix: str, payload: Any) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def _cache_file(namespace: str, cache_key: str) -> Path:
    d = settings.cache_dir() / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{cache_key}.json"


def load_json_cache(namespace: str, cache_key: str, max_age_hours: float) -> Any | None:
    if max_age_hours <= 0:
        return None

    path = _cache_file(namespace, cache_key)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        created_at = payload.get("created_at")
        if not isinstance(created_at, str):
            return None
        created_dt = datetime.fromisoformat(created_at)
        age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None
        return payload.get("data")
    except Exception as exc:
        logger.debug("Cache read failed for %s/%s: %s", namespace, cache_key, exc)
        return None


def save_json_cache(namespace: str, cache_key: str, data: Any) -> None:
    path = _cache_file(namespace, cache_key)
    wrapped = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wrapped, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.debug("Cache write failed for %s/%s: %s", namespace, cache_key, exc)
