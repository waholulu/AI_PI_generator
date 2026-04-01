from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from agents import cache_utils


def test_save_and_load_json_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    cache_utils.save_json_cache("openalex", "k1", {"a": 1})
    loaded = cache_utils.load_json_cache("openalex", "k1", max_age_hours=24)

    assert loaded == {"a": 1}


def test_cache_ttl_expired_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    path = tmp_path / "data" / "cache" / "openalex" / "k2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"created_at": old_ts, "data": {"x": 1}}, f)

    loaded = cache_utils.load_json_cache("openalex", "k2", max_age_hours=1)
    assert loaded is None
