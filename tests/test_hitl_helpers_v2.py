"""Tests for the TENTATIVE pool helpers added to agents/hitl_helpers.py — Day 6 TDD."""

import json
import os
from pathlib import Path

import pytest

from agents.hitl_helpers import (
    kill_tentative,
    load_tentative_topics,
    promote_tentative,
    rerun_tentative_reflection,
)
from models.topic_schema import (
    Contribution,
    ContributionPrimary,
    ExposureFamily,
    ExposureX,
    Frequency,
    IdentificationPrimary,
    IdentificationStrategy,
    OutcomeFamily,
    OutcomeY,
    SamplingMode,
    SpatialScope,
    TemporalScope,
    Topic,
    TopicMeta,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_topic_dict(topic_id: str = "tent001") -> dict:
    t = Topic(
        meta=TopicMeta(topic_id=topic_id),
        exposure_X=ExposureX(
            family=ExposureFamily.AIR_QUALITY,
            specific_variable="PM2.5",
            spatial_unit="tract",
        ),
        outcome_Y=OutcomeY(
            family=OutcomeFamily.HEALTH,
            specific_variable="mortality",
            spatial_unit="tract",
        ),
        spatial_scope=SpatialScope(
            geography="US cities",
            spatial_unit="tract",
            sampling_mode=SamplingMode.PANEL,
        ),
        temporal_scope=TemporalScope(
            start_year=2010, end_year=2020, frequency=Frequency.ANNUAL
        ),
        identification=IdentificationStrategy(
            primary=IdentificationPrimary.FE,
            key_threats=["confounding"],
            mitigations=["fe_controls"],
        ),
        contribution=Contribution(
            primary=ContributionPrimary.CAUSAL_REFINEMENT,
            statement="Causal estimate.",
        ),
        free_form_title=f"Tentative topic {topic_id}",
    )
    return t.model_dump(mode="json")


def make_pool_file(tmp_path: Path, n: int = 2) -> Path:
    pool_path = tmp_path / "tentative_pool.json"
    pool = {
        "run_id": "test_run",
        "tentative": [
            {
                "topic_id": f"tent{i:03d}",
                "title": f"Tentative topic {i}",
                "failed_gates": ["G1"],
                "declared_sources": ["NHGIS"],
                "legacy_six_gates": {"impact": False},
                "topic_dict": make_topic_dict(f"tent{i:03d}"),
                "trace_rounds": 1,
            }
            for i in range(n)
        ],
    }
    pool_path.write_text(json.dumps(pool, indent=2))
    return pool_path


# ── load_tentative_topics ─────────────────────────────────────────────────────

def test_load_returns_empty_list_when_file_missing(tmp_path):
    pool = load_tentative_topics(str(tmp_path / "nonexistent.json"))
    assert pool == []


def test_load_returns_tentative_entries(tmp_path):
    pool_path = make_pool_file(tmp_path, n=3)
    pool = load_tentative_topics(str(pool_path))
    assert len(pool) == 3


def test_load_returns_empty_on_corrupt_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json")
    pool = load_tentative_topics(str(bad_file))
    assert pool == []


# ── promote_tentative ─────────────────────────────────────────────────────────

def test_promote_moves_entry_to_screening(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    pool_path = make_pool_file(tmp_path, n=2)

    result = promote_tentative(0, path=str(pool_path))
    assert result is True

    # Pool should now have 1 entry
    remaining = load_tentative_topics(str(pool_path))
    assert len(remaining) == 1

    # Screening should have the promoted entry at rank-1
    screening_path = tmp_path / "output" / "topic_screening.json"
    assert screening_path.exists()
    screening = json.loads(screening_path.read_text())
    assert len(screening["candidates"]) >= 1
    assert screening["candidates"][0]["rank"] == 1
    assert screening["candidates"][0].get("promoted_from_tentative") is True


def test_promote_invalid_index_returns_false(tmp_path):
    pool_path = make_pool_file(tmp_path, n=1)
    result = promote_tentative(99, path=str(pool_path))
    assert result is False


# ── kill_tentative ────────────────────────────────────────────────────────────

def test_kill_removes_entry_from_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    pool_path = make_pool_file(tmp_path, n=2)

    result = kill_tentative(0, domain="Urban Planning", path=str(pool_path))
    assert result is True

    remaining = load_tentative_topics(str(pool_path))
    assert len(remaining) == 1


def test_kill_writes_to_graveyard(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    pool_path = make_pool_file(tmp_path, n=1)

    kill_tentative(0, domain="Urban Planning", path=str(pool_path))

    # Graveyard file should exist
    import hashlib
    domain_key = hashlib.md5("urban planning".encode()).hexdigest()[:8]
    graveyard = tmp_path / "output" / f"ideas_graveyard_{domain_key}.json"
    assert graveyard.exists()
    data = json.loads(graveyard.read_text())
    assert len(data) == 1
    assert data[0]["rejection_reason"] == "killed_from_tentative_pool"


def test_kill_invalid_index_returns_false(tmp_path):
    pool_path = make_pool_file(tmp_path, n=1)
    result = kill_tentative(99, domain="test", path=str(pool_path))
    assert result is False


# ── rerun_tentative_reflection ────────────────────────────────────────────────

def test_rerun_invalid_index_returns_none(tmp_path):
    pool_path = make_pool_file(tmp_path, n=1)
    result = rerun_tentative_reflection(99, path=str(pool_path))
    assert result is None


def test_rerun_updates_last_rerun_status(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    pool_path = make_pool_file(tmp_path, n=1)

    # Run rerun (no LLM → neutral scores → ACCEPTED typically)
    updated = rerun_tentative_reflection(0, path=str(pool_path))
    assert updated is not None
    assert "last_rerun_status" in updated
    assert updated["last_rerun_status"] in ("ACCEPTED", "TENTATIVE", "REJECTED")
