"""Tests for development_pack_status.evaluate_development_pack_readiness."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.development_pack_status import evaluate_development_pack_readiness


def _make_candidate(
    automation_risk: str = "low",
    required_secrets: list[str] | None = None,
    technology_tags: list[str] | None = None,
) -> dict:
    return {
        "candidate_id": "test_001",
        "automation_risk": automation_risk,
        "required_secrets": required_secrets or [],
        "technology_tags": technology_tags or [],
    }


def _make_gate(overall: str = "pass", shortlist: str = "ready") -> dict:
    return {"overall": overall, "shortlist_status": shortlist}


def _populate_pack(pack_dir: Path) -> None:
    required = [
        "implementation_spec.json",
        "claude_task_prompt.md",
        "data_contract.yaml",
        "feature_plan.yaml",
        "analysis_plan.yaml",
        "acceptance_tests.md",
    ]
    for fname in required:
        (pack_dir / fname).write_text("content", encoding="utf-8")


def test_ready_when_all_conditions_met(tmp_path: Path) -> None:
    pack_dir = tmp_path / "test_001"
    pack_dir.mkdir()
    _populate_pack(pack_dir)
    result = evaluate_development_pack_readiness(
        _make_candidate(), _make_gate(), pack_dir
    )
    assert result["claude_code_ready"] is True
    assert result["development_pack_status"] == "ready"
    assert result["missing_files"] == []
    assert result["blocking_reasons"] == []


def test_blocked_when_high_risk(tmp_path: Path) -> None:
    pack_dir = tmp_path / "test_001"
    pack_dir.mkdir()
    _populate_pack(pack_dir)
    result = evaluate_development_pack_readiness(
        _make_candidate(automation_risk="high"), _make_gate(), pack_dir
    )
    assert result["claude_code_ready"] is False
    assert "high_automation_risk" in result["blocking_reasons"]


def test_blocked_when_required_secrets(tmp_path: Path) -> None:
    pack_dir = tmp_path / "test_001"
    pack_dir.mkdir()
    _populate_pack(pack_dir)
    result = evaluate_development_pack_readiness(
        _make_candidate(required_secrets=["GOOGLE_MAPS_API_KEY"]),
        _make_gate(),
        pack_dir,
    )
    assert result["claude_code_ready"] is False
    assert any("required_secrets" in r for r in result["blocking_reasons"])


def test_blocked_when_experimental_tag(tmp_path: Path) -> None:
    pack_dir = tmp_path / "test_001"
    pack_dir.mkdir()
    _populate_pack(pack_dir)
    result = evaluate_development_pack_readiness(
        _make_candidate(technology_tags=["streetview_cv"]),
        _make_gate(),
        pack_dir,
    )
    assert result["claude_code_ready"] is False
    assert any("experimental_tags" in r for r in result["blocking_reasons"])


def test_blocked_when_gate_failed(tmp_path: Path) -> None:
    pack_dir = tmp_path / "test_001"
    pack_dir.mkdir()
    _populate_pack(pack_dir)
    result = evaluate_development_pack_readiness(
        _make_candidate(),
        _make_gate(overall="fail", shortlist="blocked"),
        pack_dir,
    )
    assert result["claude_code_ready"] is False
    assert "gate_failed" in result["blocking_reasons"]


def test_not_generated_when_no_pack_dir() -> None:
    result = evaluate_development_pack_readiness(
        _make_candidate(), _make_gate(), None
    )
    assert result["development_pack_status"] == "not_generated"
    assert result["claude_code_ready"] is False


def test_review_required_when_shortlist_review(tmp_path: Path) -> None:
    pack_dir = tmp_path / "test_001"
    pack_dir.mkdir()
    _populate_pack(pack_dir)
    result = evaluate_development_pack_readiness(
        _make_candidate(required_secrets=["SOME_KEY"]),
        _make_gate(overall="warning", shortlist="review"),
        pack_dir,
    )
    assert result["development_pack_status"] == "review_required"
    assert result["claude_code_ready"] is False


def test_warning_gate_still_ready_without_other_blockers(tmp_path: Path) -> None:
    """Gate warning (not fail) should not block Claude Code Ready."""
    pack_dir = tmp_path / "test_001"
    pack_dir.mkdir()
    _populate_pack(pack_dir)
    result = evaluate_development_pack_readiness(
        _make_candidate(),
        _make_gate(overall="warning", shortlist="ready"),
        pack_dir,
    )
    assert result["claude_code_ready"] is True
    assert result["development_pack_status"] == "ready"
