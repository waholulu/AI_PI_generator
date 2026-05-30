"""Tests for the candidate factory ideation path.

All tests are mock-only — no API keys or live network calls required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents import settings
from agents.candidate_factory_ideation import run_candidate_factory_ideation
from agents.ideation_agent import ideation_node


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def run_scope(tmp_path, monkeypatch):
    """Activate a temporary run scope so all file writes go to tmp_path."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    token = settings.activate_run_scope("test-run-001")
    yield tmp_path
    settings.deactivate_run_scope(token)


@pytest.fixture(scope="module")
def factory_result(tmp_path_factory):
    """Run the candidate factory once per module; share result across tests 3-5.

    Keeps AUTOPI_DATA_ROOT and the run scope active for the *whole* module so
    helpers that re-resolve paths via `settings.topic_screening_path()` (e.g.
    `apply_idea_selection_by_candidate_id`) see the same tmp directory the
    factory wrote into. Reverts both on teardown.
    """
    import os
    tmp = tmp_path_factory.mktemp("factory_run")
    prev = os.environ.get("AUTOPI_DATA_ROOT")
    os.environ["AUTOPI_DATA_ROOT"] = str(tmp)
    token = settings.activate_run_scope("test-run-shared")
    try:
        result = run_candidate_factory_ideation({
            "domain_input": "Built environment and health",
            "template_id": "built_environment_health",
            "candidate_factory_enabled": True,
            "enable_experimental": False,
            "execution_status": "starting",
        })
        yield result
    finally:
        settings.deactivate_run_scope(token)
        if prev is None:
            os.environ.pop("AUTOPI_DATA_ROOT", None)
        else:
            os.environ["AUTOPI_DATA_ROOT"] = prev


# ── test 1: ideation_node routes to factory when enabled ─────────────────────

def test_factory_path_taken_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    state = {
        "domain_input": "Built environment and health",
        "template_id": "built_environment_health",
        "candidate_factory_enabled": True,
        "execution_status": "starting",
    }
    called = []

    def fake_factory(s):
        called.append(s["template_id"])
        return {"execution_status": "ideation_complete"}

    with patch("agents.candidate_factory_ideation.run_candidate_factory_ideation", fake_factory):
        result = ideation_node(state)

    assert called, "Candidate Factory was not called"
    assert called[0] == "built_environment_health"
    assert result["execution_status"] == "ideation_complete"


# ── test 2: Candidate Factory is always called (default template applied) ────

def test_factory_called_with_default_template_when_no_template_id():
    """ideation_node() must route to Candidate Factory even when template_id is absent.

    The router applies 'built_environment_health' as the default template.
    """
    state = {
        "domain_input": "Urban heat and health",
        "execution_status": "starting",
    }
    factory_called = []

    with patch(
        "agents.candidate_factory_ideation.run_candidate_factory_ideation",
        side_effect=lambda s: factory_called.append(s.get("template_id")) or {"execution_status": "ideation_complete"},
    ):
        ideation_node(state)

    assert factory_called, "Candidate Factory must be called when no template_id is provided"
    assert factory_called[0] == "built_environment_health", (
        f"Default template should be 'built_environment_health', got {factory_called[0]!r}"
    )


# ── test 3: candidate_cards.json is written with ≥20 entries ──────────────────

def test_candidate_cards_written(factory_result):
    result = factory_result
    cards_path = Path(result["candidate_cards_path"])
    assert cards_path.exists(), "candidate_cards.json was not created"

    cards = json.loads(cards_path.read_text())
    assert len(cards) >= 20, f"Expected ≥20 candidates, got {len(cards)}"

    required_fields = {"candidate_id", "exposure_source", "outcome_source", "method", "title", "scores"}
    for card in cards:
        missing = required_fields - card.keys()
        assert not missing, f"Card {card.get('candidate_id')} missing fields: {missing}"
        assert card["scores"].get("overall", 0) > 0


# ── test 4: topic_screening.json is the shortlist (≤ shortlist_size candidates) ──

def test_topic_screening_written(factory_result):
    """topic_screening.json must contain only the shortlist (default 5), not the full pool."""
    result = factory_result
    screening_path = Path(result["candidate_topics_path"])
    assert screening_path.exists(), "topic_screening.json was not created"

    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    assert screening.get("ideation_mode") == "candidate_factory", \
        "topic_screening.json must carry ideation_mode=candidate_factory"

    candidates = screening["candidates"]
    default_shortlist_size = 5
    assert 1 <= len(candidates) <= default_shortlist_size, (
        f"topic_screening.json should have 1–{default_shortlist_size} shortlist candidates, "
        f"got {len(candidates)}"
    )

    for c in candidates:
        assert "candidate_id" in c, "candidate_id missing from screening entry"
        assert "title" in c
        assert "research_question" in c
        assert "rerank" in c, "rerank block missing from screening entry"
        assert "rerank_score" in c["rerank"], "rerank_score missing from screening entry"
        assert "empirical_value_score" in c["rerank"], "empirical value missing from rerank"
        assert "polished_title" in c, "polished_title missing from screening entry"
        assert "tech_lens_type" in c, "tech_lens_type missing from screening entry"
        assert "empirical_deepening_claim" in c, "empirical_deepening_claim missing from screening entry"
        # evaluation block must use user_visible_reasons, not raw gate flags
        assert "evaluation" in c, "evaluation block missing from screening entry"
        eval_block = c["evaluation"]
        assert "user_visible_reasons" in eval_block, "evaluation must have user_visible_reasons"
        assert "score" in eval_block, "evaluation must have score"
        assert "rerank" in eval_block, "evaluation must expose rerank diagnostics"
        # raw gate reasons must NOT appear at the top level
        assert "reasons" not in c, "raw gate reasons must not appear at top level of screening entry"

    # Full pool must still be in candidate_cards.json (not merged into topic_screening)
    cards_path = Path(result["candidate_cards_path"])
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    assert len(cards) > len(candidates), \
        "candidate_cards.json (full pool) must have more entries than topic_screening.json (shortlist)"


def test_speculative_streetview_candidates_written(factory_result):
    """Stable runs should surface review-only street-view ideas in a side lane."""
    result = factory_result
    speculative_path = Path(result["speculative_candidates_path"])
    assert speculative_path.exists(), "speculative_candidates.json was not created"

    screening = json.loads(Path(result["candidate_topics_path"]).read_text(encoding="utf-8"))
    speculative = screening.get("speculative_candidates", [])
    assert speculative, "Expected at least one speculative candidate"
    assert all(c.get("selectable") is False for c in speculative)
    assert all(c.get("readiness") == "blocked" for c in speculative)
    assert any(c.get("exposure_family") == "streetview_built_form" for c in speculative)
    assert any("streetview_cv" in (c.get("technology_tags") or []) for c in speculative)
    assert not {
        c["candidate_id"] for c in screening["candidates"]
    } & {
        c["candidate_id"] for c in speculative
    }, "Speculative candidate IDs must not collide with safe shortlist IDs"


def test_experimental_only_template_does_not_crash_when_all_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    token = settings.activate_run_scope("test-run-experimental-only")
    try:
        result = run_candidate_factory_ideation({
            "domain_input": "Built environment and health",
            "template_id": "built_environment_health_experimental",
            "candidate_factory_enabled": True,
            "enable_experimental": True,
            "technology_options": {"streetview_cv": True},
            "execution_status": "starting",
        })
    finally:
        settings.deactivate_run_scope(token)

    assert result["execution_status"] == "ideation_complete"
    screening = json.loads(Path(result["candidate_topics_path"]).read_text(encoding="utf-8"))
    assert screening["candidates"] == []
    assert screening["speculative_candidates"]


def test_tte_factory_writes_method_screening(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    token = settings.activate_run_scope("test-run-tte")
    try:
        result = run_candidate_factory_ideation({
            "domain_input": "Built environment and health",
            "template_id": "built_environment_health_tte",
            "candidate_factory_enabled": True,
            "enable_experimental": False,
            "max_candidates": 20,
            "execution_status": "starting",
        })
    finally:
        settings.deactivate_run_scope(token)

    screening = json.loads(Path(result["candidate_topics_path"]).read_text(encoding="utf-8"))
    assert screening["candidates"]
    for candidate in screening["candidates"]:
        method_screening = candidate.get("method_screening") or {}
        assert method_screening.get("primary_method") == candidate["method"]
        assert method_screening["methods"][candidate["method"]]["status"] == "eligible"
        assert method_screening["methods"]["instrumental_variable"]["status"] == "rejected"


# ── test 5: a candidate from the shortlist can be selected via candidate_id ───

def test_candidate_selection(factory_result):
    """apply_idea_selection_by_candidate_id works on any shortlist candidate."""
    result = factory_result
    # Pick the first candidate from the shortlist (rank-1 after factory ranking).
    screening = json.loads(Path(result["candidate_topics_path"]).read_text(encoding="utf-8"))
    first_id = screening["candidates"][0]["candidate_id"]

    from agents.hitl_helpers import apply_idea_selection_by_candidate_id

    selected_title = apply_idea_selection_by_candidate_id(first_id)
    assert selected_title is not None, (
        f"Selection of {first_id!r} returned None — candidate_id not found in shortlist"
    )
    assert isinstance(selected_title, str) and len(selected_title) > 0, \
        "Expected a non-empty title string"
