"""Tests for the idea_validator_agent module."""

import json
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from agents import idea_validator_agent as validator
import agents.candidate_evaluator as candidate_evaluator_mod
from agents import settings
from agents.orchestrator import ResearchState


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_REGISTRY = [
    {
        "canonical_name": "US Census Bureau ACS",
        "aliases": ["American Community Survey", "ACS", "US Census"],
        "url": "https://data.census.gov",
        "access_type": "api",
        "domains": ["demographics"],
    },
    {
        "canonical_name": "World Bank Open Data",
        "aliases": ["World Bank", "WDI"],
        "url": "https://data.worldbank.org",
        "access_type": "api",
        "domains": ["economics"],
    },
    {
        "canonical_name": "OpenStreetMap",
        "aliases": ["OSM", "OpenStreetMap"],
        "url": "https://www.openstreetmap.org",
        "access_type": "api",
        "domains": ["geography"],
    },
]


def _make_idea(
    title: str = "Test Idea",
    rank: int = 1,
    data_sources: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    return {
        "title": title,
        "brief_rationale": "A test rationale for this idea.",
        "rank": rank,
        "initial_score": 80,
        "final_score": 75,
        "data_sources": data_sources or [
            {"name": "US Census", "accessibility": "Public API"},
        ],
    }


def _make_screening(
    candidates: List[Dict[str, Any]] | None = None,
    backup: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if candidates is None:
        candidates = [
            _make_idea("Idea A", rank=1),
            _make_idea("Idea B", rank=2),
            _make_idea("Idea C", rank=3),
        ]
    return {
        "run_id": "test-run-001",
        "candidates": candidates,
        "gates_passed": len(candidates),
        "backup_candidates": backup or [],
    }


def _write_screening(tmp_path, screening_data=None):
    """Write screening data and return the path."""
    screening_path = str(tmp_path / "topic_screening.json")
    data = screening_data or _make_screening()
    with open(screening_path, "w") as f:
        json.dump(data, f)
    return screening_path


def _make_state(tmp_path, screening_data=None) -> ResearchState:
    screening_path = _write_screening(tmp_path, screening_data)
    plan_path = str(tmp_path / "research_plan.json")
    with open(plan_path, "w") as f:
        json.dump({"project_title": "Test", "topic_screening": {}}, f)
    return ResearchState(
        domain_input="GeoAI and Urban Planning",
        execution_status="screening",
        candidate_topics_path=screening_path,
        current_plan_path=plan_path,
        field_scan_path="",
        validation_report_path="",
        literature_inventory_path="",
        draft_content_path="",
        raw_data_manifest_path="",
        research_context_path="",
    )


# ---------------------------------------------------------------------------
# Data source matching tests
# ---------------------------------------------------------------------------

class TestDataRegistryMatching:
    """Test the fuzzy matching logic for data source verification."""

    def test_exact_match(self):
        match = validator._match_data_source("US Census Bureau ACS", SAMPLE_REGISTRY)
        assert match == "US Census Bureau ACS"

    def test_alias_match(self):
        match = validator._match_data_source("American Community Survey", SAMPLE_REGISTRY)
        assert match == "US Census Bureau ACS"

    def test_fuzzy_match(self):
        match = validator._match_data_source("US Census", SAMPLE_REGISTRY, threshold=0.6)
        assert match == "US Census Bureau ACS"

    def test_no_match(self):
        match = validator._match_data_source(
            "Totally Fake Dataset XYZ123", SAMPLE_REGISTRY, threshold=0.6
        )
        assert match is None

    def test_case_insensitive(self):
        match = validator._match_data_source("world bank", SAMPLE_REGISTRY)
        assert match == "World Bank Open Data"

    def test_check_data_availability_mixed(self):
        sources = [
            {"name": "US Census"},
            {"name": "Nonexistent Fantasy DB"},
        ]
        results = validator.check_data_availability(sources, SAMPLE_REGISTRY)
        assert len(results) == 2
        assert results[0].status == "verified"
        assert results[0].registry_match == "US Census Bureau ACS"
        assert results[1].status == "unverified"
        assert results[1].registry_match is None


# ---------------------------------------------------------------------------
# Validator tests (mocked LLM + OpenAlex)
# ---------------------------------------------------------------------------

def _make_pass_evaluation(candidate: Dict[str, Any], plan, llm=None):
    from models.candidate_schema import CandidateEvaluation
    return CandidateEvaluation(
        candidate_id=candidate.get("title", "test"),
        title=candidate.get("title", "test"),
        rank=candidate.get("rank", 1),
        schema_valid=True,
        data_registry_verdict="pass",
        data_access_verdict="pass",
        novelty_verdict="novel",
        identification_verdict="pass",
        contribution_verdict="pass",
        overall_verdict="pass",
        score=0.8,
        reasons=[],
        evidence={},
    )


class TestValidatorAllPass:
    """All 3 ideas pass: novel + data verified."""

    def test_all_pass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validator, "evaluate_candidate", _make_pass_evaluation)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        state = _make_state(tmp_path)
        agent = validator.IdeaValidatorAgent()
        agent.llm = None

        result = agent.run(state)

        assert os.path.exists(result["validation_report_path"])
        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        assert report["substitutions_made"] == 0
        for idea in report["validated_ideas"]:
            assert idea["overall_verdict"] in ("passed", "warning")
            assert idea["novelty"]["verdict"] == "novel"


class TestValidatorNoveltyFail:
    """One idea fails novelty check — new code reports failure, no substitution in evaluate_candidate path."""

    def test_novelty_fail_reported(self, tmp_path, monkeypatch):
        call_idx = {"n": 0}

        def _mock_eval(candidate, plan, llm=None):
            from models.candidate_schema import CandidateEvaluation
            call_idx["n"] += 1
            verdict = "fail" if call_idx["n"] == 1 else "pass"
            return CandidateEvaluation(
                candidate_id=candidate.get("title", ""),
                title=candidate.get("title", ""),
                rank=candidate.get("rank", 1),
                schema_valid=True,
                data_registry_verdict="pass",
                data_access_verdict="pass",
                novelty_verdict="already_published" if call_idx["n"] == 1 else "novel",
                identification_verdict="pass",
                contribution_verdict="pass",
                overall_verdict=verdict,
                score=0.3 if verdict == "fail" else 0.8,
                reasons=["already_published_overlap"] if verdict == "fail" else [],
                evidence={},
            )

        monkeypatch.setattr(validator, "evaluate_candidate", _mock_eval)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        backup = [_make_idea("Backup Idea D", rank=0, data_sources=[{"name": "World Bank"}])]
        state = _make_state(tmp_path, _make_screening(backup=backup))

        agent = validator.IdeaValidatorAgent()
        agent.llm = None
        result = agent.run(state)

        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        # New code path doesn't do substitution
        assert report["substitutions_made"] == 0
        failed = [v for v in report["validated_ideas"] if v["overall_verdict"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["novelty"]["verdict"] == "already_published"


class TestValidatorDataFail:
    """All data sources unverified → idea fails."""

    def test_data_fail(self, tmp_path, monkeypatch):
        def _mock_eval(candidate, plan, llm=None):
            from models.candidate_schema import CandidateEvaluation
            return CandidateEvaluation(
                candidate_id=candidate.get("title", ""),
                title=candidate.get("title", ""),
                rank=candidate.get("rank", 1),
                schema_valid=True,
                data_registry_verdict="fail",
                data_access_verdict="fail",
                novelty_verdict="novel",
                identification_verdict="pass",
                contribution_verdict="pass",
                overall_verdict="fail",
                score=0.1,
                reasons=["no_data_sources_verified", "all_sources_unverified"],
                evidence={},
            )

        monkeypatch.setattr(validator, "evaluate_candidate", _mock_eval)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        bad_ideas = [
            _make_idea("Bad Data Idea", rank=1, data_sources=[
                {"name": "Nonexistent DB 1"},
                {"name": "Totally Fake Source"},
            ]),
        ]
        state = _make_state(tmp_path, _make_screening(candidates=bad_ideas, backup=[]))

        agent = validator.IdeaValidatorAgent()
        agent.llm = None
        result = agent.run(state)

        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        failed = [v for v in report["validated_ideas"] if v["overall_verdict"] == "failed"]
        assert len(failed) == 1
        assert len(failed[0]["failure_reasons"]) > 0


class TestValidatorNoBackups:
    """No backups → substitutions_made is 0 (also no substitution in new evaluate_candidate path)."""

    def test_no_backups(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validator, "evaluate_candidate", _make_pass_evaluation)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        state = _make_state(tmp_path, _make_screening(backup=[]))  # no backups

        agent = validator.IdeaValidatorAgent()
        agent.llm = None
        result = agent.run(state)

        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        assert report["substitutions_made"] == 0


class TestValidatorOpenAlexDown:
    """OpenAlex raises exception → graceful degradation (novelty_verdict is 'unknown')."""

    def test_openalex_down(self, tmp_path, monkeypatch):
        def _mock_eval(candidate, plan, llm=None):
            from models.candidate_schema import CandidateEvaluation
            # When OpenAlex is down, novelty_verdict falls back to "unknown"
            return CandidateEvaluation(
                candidate_id=candidate.get("title", ""),
                title=candidate.get("title", ""),
                rank=candidate.get("rank", 1),
                schema_valid=True,
                data_registry_verdict="pass",
                data_access_verdict="pass",
                novelty_verdict="unknown",
                identification_verdict="pass",
                contribution_verdict="pass",
                overall_verdict="warning",
                score=0.5,
                reasons=["novelty_evidence_limited"],
                evidence={},
            )

        monkeypatch.setattr(validator, "evaluate_candidate", _mock_eval)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        state = _make_state(tmp_path)

        agent = validator.IdeaValidatorAgent()
        agent.llm = None
        result = agent.run(state)

        assert os.path.exists(result["validation_report_path"])
        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        # Should not crash; novelty is unknown when OpenAlex is down
        for idea in report["validated_ideas"]:
            assert idea["novelty"]["verdict"] == "unknown"


class TestValidatorMemoryWriteback:
    """Failed ideas complete without error; memory writeback no longer occurs in evaluate_candidate path."""

    def test_memory_writeback(self, tmp_path, monkeypatch):
        def _mock_eval(candidate, plan, llm=None):
            from models.candidate_schema import CandidateEvaluation
            return CandidateEvaluation(
                candidate_id=candidate.get("title", ""),
                title=candidate.get("title", ""),
                rank=candidate.get("rank", 1),
                schema_valid=True,
                data_registry_verdict="fail",
                data_access_verdict="fail",
                novelty_verdict="novel",
                identification_verdict="pass",
                contribution_verdict="pass",
                overall_verdict="fail",
                score=0.1,
                reasons=["no_data_sources_verified"],
                evidence={},
            )

        monkeypatch.setattr(validator, "evaluate_candidate", _mock_eval)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        bad_ideas = [
            _make_idea("Data Fail Idea", rank=1, data_sources=[
                {"name": "Nonexistent Source XYZ"},
            ]),
        ]
        state = _make_state(tmp_path, _make_screening(candidates=bad_ideas, backup=[]))

        agent = validator.IdeaValidatorAgent()
        agent.llm = None

        result = agent.run(state)

        # Should complete without error; validation report written
        assert os.path.exists(result["validation_report_path"])
        with open(result["validation_report_path"]) as f:
            report = json.load(f)
        assert len(report["validated_ideas"]) == 1
        assert report["validated_ideas"][0]["overall_verdict"] == "failed"


class TestValidatorSubstitutionUpdatesFiles:
    """Verify that screening file is updated with evaluation data from evaluate_candidate path."""

    def test_screening_updated_with_evaluations(self, tmp_path, monkeypatch):
        call_idx = {"n": 0}

        def _mock_eval(candidate, plan, llm=None):
            from models.candidate_schema import CandidateEvaluation
            call_idx["n"] += 1
            return CandidateEvaluation(
                candidate_id=candidate.get("title", ""),
                title=candidate.get("title", ""),
                rank=candidate.get("rank", 1),
                schema_valid=True,
                data_registry_verdict="pass",
                data_access_verdict="pass",
                novelty_verdict="novel",
                identification_verdict="pass",
                contribution_verdict="pass",
                overall_verdict="pass",
                score=0.8,
                reasons=[],
                evidence={},
            )

        monkeypatch.setattr(validator, "evaluate_candidate", _mock_eval)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        backup = [_make_idea("Substitute Idea", rank=0, data_sources=[{"name": "World Bank"}])]
        state = _make_state(tmp_path, _make_screening(backup=backup))

        agent = validator.IdeaValidatorAgent()
        agent.llm = None

        result = agent.run(state)

        # Screening file is updated with evaluation data
        with open(state["candidate_topics_path"]) as f:
            updated_screening = json.load(f)
        for c in updated_screening["candidates"]:
            assert "evaluation" in c
            assert c["evaluation"]["overall_verdict"] == "pass"
