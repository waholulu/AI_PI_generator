"""Tests for the idea_validator_agent module."""

import json
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from agents import idea_validator_agent as validator
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

class TestValidatorAllPass:
    """All 3 ideas pass: novel + data verified."""

    def test_all_pass(self, tmp_path, monkeypatch):
        # Mock search_openalex to return no papers → novel
        monkeypatch.setattr(validator, "search_openalex", lambda *a, **kw: [])
        # Mock registry
        monkeypatch.setattr(validator, "_load_registry", lambda: SAMPLE_REGISTRY)
        # Mock settings paths
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        state = _make_state(tmp_path)
        agent = validator.IdeaValidatorAgent()
        agent.llm = None  # Force fallback (no LLM → all novel)

        result = agent.run(state)

        assert os.path.exists(result["validation_report_path"])
        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        assert report["substitutions_made"] == 0
        for idea in report["validated_ideas"]:
            assert idea["overall_verdict"] in ("passed", "warning")
            assert idea["novelty"]["verdict"] == "novel"


class TestValidatorNoveltyFail:
    """One idea is already_published → auto-substitute from backup."""

    def test_novelty_fail_with_substitution(self, tmp_path, monkeypatch):
        # Make idea A fail novelty check via mocked search_openalex + LLM
        call_count = {"n": 0}

        def _mock_search(query, limit=20, from_year=None):
            call_count["n"] += 1
            # First idea's queries return highly similar papers
            if call_count["n"] <= 3:  # first idea's 3 queries
                return [{"title": "Exact Same Idea A", "year": 2025, "doi": "10.1234/test",
                          "citationCount": 50, "abstract": "Same approach same data."}]
            return []

        monkeypatch.setattr(validator, "search_openalex", _mock_search)
        monkeypatch.setattr(validator, "_load_registry", lambda: SAMPLE_REGISTRY)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        # Create a backup candidate
        backup = [_make_idea("Backup Idea D", rank=0, data_sources=[{"name": "World Bank"}])]
        state = _make_state(tmp_path, _make_screening(backup=backup))

        agent = validator.IdeaValidatorAgent()
        # Mock LLM to return "already_published" for first idea, "novel" for rest
        mock_llm = MagicMock()
        assess_call_count = {"n": 0}

        def _mock_assess(llm, title, rationale, papers):
            assess_call_count["n"] += 1
            if papers and assess_call_count["n"] == 1:
                return validator.NoveltyAssessment(
                    verdict="already_published",
                    similar_papers=[validator.SimilarPaper(
                        title="Exact Same Idea A",
                        year=2025,
                        doi="10.1234/test",
                        similarity_verdict="highly_similar",
                        overlap_explanation="Same method and data.",
                    )],
                )
            return validator.NoveltyAssessment(verdict="novel", similar_papers=[])

        monkeypatch.setattr(validator, "_assess_novelty", _mock_assess)
        agent.llm = None

        result = agent.run(state)

        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        assert report["substitutions_made"] == 1
        # The screening file should be updated
        with open(state["candidate_topics_path"]) as f:
            updated = json.load(f)
        assert updated["candidates"][0]["title"] == "Backup Idea D"


class TestValidatorDataFail:
    """All data sources unverified → idea fails."""

    def test_data_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validator, "search_openalex", lambda *a, **kw: [])
        monkeypatch.setattr(validator, "_load_registry", lambda: SAMPLE_REGISTRY)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        # All data sources are fake
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
        assert any("unverified" in r.lower() for r in failed[0]["failure_reasons"])


class TestValidatorNoBackups:
    """Backup pool is empty → failed idea stays, no substitution."""

    def test_no_backups(self, tmp_path, monkeypatch):
        call_count = {"n": 0}

        def _mock_search(query, limit=20, from_year=None):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                return [{"title": "Overlap Paper", "year": 2025, "doi": "10.x/y",
                          "citationCount": 10, "abstract": "Overlap."}]
            return []

        def _mock_assess(llm, title, rationale, papers):
            if papers:
                return validator.NoveltyAssessment(
                    verdict="already_published",
                    similar_papers=[],
                )
            return validator.NoveltyAssessment(verdict="novel", similar_papers=[])

        monkeypatch.setattr(validator, "search_openalex", _mock_search)
        monkeypatch.setattr(validator, "_assess_novelty", _mock_assess)
        monkeypatch.setattr(validator, "_load_registry", lambda: SAMPLE_REGISTRY)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        state = _make_state(tmp_path, _make_screening(backup=[]))  # no backups

        agent = validator.IdeaValidatorAgent()
        agent.llm = None
        result = agent.run(state)

        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        assert report["substitutions_made"] == 0


class TestValidatorOpenAlexDown:
    """OpenAlex raises exception → graceful degradation to warning."""

    def test_openalex_down(self, tmp_path, monkeypatch):
        def _failing_search(*a, **kw):
            raise ConnectionError("OpenAlex is down")

        monkeypatch.setattr(validator, "search_openalex", _failing_search)
        monkeypatch.setattr(validator, "_load_registry", lambda: SAMPLE_REGISTRY)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        state = _make_state(tmp_path)

        agent = validator.IdeaValidatorAgent()
        agent.llm = None
        result = agent.run(state)

        assert os.path.exists(result["validation_report_path"])
        with open(result["validation_report_path"]) as f:
            report = json.load(f)

        # Should not crash; all ideas get "novel" fallback
        for idea in report["validated_ideas"]:
            assert idea["novelty"]["verdict"] == "novel"


class TestValidatorMemoryWriteback:
    """Failed ideas are written back to memory."""

    def test_memory_writeback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validator, "search_openalex", lambda *a, **kw: [])
        monkeypatch.setattr(validator, "_load_registry", lambda: SAMPLE_REGISTRY)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        # Make an idea that fails data check
        bad_ideas = [
            _make_idea("Data Fail Idea", rank=1, data_sources=[
                {"name": "Nonexistent Source XYZ"},
                {"name": "Another Fake DB"},
            ]),
        ]
        state = _make_state(tmp_path, _make_screening(candidates=bad_ideas, backup=[]))

        agent = validator.IdeaValidatorAgent()
        agent.llm = None
        agent.memory = MagicMock()

        result = agent.run(state)

        # Verify store_idea was called for the failed idea
        assert agent.memory.store_idea.called
        call_args = agent.memory.store_idea.call_args
        assert "failed_data_check" in call_args.kwargs.get("status", "") or \
               "failed_data_check" in (call_args[1].get("status", "") if len(call_args) > 1 else "")


class TestValidatorSubstitutionUpdatesFiles:
    """Verify that substitution updates topic_screening.json and research_plan.json."""

    def test_substitution_updates_files(self, tmp_path, monkeypatch):
        call_count = {"n": 0}

        def _mock_search(query, limit=20, from_year=None):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                return [{"title": "Existing Work", "year": 2025}]
            return []

        def _mock_assess(llm, title, rationale, papers):
            if papers:
                return validator.NoveltyAssessment(
                    verdict="already_published", similar_papers=[]
                )
            return validator.NoveltyAssessment(verdict="novel", similar_papers=[])

        monkeypatch.setattr(validator, "search_openalex", _mock_search)
        monkeypatch.setattr(validator, "_assess_novelty", _mock_assess)
        monkeypatch.setattr(validator, "_load_registry", lambda: SAMPLE_REGISTRY)
        monkeypatch.setattr(settings, "idea_validation_path", lambda: str(tmp_path / "idea_validation.json"))

        backup = [_make_idea("Substitute Idea", rank=0, data_sources=[{"name": "World Bank"}])]
        state = _make_state(tmp_path, _make_screening(backup=backup))

        agent = validator.IdeaValidatorAgent()
        agent.llm = None
        agent.memory = MagicMock()

        result = agent.run(state)

        # Check screening was updated
        with open(state["candidate_topics_path"]) as f:
            updated_screening = json.load(f)
        assert updated_screening["candidates"][0]["title"] == "Substitute Idea"

        # Check research plan was updated
        with open(state["current_plan_path"]) as f:
            updated_plan = json.load(f)
        assert updated_plan["project_title"] == "Substitute Idea"
