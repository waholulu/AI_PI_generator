import json
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agents.field_scanner_agent import FieldScannerAgent, field_scanner_node, summarize_field_scan
from agents.orchestrator import ResearchState
from agents import openalex_utils


# ---------------------------------------------------------------------------
# Offline stub helpers
# ---------------------------------------------------------------------------

def _make_fake_result(openalex_id: str, title: str, concepts: List[str] | None = None) -> Dict[str, Any]:
    return {
        "title": title,
        "citations": 20,
        "year": 2023,
        "author": "Author X",
        "authors": ["Author X"],
        "concepts": concepts or ["GeoAI", "Remote Sensing"],
        "concept_details": [],
        "openalex_id": openalex_id,
        "doi": f"10.1234/{openalex_id[-2:]}",
        "type": "article",
        "journal": "Journal X",
        "publication_date": "2023-01-01",
        "is_open_access": True,
        "oa_status": "gold",
        "oa_url": "",
        "landing_page_url": "",
        "pdf_url": "",
        "candidate_download_urls": [],
        "referenced_works_count": 8,
        "is_retracted": False,
    }


def _make_fake_metadata(openalex_id: str, title: str, concepts: List[str] | None = None) -> Dict[str, Any]:
    """Return a dict matching extract_work_metadata output format."""
    concept_names = concepts or ["GeoAI", "Remote Sensing"]
    return {
        "title": title,
        "openalex_id": openalex_id,
        "paperId": openalex_id.replace("https://openalex.org/", ""),
        "doi": f"10.1234/{openalex_id[-2:]}",
        "year": 2023,
        "publication_date": "2023-01-01",
        "type": "article",
        "language": "en",
        "citationCount": 20,
        "authors": [{"name": "Author X"}],
        "author_names": ["Author X"],
        "first_author": "Author X",
        "journal": "Journal X",
        "journal_id": "J1",
        "host_organization": "Org",
        "biblio": {},
        "abstract": "",
        "concepts": [{"name": c, "level": 0, "score": 0.9} for c in concept_names],
        "broad_concepts": [{"name": c, "level": 0, "score": 0.9} for c in concept_names],
        "isOpenAccess": True,
        "oa_status": "gold",
        "oa_url": "",
        "best_oa_location": {},
        "landing_page_url": "",
        "pdf_url": "",
        "candidate_download_urls": [],
        "has_fulltext": False,
        "referenced_works": [],
        "referenced_works_count": 8,
        "is_retracted": False,
    }


# ---------------------------------------------------------------------------
# Fixtures for planner mock
# ---------------------------------------------------------------------------

def _patch_planner(scanner: FieldScannerAgent, queries: List[str], used_fallback: bool = False) -> None:
    mock_plan = {
        "query_pool": queries,
        "topics": [{"label": q, "queries": [q]} for q in queries],
        "primary_domains": ["Test Domain"],
        "methods": [],
        "used_fallback": used_fallback,
    }
    scanner._planner = MagicMock()
    scanner._planner.plan.return_value = mock_plan


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_field_scanner_node_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure field scanner node runs offline and produces a valid field_scan.json."""
    paper = _make_fake_metadata("https://openalex.org/W1234", "Test Paper")

    def _fake_search(query: str, limit: int = 20, from_year: int | None = None, **kwargs) -> List[Dict[str, Any]]:
        return [paper]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)

    state = ResearchState(domain_input="GeoAI and Urban Planning", execution_status="starting")
    new_state = field_scanner_node(state)

    assert new_state["execution_status"] == "ideation"
    assert "field_scan_path" in new_state
    assert os.path.exists("output/field_scan.json")

    with open("output/field_scan.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["domain_scanned"] == "GeoAI and Urban Planning"
    assert "openalex_traction" in data
    assert "top_results" in data["openalex_traction"]
    assert isinstance(data["openalex_traction"]["top_results"], list)

    for result in data["openalex_traction"]["top_results"]:
        assert "title" in result
        assert "citations" in result
        assert "year" in result
        assert "author" in result


def test_field_scanner_multi_query_expands_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple queries should yield more (or equal) results than a single query."""
    def _fake_search(query: str, limit: int = 20, from_year: int | None = None, **kwargs) -> List[Dict[str, Any]]:
        return [_make_fake_metadata(
            openalex_id=f"https://openalex.org/W{abs(hash(query)) % 1000}",
            title=f"Paper for '{query}'",
        )]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)

    scanner = FieldScannerAgent()
    _patch_planner(scanner, queries=["query alpha", "query beta", "query gamma"])

    state = {"domain_input": "broad multi-topic domain", "execution_status": "starting"}
    scanner.run(state)

    with open("output/field_scan.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    top_results = data["openalex_traction"]["top_results"]
    # Each distinct query should contribute at least one unique paper
    assert len(top_results) >= 3, "Expected at least one result per unique query"

    strategy = data["search_strategy"]
    assert strategy["queries_executed"] == 3
    assert set(strategy["query_pool"]) == {"query alpha", "query beta", "query gamma"}


def test_field_scanner_deduplicates_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """If multiple queries return the same openalex_id, only one copy should appear."""
    def _fake_search(query: str, limit: int = 20, from_year: int | None = None, **kwargs) -> List[Dict[str, Any]]:
        return [_make_fake_metadata(
            openalex_id="https://openalex.org/W_FIXED",
            title="Always the same paper",
        )]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)

    scanner = FieldScannerAgent()
    _patch_planner(scanner, queries=["q1", "q2", "q3"])

    state = {"domain_input": "duplicate test domain", "execution_status": "starting"}
    scanner.run(state)

    with open("output/field_scan.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    top_results = data["openalex_traction"]["top_results"]
    ids = [r["openalex_id"] for r in top_results]
    assert len(ids) == len(set(ids)), "Duplicate openalex_ids found in merged results"
    assert len(top_results) == 1, "Expected exactly 1 unique paper after dedup"


def test_field_scan_includes_search_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """field_scan.json must contain the search_strategy block with expected keys."""
    def _fake_search(query: str, limit: int = 20, from_year: int | None = None, **kwargs) -> List[Dict[str, Any]]:
        return [_make_fake_metadata(
            openalex_id=f"https://openalex.org/W{abs(hash(query)) % 1000}",
            title=f"Paper for '{query}'",
        )]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)

    scanner = FieldScannerAgent()
    _patch_planner(scanner, queries=["built environment health", "walkability"], used_fallback=False)

    state = {"domain_input": "public health urban", "execution_status": "starting"}
    scanner.run(state)

    with open("output/field_scan.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "search_strategy" in data
    strategy = data["search_strategy"]
    assert "query_pool" in strategy
    assert "queries_executed" in strategy
    assert "used_fallback" in strategy
    assert strategy["used_fallback"] is False

    assert "query_hits" in data
    for q in ["built environment health", "walkability"]:
        assert q in data["query_hits"]


def test_field_scanner_fallback_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """When planner returns used_fallback=True the scanner still runs correctly."""
    def _fake_search(query: str, limit: int = 20, from_year: int | None = None, **kwargs) -> List[Dict[str, Any]]:
        return [_make_fake_metadata(
            openalex_id="https://openalex.org/W_FB",
            title="Fallback Paper",
        )]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)

    scanner = FieldScannerAgent()
    _patch_planner(scanner, queries=["my raw domain"], used_fallback=True)

    state = {"domain_input": "my raw domain", "execution_status": "starting"}
    result = scanner.run(state)

    assert result["execution_status"] == "ideation"

    with open("output/field_scan.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["search_strategy"]["used_fallback"] is True


def test_summarize_field_scan_structure() -> None:
    """summarize_field_scan must produce all expected summary keys."""
    synthetic_scan: Dict[str, Any] = {
        "domain_scanned": "GeoAI and Urban Planning",
        "meta": {"scan_status": "full"},
        "search_strategy": {
            "query_pool": ["geoai urban planning", "remote sensing city"],
            "queries_executed": 2,
            "used_fallback": False,
            "primary_domains": ["Urban Science"],
            "methods": [],
            "topics": [],
        },
        "query_hits": {"geoai urban planning": 3, "remote sensing city": 2},
        "openalex_traction": {
            "top_results": [
                {
                    "title": "Paper A",
                    "year": 2022,
                    "citations": 30,
                    "concepts": ["GeoAI", "Urban Planning"],
                    "doi": "10.1234/pa",
                },
                {
                    "title": "Paper B",
                    "year": 2023,
                    "citations": 10,
                    "concepts": ["GeoAI"],
                    "doi": "10.1234/pb",
                },
            ]
        },
        "keywords": {"raw_query": "GeoAI and Urban Planning", "high_traction": ["GeoAI", "Urban Planning"]},
    }

    summary = summarize_field_scan(synthetic_scan)

    assert "high_traction_keywords" in summary
    assert "GeoAI" in summary["high_traction_keywords"]

    assert "top_papers" in summary
    assert len(summary["top_papers"]) == 2
    assert summary["top_papers"][0]["title"] == "Paper A"

    assert "year_distribution" in summary
    assert "2022" in summary["year_distribution"] or "2023" in summary["year_distribution"]

    assert "crowded_concepts" in summary
    assert "scan_status" in summary
    assert summary["scan_status"] == "full"
    assert summary["domain_scanned"] == "GeoAI and Urban Planning"

    # New: search_strategy summary should be present
    assert "search_strategy" in summary
    assert summary["search_strategy"]["queries_executed"] == 2
    assert summary["search_strategy"]["used_fallback"] is False
