import json
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agents.orchestrator import ResearchState
from agents import literature_agent
from agents import openalex_utils


# ---------------------------------------------------------------------------
# Shared fake paper
# ---------------------------------------------------------------------------

def _fake_paper(paper_id: str = "paper1", title: str = "Test Paper") -> Dict[str, Any]:
    return {
        "paperId": paper_id,
        "openalex_id": f"https://openalex.org/{paper_id}",
        "title": title,
        "authors": [{"name": "Author One"}],
        "year": 2024,
        "publication_date": "2024-01-01",
        "type": "article",
        "language": "en",
        "doi": f"10.1234/{paper_id}",
        "journal": "Journal Test",
        "journal_id": "J1",
        "host_organization": "Org",
        "abstract": "Abstract",
        "citationCount": 10,
        "isOpenAccess": True,
        "oa_status": "gold",
        "oa_url": "",
        "landing_page_url": "",
        "pdf_url": "",
        "candidate_download_urls": ["https://example.com/paper.pdf"],
        "biblio": {},
        "concepts": [],
        "referenced_works": [],
        "referenced_works_count": 0,
        "is_retracted": False,
    }


def _fake_download(url: str, dest: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write("PDF bytes here")
    return {"status": "pdf_downloaded", "path": dest, "reason": ""}


def _setup_dirs_and_plan(plan_content: str = '{"keywords": ["artificial intelligence", "urban"]}') -> None:
    os.makedirs("config", exist_ok=True)
    os.makedirs("data/literature", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    with open("config/research_plan.json", "w", encoding="utf-8") as f:
        f.write(plan_content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_literature_node_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure literature node runs using a mocked OpenAlex search and download."""
    _setup_dirs_and_plan()

    paper = _fake_paper()

    def _fake_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
        assert query  # ensure query was built
        return [paper]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    state = ResearchState(
        current_plan_path="config/research_plan.json",
        execution_status="harvesting",
    )
    new_state = literature_agent.literature_node(state)

    assert new_state["execution_status"] == "drafting"
    assert os.path.exists("data/literature/index.json")
    assert os.path.exists("output/references.bib")

    with open("data/literature/index.json", "r", encoding="utf-8") as f:
        inventory = json.load(f)
    assert len(inventory) >= 1
    assert inventory[0]["title"] == "Test Paper"
    assert inventory[0]["pdf_status"] == "pdf_downloaded"


def test_literature_multi_query_deduplication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple queries returning the same openalex_id should yield only one paper."""
    _setup_dirs_and_plan()

    call_count = {"n": 0}
    shared_paper = _fake_paper(paper_id="W_SHARED", title="Shared Paper")

    def _fake_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
        call_count["n"] += 1
        return [shared_paper]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    papers, query_hits = openalex_utils.multi_search_openalex(
        query_pool=["query alpha", "query beta", "query gamma"],
        per_query_limit=5,
        final_limit=5,
    )

    ids = [p.get("openalex_id") or p.get("paperId") for p in papers]
    assert len(papers) == 1, f"Expected 1 unique paper after dedup, got {len(papers)}"
    assert call_count["n"] >= 1


def test_literature_multi_query_collects_distinct_papers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct papers from different queries should all be collected."""
    _setup_dirs_and_plan()

    def _fake_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [_fake_paper(paper_id=f"W_{query.replace(' ', '_')}", title=f"Paper for {query}")]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    papers, query_hits = openalex_utils.multi_search_openalex(
        query_pool=["query alpha", "query beta", "query gamma"],
        per_query_limit=5,
        final_limit=10,
    )

    titles = [p["title"] for p in papers]
    assert len(papers) == 3
    queries_with_hits = [q for q, h in query_hits.items() if h > 0]
    assert len(queries_with_hits) == 3


def test_literature_queries_used_written_to_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """literature_summary in research_context.json should include queries_used list."""
    _setup_dirs_and_plan()

    # Write a minimal research context
    context_path = "output/research_context.json"
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump({"domain": "GeoAI", "selected_topic": {"title": "GeoAI health"}}, f)

    paper = _fake_paper()

    def _fake_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [paper]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    state = ResearchState(
        current_plan_path="config/research_plan.json",
        research_context_path=context_path,
        execution_status="harvesting",
    )
    literature_agent.literature_node(state)

    with open(context_path, "r", encoding="utf-8") as f:
        ctx = json.load(f)

    assert "literature_summary" in ctx
    lit_summary = ctx["literature_summary"]
    assert "queries_used" in lit_summary
    assert isinstance(lit_summary["queries_used"], list)
    assert "query_pool" in lit_summary
    assert "used_fallback" in lit_summary


def test_literature_fallback_when_planner_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With OPENALEX_QUERY_REWRITE_ENABLED=false, harvester must still run correctly."""
    monkeypatch.setenv("OPENALEX_QUERY_REWRITE_ENABLED", "false")
    _setup_dirs_and_plan()

    paper = _fake_paper()

    def _fake_search(query: str, limit: int = 20) -> List[Dict[str, Any]]:
        return [paper]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    state = ResearchState(
        current_plan_path="config/research_plan.json",
        execution_status="harvesting",
    )
    new_state = literature_agent.literature_node(state)

    assert new_state["execution_status"] == "drafting"
    with open("data/literature/index.json", "r", encoding="utf-8") as f:
        inventory = json.load(f)
    assert len(inventory) >= 1
