import json
import os
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from agents.orchestrator import ResearchState
from agents import literature_agent
from agents import openalex_utils


# ---------------------------------------------------------------------------
# Shared fake paper and helpers
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


_PLAN_CONTENT = json.dumps({
    "run_id": "r1",
    "project_title": "AI and urban outcomes",
    "research_question": "How does AI adoption affect urban outcomes?",
    "short_rationale": "Practical policy relevance with public data.",
    "geography": "US",
    "time_window": "2010-2020",
    "exposure": {"name": "AI adoption", "measurement_proxy": "ai index"},
    "outcome": {"name": "Urban outcomes", "measurement_proxy": "outcome index"},
    "identification": {"primary_method": "fixed_effects", "key_threats": ["confounding"]},
    "data_sources": [{"name": "Source", "access_url": "https://example.org/source.csv", "expected_format": "csv"}],
    "literature_queries": [
        "ai adoption urban outcomes fixed effects",
        "urban ai policy outcomes",
        "ai urban panel data",
    ],
    "feasibility": {"overall_verdict": "warning"},
})


@pytest.fixture
def plan_path(tmp_path):
    """Write a minimal research plan into a temp dir and return its path."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    p = cfg / "research_plan.json"
    p.write_text(_PLAN_CONTENT, encoding="utf-8")
    (tmp_path / "data" / "literature").mkdir(parents=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    return str(p)


@pytest.fixture(autouse=True)
def _mock_arxiv():
    """Always stub out arXiv so tests never hit the network."""
    with patch("agents.literature_agent.multi_search_arxiv", return_value=[]):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_literature_node_offline(monkeypatch, plan_path, tmp_path):
    """Ensure literature node runs using a mocked OpenAlex search and download."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    paper = _fake_paper()

    def _fake_search(query: str, limit: int = 20, from_year: int | None = None, **kwargs):
        assert query
        return [paper]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    state = ResearchState(current_plan_path=plan_path, execution_status="harvesting")
    new_state = literature_agent.literature_node(state)

    assert new_state["execution_status"] == "drafting"
    index_path = tmp_path / "data" / "literature" / "index.json"
    assert index_path.exists()
    inventory = json.loads(index_path.read_text())
    assert len(inventory) >= 1
    assert inventory[0]["title"] == "Test Paper"
    assert inventory[0]["pdf_status"] == "pdf_downloaded"


def test_literature_multi_query_deduplication(monkeypatch, plan_path, tmp_path):
    """Multiple queries returning the same openalex_id should yield only one paper."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    shared_paper = _fake_paper(paper_id="W_SHARED", title="Shared Paper")

    monkeypatch.setattr(openalex_utils, "search_openalex", lambda *a, **kw: [shared_paper])
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    papers, query_hits = openalex_utils.multi_search_openalex(
        query_pool=["query alpha", "query beta", "query gamma"],
        per_query_limit=5,
        final_limit=5,
    )
    assert len(papers) == 1, f"Expected 1 unique paper after dedup, got {len(papers)}"


def test_literature_multi_query_collects_distinct_papers(monkeypatch, plan_path, tmp_path):
    """Distinct papers from different queries should all be collected."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(
        openalex_utils,
        "search_openalex",
        lambda query, **kw: [_fake_paper(paper_id=f"W_{query.replace(' ', '_')}", title=f"Paper for {query}")],
    )
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    papers, query_hits = openalex_utils.multi_search_openalex(
        query_pool=["query alpha", "query beta", "query gamma"],
        per_query_limit=5,
        final_limit=10,
    )
    assert len(papers) == 3
    assert sum(1 for h in query_hits.values() if h > 0) == 3


def test_literature_queries_used_written_to_context(monkeypatch, plan_path, tmp_path):
    """literature_summary in research_context.json should include queries_used list."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    context_path = tmp_path / "output" / "research_context.json"
    context_path.write_text(
        json.dumps({"domain": "GeoAI", "selected_topic": {"title": "GeoAI health"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(openalex_utils, "search_openalex", lambda *a, **kw: [_fake_paper()])
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    state = ResearchState(
        current_plan_path=plan_path,
        research_context_path=str(context_path),
        execution_status="harvesting",
    )
    literature_agent.literature_node(state)

    ctx = json.loads(context_path.read_text(encoding="utf-8"))
    assert "literature_summary" in ctx
    lit = ctx["literature_summary"]
    assert "queries_used" in lit
    assert isinstance(lit["queries_used"], list)
    assert "query_pool" in lit
    assert "used_fallback" in lit


def test_literature_fallback_when_planner_disabled(monkeypatch, plan_path, tmp_path):
    """With OPENALEX_QUERY_REWRITE_ENABLED=false, harvester must still run correctly."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENALEX_QUERY_REWRITE_ENABLED", "false")

    monkeypatch.setattr(openalex_utils, "search_openalex", lambda *a, **kw: [_fake_paper()])
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    state = ResearchState(current_plan_path=plan_path, execution_status="harvesting")
    new_state = literature_agent.literature_node(state)

    assert new_state["execution_status"] == "drafting"
    index_path = tmp_path / "data" / "literature" / "index.json"
    assert len(json.loads(index_path.read_text())) >= 1


def test_literature_cards_include_matched_query(monkeypatch, plan_path, tmp_path):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(openalex_utils, "search_openalex", lambda *a, **kw: [_fake_paper()])
    monkeypatch.setattr(literature_agent, "download_pdf", _fake_download)

    state = ResearchState(current_plan_path=plan_path, execution_status="harvesting")
    literature_agent.literature_node(state)

    index_path = tmp_path / "data" / "literature" / "index.json"
    inventory = json.loads(index_path.read_text())
    assert inventory
    assert "matched_query" in inventory[0]
    assert inventory[0]["matched_query"]
    assert "relevance_to_plan" in inventory[0]
    assert "source_database" in inventory[0]
