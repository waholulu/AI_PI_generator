import json
import os
from typing import Any, Dict, List

import pytest

from agents.field_scanner_agent import FieldScannerAgent, field_scanner_node
from agents.memory_retriever import MemoryRetriever
from agents.orchestrator import ResearchState
from agents import openalex_utils


def _write_dummy_memory_csv(path: str, domain: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "domain,title_or_summary,verdict,reason,source_file,metadata_json,created_at\n"
        )
        f.write(
            f"{domain},Test topic,discarded,Data not available,output/ideas_graveyard.json,{{}},2025-01-01T00:00:00\n"
        )


def _make_fake_metadata(openalex_id: str, title: str, concepts: List[str] | None = None) -> Dict[str, Any]:
    concept_names = concepts or ["GeoAI", "Remote Sensing"]
    return {
        "title": title,
        "openalex_id": openalex_id,
        "paperId": openalex_id.replace("https://openalex.org/", ""),
        "doi": "10.1234/a",
        "year": 2022,
        "publication_date": "2022-01-01",
        "type": "article",
        "language": "en",
        "citationCount": 10,
        "authors": [{"name": "Author A"}],
        "author_names": ["Author A"],
        "first_author": "Author A",
        "journal": "Journal A",
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
        "referenced_works_count": 5,
        "is_retracted": False,
    }


def test_memory_and_field_scan_with_history(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    正常路径：存在历史 CSV 记忆，field_scan 使用伪造的 OpenAlex 结果。
    """
    # 准备内存 CSV
    csv_path = tmp_path / "memory" / "idea_memory.csv"
    _write_dummy_memory_csv(str(csv_path), "GeoAI and Urban Planning")

    retriever = MemoryRetriever(csv_path=str(csv_path))
    history = retriever.retrieve_domain_context("GeoAI and Urban Planning")
    assert history, "Expected to retrieve at least one history row from CSV memory"

    # Patch OpenAlex search to return fake data
    def _fake_search(query: str, limit: int = 20, from_year: int | None = None, **kwargs) -> List[Dict[str, Any]]:
        return [
            _make_fake_metadata("https://openalex.org/W1", "Paper A", ["GeoAI", "Urban Planning"]),
            _make_fake_metadata("https://openalex.org/W2", "Paper B", ["GeoAI"]),
        ]

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)

    # 运行 field scanner
    state: ResearchState = ResearchState(
        domain_input="GeoAI and Urban Planning", execution_status="starting"
    )
    agent = FieldScannerAgent()
    new_state = agent.run(state)

    assert new_state["execution_status"] == "ideation"
    assert "field_scan_path" in new_state

    field_scan_path = new_state["field_scan_path"]
    assert os.path.exists(field_scan_path)

    with open(field_scan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["domain_scanned"] == "GeoAI and Urban Planning"
    assert "openalex_traction" in data
    assert "top_results" in data["openalex_traction"]
    assert "keywords" in data
    assert data["keywords"]["raw_query"] == "GeoAI and Urban Planning"
    # 至少 GeoAI 概念应该出现在高牵引关键词列表中
    assert "high_traction" in data["keywords"]
    assert isinstance(data["keywords"]["high_traction"], list)
    assert "GeoAI" in data["keywords"]["high_traction"]

    # 元信息中应标记扫描状态
    assert "meta" in data
    assert data["meta"]["scan_status"] in ("full", "partial")


def test_memory_and_field_scan_empty_graceful(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    降级路径：没有历史 CSV 记忆，field_scan 返回空结果也不应报错。
    """
    def _fake_search(query: str, limit: int = 20, from_year: int | None = None, **kwargs) -> List[Dict[str, Any]]:
        return []

    monkeypatch.setattr(openalex_utils, "search_openalex", _fake_search)

    # 不创建任何 CSV 文件
    retriever = MemoryRetriever(csv_path=str(tmp_path / "memory" / "idea_memory.csv"))
    history = retriever.retrieve_domain_context("GeoAI and Urban Planning")
    assert history == []

    state: ResearchState = ResearchState(
        domain_input="GeoAI and Urban Planning", execution_status="starting"
    )
    agent = FieldScannerAgent()
    new_state = agent.run(state)

    field_scan_path = new_state["field_scan_path"]
    assert os.path.exists(field_scan_path)

    with open(field_scan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["domain_scanned"] == "GeoAI and Urban Planning"
    assert "openalex_traction" in data
    assert isinstance(data["openalex_traction"].get("top_results"), list)

    assert "keywords" in data
    assert data["keywords"]["raw_query"] == "GeoAI and Urban Planning"
    assert isinstance(data["keywords"]["high_traction"], list)

    assert "meta" in data
    assert data["meta"]["scan_status"] in ("empty", "partial", "full")

    # 即使一切为空，也不应抛出异常或缺字段
