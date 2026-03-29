import json
import os
from typing import Any, Dict

from agents.field_scanner_agent import FieldScannerAgent, field_scanner_node
from agents.memory_retriever import MemoryRetriever
from agents.orchestrator import ResearchState


def _write_dummy_memory_csv(path: str, domain: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "domain,title_or_summary,verdict,reason,source_file,metadata_json,created_at\n"
        )
        f.write(
            f"{domain},Test topic,discarded,Data not available,output/ideas_graveyard.json,{{}},2025-01-01T00:00:00\n"
        )


class DummyFieldScanner(FieldScannerAgent):
    """
    A thin subclass that overrides _search_openalex to avoid real network calls.
    """

    def _search_openalex(self, query: str, limit: int = 5) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "organic_results": [
                {
                    "title": "Paper A",
                    "citations": 10,
                    "year": 2022,
                    "author": "Author A",
                    "authors": ["Author A"],
                    "concepts": ["GeoAI", "Urban Planning"],
                    "concept_details": [],
                    "openalex_id": "https://openalex.org/W1",
                    "doi": "10.1234/a",
                    "type": "article",
                    "journal": "Journal A",
                    "publication_date": "2022-01-01",
                    "is_open_access": True,
                    "oa_status": "gold",
                    "oa_url": "",
                    "landing_page_url": "",
                    "pdf_url": "",
                    "candidate_download_urls": [],
                    "referenced_works_count": 5,
                    "is_retracted": False,
                },
                {
                    "title": "Paper B",
                    "citations": 5,
                    "year": 2021,
                    "author": "Author B",
                    "authors": ["Author B"],
                    "concepts": ["GeoAI"],
                    "concept_details": [],
                    "openalex_id": "https://openalex.org/W2",
                    "doi": "10.1234/b",
                    "type": "article",
                    "journal": "Journal B",
                    "publication_date": "2021-01-01",
                    "is_open_access": True,
                    "oa_status": "gold",
                    "oa_url": "",
                    "landing_page_url": "",
                    "pdf_url": "",
                    "candidate_download_urls": [],
                    "referenced_works_count": 3,
                    "is_retracted": False,
                },
            ]
        }


def test_memory_and_field_scan_with_history(tmp_path) -> None:
    """
    正常路径：存在历史 CSV 记忆，field_scan 使用伪造的 OpenAlex 结果。
    """
    # 准备内存 CSV
    csv_path = tmp_path / "memory" / "idea_memory.csv"
    _write_dummy_memory_csv(str(csv_path), "GeoAI and Urban Planning")

    retriever = MemoryRetriever(csv_path=str(csv_path))
    history = retriever.retrieve_domain_context("GeoAI and Urban Planning")
    assert history, "Expected to retrieve at least one history row from CSV memory"

    # 运行 field scanner（使用 dummy 实现避免真实网络调用）
    state: ResearchState = ResearchState(
        domain_input="GeoAI and Urban Planning", execution_status="starting"
    )
    agent = DummyFieldScanner()
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


def test_memory_and_field_scan_empty_graceful(tmp_path) -> None:
    """
    降级路径：没有历史 CSV 记忆，field_scan 返回空结果也不应报错。
    """

    class EmptyFieldScanner(FieldScannerAgent):
        def _search_openalex(  # type: ignore[override]
            self, query: str, limit: int = 5
        ) -> Dict[str, Any]:
            return {"organic_results": [], "total_results": 0}

    # 不创建任何 CSV 文件
    retriever = MemoryRetriever(csv_path=str(tmp_path / "memory" / "idea_memory.csv"))
    history = retriever.retrieve_domain_context("GeoAI and Urban Planning")
    assert history == []

    state: ResearchState = ResearchState(
        domain_input="GeoAI and Urban Planning", execution_status="starting"
    )
    agent = EmptyFieldScanner()
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
