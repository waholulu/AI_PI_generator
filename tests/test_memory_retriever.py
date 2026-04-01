import json
import os
from typing import Any, Dict, List

from agents.memory_retriever import MemoryRetriever


def _read_all_rows(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # first line is header
    header = lines[0].split(",")
    rows: List[Dict[str, Any]] = []
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split(",")
        row: Dict[str, Any] = {}
        for key, value in zip(header, parts):
            row[key] = value
        rows.append(row)
    return rows


def test_store_idea_creates_file_and_header(tmp_path) -> None:
    csv_path = tmp_path / "memory" / "idea_memory.csv"
    retriever = MemoryRetriever(csv_path=str(csv_path))

    retriever.store_idea(
        topic="Test Topic",
        domain="GeoAI",
        status="selected",
        metadata={"score": 90},
        rejection_reason="",
        source_file="output/topic_screening.json",
    )

    assert csv_path.exists()
    rows = _read_all_rows(str(csv_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["domain"] == "GeoAI"
    assert row["title_or_summary"] == "Test Topic"
    assert row["verdict"] == "selected"
    assert row["source_file"] == "output/topic_screening.json"

    metadata = json.loads(row["metadata_json"])
    assert metadata["score"] == 90
    assert row["created_at"]


def test_retrieve_domain_context_filters_and_limits(tmp_path) -> None:
    csv_path = tmp_path / "memory" / "idea_memory.csv"
    retriever = MemoryRetriever(csv_path=str(csv_path))

    # Two different domains, multiple rows
    retriever.store_idea("Topic A", "GeoAI and Urban Planning", "discarded")
    retriever.store_idea("Topic B", "Other Domain", "selected")
    retriever.store_idea("Topic C about GeoAI", "Misc", "discarded")

    history = retriever.retrieve_domain_context("GeoAI", limit=2)
    # Should only include rows whose domain or title mention GeoAI
    assert len(history) <= 2
    assert all("GeoAI" in item["topic"] or "GeoAI" in item.get("source_file", "") or "GeoAI" in "GeoAI and Urban Planning" for item in history)


def test_retrieve_empty_or_missing_file_returns_empty(tmp_path) -> None:
    csv_path = tmp_path / "memory" / "idea_memory.csv"
    retriever = MemoryRetriever(csv_path=str(csv_path))

    # File does not exist yet
    assert retriever.retrieve_domain_context("Any") == []

    # Create empty file
    os.makedirs(csv_path.parent, exist_ok=True)
    csv_path.write_text("", encoding="utf-8")
    assert retriever.retrieve_domain_context("Any") == []


def test_retrieve_handles_corrupt_metadata_gracefully(tmp_path) -> None:
    csv_path = tmp_path / "memory" / "idea_memory.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    csv_path.write_text(
        "domain,title_or_summary,verdict,reason,source_file,metadata_json,created_at\n"
        "GeoAI,Bad meta,discarded,reason,src,{not-json},2025-01-01T00:00:00\n",
        encoding="utf-8",
    )

    retriever = MemoryRetriever(csv_path=str(csv_path))
    history = retriever.retrieve_domain_context("GeoAI", limit=5)
    assert len(history) == 1
    assert history[0]["metadata"] == {}


def test_build_prompt_context_aggregates_csv_jsonl_and_graveyard(tmp_path) -> None:
    csv_path = tmp_path / "memory" / "idea_memory.csv"
    archive_path = tmp_path / "memory" / "enriched_top_candidates.jsonl"
    graveyard_path = tmp_path / "output" / "ideas_graveyard.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    graveyard_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = MemoryRetriever(csv_path=str(csv_path))
    retriever.store_idea(
        topic="GeoAI causal topic",
        domain="GeoAI and Urban Planning",
        status="selected",
        metadata={"score": 91},
        source_file="output/topic_screening.json",
    )

    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "created_at": "2025-01-01T00:00:00+00:00",
            "domain": "GeoAI and Urban Planning",
            "run_id": "r1",
            "rank": 1,
            "title": "GeoAI archival topic",
            "final_score": 93,
            "novelty_gap_type": "Measurement Gap",
            "quantitative_specs": {"outcomes": ["Y"]},
        }) + "\n")
        f.write(json.dumps({
            "created_at": "2025-01-01T00:00:00+00:00",
            "domain": "Other domain",
            "run_id": "r2",
            "rank": 1,
            "title": "Other topic",
        }) + "\n")

    with open(graveyard_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"title": "Rejected GeoAI idea", "rejection_reason": "No data"},
                {"title": "Irrelevant idea", "rejection_reason": "Off-topic"},
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    context = retriever.build_prompt_context(
        domain="GeoAI",
        enriched_jsonl_path=str(archive_path),
        graveyard_path=str(graveyard_path),
        recent_limit=5,
        archive_limit=5,
        rejected_limit=5,
    )

    assert context["summary"]["recent_count"] >= 1
    assert context["summary"]["archive_count"] == 1
    assert context["summary"]["rejected_count"] == 1
    assert context["recent_memory"][0]["topic"] == "GeoAI causal topic"
    assert context["enriched_archive"][0]["title"] == "GeoAI archival topic"
    assert context["rejected_history"][0]["title"] == "Rejected GeoAI idea"

