"""Tests for candidate_output_writer: feasibility_report and development_pack_index."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.candidate_output_writer import (
    write_development_pack_index,
    write_feasibility_report,
    write_gate_trace,
)


def _make_card(
    candidate_id: str,
    shortlist: str = "ready",
    overall: str = "pass",
    risk: str = "low",
    claude_code_ready: bool = True,
    pack_files: list[str] | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "shortlist_status": shortlist,
        "automation_risk": risk,
        "claude_code_ready": claude_code_ready,
        "development_pack_status": "ready" if claude_code_ready else "blocked",
        "development_pack_files": pack_files or ["claude_task_prompt.md", "implementation_spec.json"],
        "gate_status": {
            "overall": overall,
            "shortlist_status": shortlist,
            "subchecks": {},
            "reasons": [],
        },
        "repair_history": [],
        "scores": {"overall": 0.8},
    }


@pytest.fixture()
def output_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_feasibility_report_structure(output_dir: Path) -> None:
    cards = [
        _make_card("c001", shortlist="ready", risk="low", claude_code_ready=True),
        _make_card("c002", shortlist="review", risk="medium", claude_code_ready=False),
        _make_card("c003", shortlist="blocked", overall="fail", risk="high", claude_code_ready=False),
    ]
    path = write_feasibility_report("run_test", cards)
    assert path.exists()
    report = json.loads(path.read_text())

    assert report["run_id"] == "run_test"
    assert report["candidate_count"] == 3
    assert report["summary"]["ready"] == 1
    assert report["summary"]["review"] == 1
    assert report["summary"]["blocked"] == 1
    assert report["summary"]["claude_code_ready"] == 1
    assert report["summary"]["low_risk"] == 1
    assert report["summary"]["medium_risk"] == 1
    assert report["summary"]["high_risk"] == 1
    assert len(report["candidates"]) == 3
    assert {c["candidate_id"] for c in report["candidates"]} == {"c001", "c002", "c003"}


def test_development_pack_index_structure(output_dir: Path) -> None:
    cards = [
        _make_card("c001", claude_code_ready=True),
        _make_card("c002", claude_code_ready=False),
    ]
    path = write_development_pack_index("run_test", cards)
    assert path.exists()
    index = json.loads(path.read_text())

    assert index["run_id"] == "run_test"
    assert len(index["packs"]) == 2
    ids = {p["candidate_id"] for p in index["packs"]}
    assert ids == {"c001", "c002"}

    ready_packs = [p for p in index["packs"] if p["claude_code_ready"]]
    assert len(ready_packs) == 1
    assert ready_packs[0]["candidate_id"] == "c001"


def test_gate_trace_structure(output_dir: Path) -> None:
    cards = [_make_card("c001"), _make_card("c002")]
    path = write_gate_trace("run_test", cards)
    assert path.exists()
    trace = json.loads(path.read_text())

    assert trace["run_id"] == "run_test"
    assert len(trace["candidates"]) == 2
    assert all("gate_status" in c for c in trace["candidates"])


def test_candidate_ids_consistent_across_outputs(output_dir: Path) -> None:
    """candidate_ids in feasibility_report and development_pack_index must match."""
    cards = [_make_card(f"c{i:03d}") for i in range(5)]
    fr_path = write_feasibility_report("run_test", cards)
    di_path = write_development_pack_index("run_test", cards)

    fr = json.loads(fr_path.read_text())
    di = json.loads(di_path.read_text())

    fr_ids = {c["candidate_id"] for c in fr["candidates"]}
    di_ids = {p["candidate_id"] for p in di["packs"]}
    assert fr_ids == di_ids, "candidate_ids must be identical across output files"
