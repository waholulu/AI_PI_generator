from __future__ import annotations

import json

import pytest

from autopi_engine import workflow
from autopi_engine.storage import input_path, run_root


def test_init_creates_manifest_under_data_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    manifest = workflow.init_run("Built environment and health", "demo_run")

    assert manifest.status == "awaiting_input"
    assert manifest.pending_action is not None
    assert manifest.pending_action.kind == "search_plan"
    assert (tmp_path / "runs" / "demo_run" / "manifest.json").exists()


def test_run_id_path_confinement(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    with pytest.raises(ValueError):
        run_root("..\\escape")

    with pytest.raises(ValueError):
        input_path("demo", "../bad")


def test_submit_search_plan_resumes_from_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    workflow.init_run("Built environment and health", "resume_run")
    payload = {
        "purpose": "field scan",
        "queries": ["walkability health census tract"],
        "per_query_limit": 3,
        "final_limit": 5,
    }
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(payload), encoding="utf-8")

    result = workflow.submit("resume_run", "search_plan", payload)
    loaded = workflow.status("resume_run")

    assert result.current_stage == "field_scan"
    assert loaded.status == "active"
    assert loaded.current_stage == "field_scan"
