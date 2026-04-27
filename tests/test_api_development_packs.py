import asyncio
import json

import pytest
from fastapi import HTTPException

from agents import settings
from api import server


class _Run:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "awaiting_approval"
        self.thread_id = f"thread-{run_id}"


def _base_candidate(**overrides) -> dict:
    base = {
        "candidate_id": "beh_123",
        "template_id": "built_environment_health_v1",
        "exposure_family": "street_network",
        "exposure_source": "OSMnx_OpenStreetMap",
        "exposure_variables": ["intersection_density"],
        "outcome_family": "physical_inactivity",
        "outcome_source": "CDC_PLACES",
        "outcome_variables": ["physical_inactivity"],
        "unit_of_analysis": "census_tract",
        "join_plan": {"join_key": "GEOID"},
        "method_template": "cross_sectional_spatial_association",
        "claim_strength": "associational",
        "key_threats": ["confounding"],
        "mitigations": {"confounding": "controls"},
        "technology_tags": ["osmnx"],
        "required_secrets": [],
        "automation_risk": "low",
        "cloud_safe": True,
    }
    base.update(overrides)
    return base


def _setup_run(tmp_path, run_id: str, candidate: dict) -> None:
    run_root = settings.run_root(run_id, create=True)
    out = run_root / "output" / "candidate_cards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"candidates": [candidate]}), encoding="utf-8")


def test_development_pack_preview_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-dev-pack"
    candidate = _base_candidate()
    _setup_run(tmp_path, run_id, candidate)
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    asyncio.run(server.generate_development_pack(run_id, "beh_123"))
    summary = asyncio.run(server.get_development_pack(run_id, "beh_123"))

    assert summary["status"] == "claude_code_ready"
    assert summary["claude_code_ready"] is True

    checklist = summary["readiness_checklist"]
    assert checklist["implementation_spec"] is True
    assert checklist["claude_task_prompt"] is True
    assert checklist["data_contract"] is True
    assert checklist["feature_plan"] is True
    assert checklist["analysis_plan"] is True
    assert checklist["acceptance_tests"] is True
    assert checklist["no_required_secrets"] is True
    assert checklist["not_high_risk"] is True

    files = {f["filename"]: f for f in summary["files"]}
    assert "claude_task_prompt.md" in files
    assert files["claude_task_prompt.md"]["download_url"].endswith("/claude_task_prompt.md")
    assert isinstance(files["claude_task_prompt.md"]["preview_text"], str)
    assert "implementation_spec.json" in files


def test_development_pack_not_claude_code_ready_when_secrets_required(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-dev-pack-secrets"
    candidate = _base_candidate(
        candidate_id="beh_secret",
        required_secrets=["MAPILLARY_TOKEN"],
        automation_risk="medium",
    )
    _setup_run(tmp_path, run_id, candidate)
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    asyncio.run(server.generate_development_pack(run_id, "beh_secret"))
    summary = asyncio.run(server.get_development_pack(run_id, "beh_secret"))

    assert summary["readiness_checklist"]["no_required_secrets"] is False
    assert summary["claude_code_ready"] is False


def test_development_pack_not_claude_code_ready_when_high_risk(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-dev-pack-high-risk"
    candidate = _base_candidate(
        candidate_id="beh_highrisk",
        required_secrets=[],
        automation_risk="high",
    )
    _setup_run(tmp_path, run_id, candidate)
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    asyncio.run(server.generate_development_pack(run_id, "beh_highrisk"))
    summary = asyncio.run(server.get_development_pack(run_id, "beh_highrisk"))

    assert summary["readiness_checklist"]["not_high_risk"] is False
    assert summary["claude_code_ready"] is False


def test_development_pack_404_when_not_generated(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-dev-pack-missing"
    candidate = _base_candidate()
    _setup_run(tmp_path, run_id, candidate)
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.get_development_pack(run_id, "beh_123"))
    assert exc_info.value.status_code == 404


def test_development_pack_files_have_preview_and_download_url(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-dev-pack-files"
    candidate = _base_candidate()
    _setup_run(tmp_path, run_id, candidate)
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    asyncio.run(server.generate_development_pack(run_id, "beh_123"))
    summary = asyncio.run(server.get_development_pack(run_id, "beh_123"))

    for f in summary["files"]:
        assert "filename" in f
        assert "preview_text" in f
        assert "download_url" in f
        assert "file_type" in f
        assert f["download_url"].startswith(f"/runs/{run_id}/development-packs/")
        assert isinstance(f["preview_text"], str)
