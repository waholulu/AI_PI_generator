import asyncio
import json

from agents import settings
from api import server


class _Run:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "awaiting_approval"
        self.thread_id = f"thread-{run_id}"


def _candidate_payload() -> dict:
    return {
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


def test_development_pack_preview_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-dev-pack"
    run_root = settings.run_root(run_id, create=True)

    output_file = run_root / "output" / "candidate_cards.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps({"candidates": [_candidate_payload()]}), encoding="utf-8")

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    asyncio.run(server.generate_development_pack(run_id, "beh_123"))
    summary = asyncio.run(server.get_development_pack(run_id, "beh_123"))

    assert summary["status"] == "ready"
    assert summary["claude_code_ready"] is True
    assert summary["checklist"]["has_implementation_spec"] is True
    assert summary["checklist"]["has_claude_task_prompt"] is True
    files = {f["filename"]: f for f in summary["files"]}
    assert "claude_task_prompt.md" in files
    assert files["claude_task_prompt.md"]["download_url"].endswith("/claude_task_prompt.md")
    assert isinstance(files["claude_task_prompt.md"]["preview_text"], str)
