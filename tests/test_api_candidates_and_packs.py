import asyncio
import json

from agents import settings
from api import server


class _Run:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "awaiting_approval"


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
        "key_threats": ["confounding"],
        "mitigations": {"confounding": "controls"},
        "technology_tags": ["osmnx"],
        "automation_risk": "low",
    }


def test_candidate_and_development_pack_endpoints(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-api-1"
    run_root = settings.run_root(run_id, create=True)

    candidate_cards = {"candidates": [_candidate_payload()]}
    output_file = run_root / "output" / "candidate_cards.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(candidate_cards), encoding="utf-8")

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    listed = asyncio.run(server.list_candidates(run_id))
    assert listed["candidates"][0]["candidate_id"] == "beh_123"

    generated = asyncio.run(server.generate_development_pack(run_id, "beh_123"))
    assert generated["candidate_id"] == "beh_123"
    assert "implementation_spec.json" in generated["files"]

    packs = asyncio.run(server.list_development_packs(run_id))
    assert packs["development_packs"][0]["candidate_id"] == "beh_123"

    files = asyncio.run(server.list_development_pack_files(run_id, "beh_123"))
    names = [f["filename"] for f in files["files"]]
    assert "claude_task_prompt.md" in names
