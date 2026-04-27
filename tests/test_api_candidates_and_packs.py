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


def test_select_candidate_by_id_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-api-select"
    run_root = settings.run_root(run_id, create=True)
    output_file = run_root / "output" / "topic_screening.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "candidates": [
                    {"title": "A", "candidate_id": "cand_a", "rank": 1},
                    {"title": "B", "candidate_id": "cand_b", "rank": 2},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))
    monkeypatch.setattr(server.run_manager, "record_milestone", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_resume_pipeline", lambda *args, **kwargs: asyncio.sleep(0))

    response = asyncio.run(server.select_candidate_by_id(run_id, "cand_b"))
    assert response.status == "running"
    assert response.selected_idea == "B"


def test_select_candidate_by_id_endpoint_supports_legacy_candidate_id(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-api-select-legacy"
    run_root = settings.run_root(run_id, create=True)
    output_file = run_root / "output" / "topic_screening.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "candidates": [
                    {"title": "A", "topic_id": "topic_a", "rank": 1},
                    {"title": "B", "topic_id": "topic_b", "rank": 2},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))
    monkeypatch.setattr(server.run_manager, "record_milestone", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_resume_pipeline", lambda *args, **kwargs: asyncio.sleep(0))

    response = asyncio.run(server.select_candidate_by_id(run_id, "legacy_002"))
    assert response.status == "running"
    assert response.selected_idea == "B"
