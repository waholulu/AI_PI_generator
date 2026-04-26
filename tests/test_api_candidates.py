import asyncio
import json

from agents import settings
from api import server


class _Run:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "awaiting_approval"
        self.thread_id = f"thread-{run_id}"


def _write_cards(run_id: str, payload: dict) -> None:
    run_root = settings.run_root(run_id, create=True)
    output_file = run_root / "output" / "candidate_cards.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload), encoding="utf-8")


def test_candidates_list_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-contract-cards"
    _write_cards(
        run_id,
        {
            "candidates": [
                {
                    "candidate_id": "beh_001",
                    "title": "Street Network Connectivity and Adult Physical Inactivity",
                    "research_question": "How does street connectivity affect inactivity?",
                    "exposure_label": "street_connectivity",
                    "exposure_source": "OSMnx_OpenStreetMap",
                    "outcome_label": "physical_inactivity",
                    "outcome_source": "CDC_PLACES",
                    "unit_of_analysis": "census_tract",
                    "method": "cross_sectional_spatial_association",
                    "claim_strength": "associational",
                    "technology_tags": ["osmnx", "geospatial"],
                    "required_secrets": [],
                    "automation_risk": "low",
                    "scores": {
                        "data_feasibility": 0.92,
                        "automation_feasibility": 0.88,
                        "identification_quality": 0.74,
                        "novelty": 0.67,
                        "technology_innovation": 0.80,
                        "overall": 0.82,
                    },
                    "gate_status": {"overall": "pass", "warnings": [], "failed_gates": []},
                    "development_pack_status": "ready",
                }
            ]
        },
    )

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    listed = asyncio.run(server.list_candidates(run_id))
    assert listed["run_id"] == run_id
    assert listed["count"] == 1
    card = listed["candidates"][0]
    assert card["candidate_id"] == "beh_001"
    assert card["shortlist_status"] == "ready"
    assert card["gate_summary"] == {"overall": "pass", "failed_count": 0, "warning_count": 0}


def test_candidate_detail_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-contract-detail"
    _write_cards(
        run_id,
        {
            "candidates": [
                {
                    "candidate_id": "beh_001",
                    "title": "Title",
                    "research_question": "RQ",
                    "exposure_family": "intersection_density",
                    "exposure_source": "OSMnx_OpenStreetMap",
                    "outcome_family": "physical_inactivity",
                    "outcome_source": "CDC_PLACES",
                    "unit_of_analysis": "tract",
                    "method_template": "cross_sectional_spatial_association",
                    "claim_strength": "associational",
                    "key_threats": ["confounding"],
                    "mitigations": {"confounding": "controls"},
                    "join_plan": {"steps": ["a", "b"], "join_key": "GEOID"},
                    "technology_tags": ["osmnx"],
                    "automation_risk": "low",
                    "cloud_safe": True,
                }
            ]
        },
    )

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    detail = asyncio.run(server.get_candidate(run_id, "beh_001"))
    candidate = detail["candidate"]
    assert candidate["candidate_id"] == "beh_001"
    assert candidate["join_plan"]["join_key"] == "GEOID"
    assert candidate["identification"]["key_threats"] == ["confounding"]
    assert candidate["technology"]["automation_risk"] == "low"
    assert len(candidate["x_y_structure"]) >= 2


def test_candidates_list_handles_missing_declared_sources(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-contract-sources"
    _write_cards(
        run_id,
        {
            "candidates": [
                {
                    "candidate_id": "cand_1",
                    "title": "Title",
                    "research_question": "RQ",
                    "exposure_family": "x",
                    "outcome_family": "y",
                    "declared_sources": ["OSMnx", None],
                    "unit_of_analysis": "tract",
                    "method_template": "cross_sectional_spatial_association",
                }
            ]
        },
    )
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))
    listed = asyncio.run(server.list_candidates(run_id))
    card = listed["candidates"][0]
    assert card["exposure_source"] == "OSMnx"
    assert card["outcome_source"] == ""
