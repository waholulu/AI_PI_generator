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


_FULL_CANDIDATE = {
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


def test_candidates_list_contract(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-contract-cards"
    _write_cards(run_id, {"candidates": [_FULL_CANDIDATE]})
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    listed = asyncio.run(server.list_candidates(run_id))
    assert listed["run_id"] == run_id
    assert listed["count"] == 1
    card = listed["candidates"][0]

    # Core identity
    assert card["candidate_id"] == "beh_001"
    assert card["title"] == "Street Network Connectivity and Adult Physical Inactivity"
    assert "research_question" in card

    # X/Y structure
    assert card["exposure_label"] == "street_connectivity"
    assert card["exposure_source"] == "OSMnx_OpenStreetMap"
    assert card["outcome_label"] == "physical_inactivity"
    assert card["outcome_source"] == "CDC_PLACES"
    assert card["unit_of_analysis"] == "census_tract"

    # Method and claim
    assert card["method"] == "cross_sectional_spatial_association"
    assert card["claim_strength"] == "associational"

    # Technology
    assert card["technology_tags"] == ["osmnx", "geospatial"]
    assert card["required_secrets"] == []
    assert card["automation_risk"] == "low"

    # Shortlist and gate summary
    assert card["shortlist_status"] == "ready"
    assert card["gate_summary"] == {"overall": "pass", "failed_count": 0, "warning_count": 0}

    # All six score dimensions
    scores = card["scores"]
    assert isinstance(scores["data_feasibility"], float)
    assert isinstance(scores["automation_feasibility"], float)
    assert isinstance(scores["identification_quality"], float)
    assert isinstance(scores["novelty"], float)
    assert isinstance(scores["technology_innovation"], float)
    assert isinstance(scores["overall"], float)
    assert abs(scores["overall"] - 0.82) < 0.01

    # Dev pack status
    assert card["development_pack_status"] == "ready"

    # Raw blob must be stripped
    assert "_raw" not in card


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
                    "key_threats": ["confounding", "self_selection"],
                    "mitigations": {"confounding": "Add ACS controls", "self_selection": "County FE"},
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
    assert candidate["join_plan"]["steps"] == ["a", "b"]

    ident = candidate["identification"]
    assert ident["key_threats"] == ["confounding", "self_selection"]
    assert "confounding" in ident["mitigations"]
    assert ident["claim_strength"] == "associational"

    tech = candidate["technology"]
    assert tech["automation_risk"] == "low"
    assert tech["cloud_safe"] is True
    assert isinstance(tech["required_secrets"], list)
    assert isinstance(tech["policy_constraints"], list)

    assert len(candidate["x_y_structure"]) >= 2
    roles = {row["role"] for row in candidate["x_y_structure"]}
    assert "exposure" in roles
    assert "outcome" in roles

    assert isinstance(candidate["gate_status"], dict)
    assert isinstance(candidate["repair_history"], list)


def test_candidate_detail_404_on_unknown_id(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-contract-detail-404"
    _write_cards(run_id, {"candidates": [_FULL_CANDIDATE]})
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.get_candidate(run_id, "nonexistent_id"))
    assert exc_info.value.status_code == 404


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


def test_candidates_list_returns_empty_for_no_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-no-files"
    # Create run root but no candidate_cards.json or topic_screening.json
    settings.run_root(run_id, create=True)
    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))

    listed = asyncio.run(server.list_candidates(run_id))
    assert listed["count"] == 0
    assert listed["candidates"] == []
