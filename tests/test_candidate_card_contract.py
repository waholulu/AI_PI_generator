import asyncio
import json

from agents import settings
from api import server


class _Run:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "awaiting_approval"
        self.thread_id = f"thread-{run_id}"


_REQUIRED_CARD_KEYS = {
    "candidate_id", "title", "research_question",
    "exposure_label", "exposure_source", "outcome_label", "outcome_source",
    "unit_of_analysis", "method", "claim_strength",
    "technology_tags", "required_secrets", "automation_risk",
    "shortlist_status", "scores", "gate_summary", "development_pack_status",
}

_REQUIRED_SCORE_KEYS = {
    "data_feasibility", "automation_feasibility", "identification_quality",
    "novelty", "technology_innovation", "overall",
}


def test_candidate_card_contract_has_no_raw_blob(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-card-contract"
    run_root = settings.run_root(run_id, create=True)
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_001",
                "title": "Title",
                "research_question": "RQ",
                "exposure_family": "x",
                "outcome_family": "y",
                "exposure_source": "src_x",
                "outcome_source": "src_y",
                "method_template": "cross_sectional_spatial_association",
                "unit_of_analysis": "tract",
            }
        ]
    }
    out = run_root / "output" / "candidate_cards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))
    response = asyncio.run(server.list_candidates(run_id))
    card = response["candidates"][0]

    # Raw blob must be stripped
    assert "_raw" not in card

    # All required top-level keys must be present
    missing = _REQUIRED_CARD_KEYS - card.keys()
    assert not missing, f"Card missing keys: {missing}"

    # All score sub-keys must be present
    missing_scores = _REQUIRED_SCORE_KEYS - card["scores"].keys()
    assert not missing_scores, f"Card scores missing keys: {missing_scores}"

    # gate_summary must have the right structure
    gs = card["gate_summary"]
    assert "overall" in gs
    assert "failed_count" in gs
    assert "warning_count" in gs
    assert isinstance(gs["failed_count"], int)
    assert isinstance(gs["warning_count"], int)

    # Scores must be floats
    for k in _REQUIRED_SCORE_KEYS:
        assert isinstance(card["scores"][k], float), f"Score {k!r} must be float"

    # Lists must actually be lists
    assert isinstance(card["technology_tags"], list)
    assert isinstance(card["required_secrets"], list)


def test_candidate_card_contract_full_candidate(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-card-full"
    run_root = settings.run_root(run_id, create=True)
    payload = {
        "candidates": [
            {
                "candidate_id": "full_001",
                "title": "Full candidate",
                "research_question": "Does X affect Y?",
                "exposure_label": "x_label",
                "exposure_source": "EPA_AQS",
                "outcome_label": "y_label",
                "outcome_source": "CDC_PLACES",
                "unit_of_analysis": "county",
                "method": "diff_in_diff",
                "claim_strength": "causal",
                "technology_tags": ["api", "geospatial"],
                "required_secrets": ["EPA_API_KEY"],
                "automation_risk": "medium",
                "scores": {
                    "data_feasibility": 0.75,
                    "automation_feasibility": 0.65,
                    "identification_quality": 0.85,
                    "novelty": 0.70,
                    "technology_innovation": 0.60,
                    "overall": 0.71,
                },
                "gate_status": {"overall": "warning", "warnings": ["G5"], "failed_gates": []},
                "development_pack_status": "not_generated",
            }
        ]
    }
    out = run_root / "output" / "candidate_cards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(server.run_manager, "get_run", lambda _: _Run(run_id))
    response = asyncio.run(server.list_candidates(run_id))
    card = response["candidates"][0]

    assert card["candidate_id"] == "full_001"
    assert card["shortlist_status"] in ("ready", "review", "blocked")
    assert card["gate_summary"]["overall"] in ("pass", "warning", "fail")
    assert card["required_secrets"] == ["EPA_API_KEY"]
    assert card["automation_risk"] == "medium"
    assert abs(card["scores"]["overall"] - 0.71) < 0.01
    assert card["development_pack_status"] == "not_generated"
    assert "_raw" not in card
