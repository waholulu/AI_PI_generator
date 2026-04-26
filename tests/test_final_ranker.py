from agents.final_ranker import rank_candidates, score_candidate


def _base_candidate(**kwargs):
    data = {
        "automation_risk": "low",
        "required_secrets": [],
        "cloud_safe": True,
        "key_threats": ["a", "b", "c", "d"],
        "mitigations": {"a": "1", "b": "2", "c": "3", "d": "4"},
        "technology_tags": ["osmnx"],
    }
    data.update(kwargs)
    return data


def test_score_candidate_populates_overall_score() -> None:
    scores = score_candidate(_base_candidate(), {"overall": "pass"}, [])
    assert scores["overall"] > 0
    assert scores["data_feasibility"] == 1.0


def test_low_risk_can_rank_above_high_risk() -> None:
    low = {
        "candidate_id": "low",
        "automation_risk": "low",
        "shortlist_status": "ready",
        "scores": score_candidate(_base_candidate(), {"overall": "pass"}, []),
    }
    high = {
        "candidate_id": "high",
        "automation_risk": "high",
        "shortlist_status": "blocked",
        "scores": score_candidate(
            _base_candidate(automation_risk="high", required_secrets=["Google_Street_View_Static_API:api_key"], technology_tags=["experimental"]),
            {"overall": "warning"},
            [{"result": "blocked"}],
        ),
    }
    ranked = rank_candidates([high, low])
    assert ranked[0]["candidate_id"] == "low"


def test_tech_innovation_does_not_override_low_data_feasibility() -> None:
    poor_data = score_candidate(
        _base_candidate(technology_tags=["experimental"]),
        {"overall": "fail"},
        [],
    )
    assert poor_data["data_feasibility"] == 0.2
    assert poor_data["overall"] < 0.7
