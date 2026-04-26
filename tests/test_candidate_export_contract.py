from agents.candidate_export_validator import validate_candidate_export_contract
from models.candidate_composer_schema import ComposedCandidate


def _candidate(**kwargs) -> ComposedCandidate:
    base = dict(
        candidate_id="beh_001",
        template_id="built_environment_health_v1",
        exposure_family="street_network",
        exposure_source="OSMnx_OpenStreetMap",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
        join_plan={"controls": ["ACS"], "boundary_source": ["TIGER_Lines"]},
        method_template="cross_sectional_spatial_association",
        key_threats=["t1", "t2", "t3"],
        mitigations={"t1": "m1", "t2": "m2", "t3": "m3"},
        required_secrets=[],
        technology_tags=[],
        automation_risk="low",
        cloud_safe=True,
    )
    base.update(kwargs)
    return ComposedCandidate(**base)


def test_export_contract_ready() -> None:
    result = validate_candidate_export_contract(_candidate(), {"overall": "pass", "shortlist_status": "ready"})
    assert result["shortlist_status"] == "ready"
    assert result["claude_code_ready"] is True


def test_export_contract_required_secrets_forces_review() -> None:
    result = validate_candidate_export_contract(
        _candidate(required_secrets=["MAPILLARY_KEY"]),
        {"overall": "pass", "shortlist_status": "ready"},
    )
    assert result["shortlist_status"] == "review"
    assert result["claude_code_ready"] is False


def test_export_contract_missing_threats_blocked() -> None:
    result = validate_candidate_export_contract(
        _candidate(key_threats=["t1"]),
        {"overall": "pass", "shortlist_status": "ready"},
    )
    assert result["shortlist_status"] == "blocked"
    assert "missing_identification_threats" in result["blocking_reasons"]
