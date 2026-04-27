from agents.identification_template_filler import fill_identification_from_method
from models.candidate_composer_schema import ComposedCandidate


def test_filler_adds_default_threats_and_mitigations() -> None:
    candidate = ComposedCandidate(
        candidate_id="beh_001",
        template_id="built_environment_health_v1",
        exposure_family="street_network",
        exposure_source="OSMnx_OpenStreetMap",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
        method_template="cross_sectional_spatial_association",
        key_threats=[],
        mitigations={},
    )

    threats, mitigations = fill_identification_from_method(candidate)
    assert len(threats) >= 4
    assert set(threats).issubset(set(mitigations.keys()))
