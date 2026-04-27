from agents.implementation_spec_builder import build_implementation_spec
from models.candidate_composer_schema import ComposedCandidate


def test_implementation_spec_contains_core_sections() -> None:
    candidate = ComposedCandidate(
        candidate_id="beh_001",
        template_id="built_environment_health_v1",
        exposure_family="street_network",
        exposure_source="OSMnx_OpenStreetMap",
        exposure_variables=["intersection_density", "edge_density"],
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        outcome_variables=["physical_inactivity"],
        unit_of_analysis="census_tract",
        join_plan={"join_key": "GEOID"},
        method_template="cross_sectional_spatial_association",
        key_threats=["socioeconomic_confounding"],
        mitigations={"socioeconomic_confounding": "add_acs_controls"},
        technology_tags=["osmnx"],
        automation_risk="low",
    )

    spec = build_implementation_spec(candidate)

    assert spec.candidate_id == "beh_001"
    assert spec.data_acquisition_steps
    assert spec.feature_engineering_steps
    assert spec.analysis_steps
    assert spec.smoke_test_plan
