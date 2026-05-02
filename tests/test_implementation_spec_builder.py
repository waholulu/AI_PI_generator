from agents.implementation_spec_builder import build_implementation_spec
from models.candidate_composer_schema import ComposedCandidate


def _make_candidate(
    exposure_family="street_network",
    exposure_source="OSMnx_OpenStreetMap",
    outcome_family="physical_inactivity",
    outcome_source="CDC_PLACES",
    unit_of_analysis="census_tract",
    technology_tags=None,
) -> ComposedCandidate:
    return ComposedCandidate(
        candidate_id="beh_001",
        template_id="built_environment_health_v1",
        exposure_family=exposure_family,
        exposure_source=exposure_source,
        exposure_variables=["intersection_density", "edge_density"],
        outcome_family=outcome_family,
        outcome_source=outcome_source,
        outcome_variables=[f"{outcome_family}_prevalence"],
        unit_of_analysis=unit_of_analysis,
        join_plan={"join_key": "GEOID", "boundary_source": ["TIGER_Lines"], "controls": ["ACS"]},
        method_template="cross_sectional_spatial_association",
        key_threats=["socioeconomic_confounding"],
        mitigations={"socioeconomic_confounding": "add_acs_controls"},
        technology_tags=technology_tags or ["osmnx"],
        automation_risk="low",
    )


def test_implementation_spec_contains_core_sections() -> None:
    candidate = _make_candidate()
    spec = build_implementation_spec(candidate)

    assert spec.candidate_id == "beh_001"
    assert spec.data_acquisition_steps
    assert spec.feature_engineering_steps
    assert spec.analysis_steps
    assert spec.smoke_test_plan


def test_implementation_spec_contains_source_use_specs() -> None:
    """ImplementationSpec.source_use_specs is populated for catalogued sources."""
    candidate = _make_candidate(
        exposure_family="density",
        exposure_source="EPA_Smart_Location_Database",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        technology_tags=[],
    )
    spec = build_implementation_spec(candidate)

    assert spec.source_use_specs, "source_use_specs must be non-empty"
    roles = {s.role for s in spec.source_use_specs}
    assert "exposure" in roles
    assert "outcome" in roles

    exp_sus = next(s for s in spec.source_use_specs if s.role == "exposure")
    assert exp_sus.source_id == "EPA_Smart_Location_Database"
    assert exp_sus.native_unit == "census_block_group"
    assert exp_sus.aggregation_method == "population_weighted_mean"
    assert exp_sus.target_unit == "census_tract"


def test_implementation_spec_data_lineage_plan_populated() -> None:
    """data_lineage_plan includes aggregation step for block-group sources."""
    candidate = _make_candidate(
        exposure_family="density",
        exposure_source="EPA_Smart_Location_Database",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        technology_tags=[],
    )
    spec = build_implementation_spec(candidate)

    assert spec.data_lineage_plan, "data_lineage_plan must be non-empty"
    assert "lineage_steps" in spec.data_lineage_plan
    steps = spec.data_lineage_plan["lineage_steps"]
    assert any(s.get("step") == "aggregate_exposure" for s in steps), (
        "Expected aggregate_exposure step for block_group → tract"
    )


def test_acquisition_method_from_catalog() -> None:
    """Acquisition method is read from data catalog when profile is available."""
    # EPA SLD has method: download in its catalog profile
    candidate = _make_candidate(
        exposure_family="density",
        exposure_source="EPA_Smart_Location_Database",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        technology_tags=[],
    )
    spec = build_implementation_spec(candidate)
    exp_step = next(s for s in spec.data_acquisition_steps if s.source_role == "exposure")
    assert exp_step.method == "download"
