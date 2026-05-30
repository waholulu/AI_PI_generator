import pytest

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


@pytest.mark.parametrize(
    "method",
    [
        "target_trial_emulation_iptw",
        "target_trial_emulation_overlap_weighting",
        "target_trial_emulation_tmle",
        "target_trial_emulation_aipw",
        "target_trial_emulation_matching",
        "causal_forest_heterogeneous_treatment_effects",
        "regression_discontinuity",
        "interrupted_time_series",
        "instrumental_variable",
        "diff_in_diff_event_study",
        "synthetic_control",
    ],
)
def test_filler_adds_quasi_causal_method_metadata(method: str) -> None:
    candidate = ComposedCandidate(
        candidate_id="tte_001",
        template_id="built_environment_health_tte_v1",
        exposure_family="walkability",
        exposure_source="EPA_National_Walkability_Index",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
        method_template=method,
        claim_strength="quasi_causal",
        key_threats=[],
        mitigations={},
    )

    threats, mitigations = fill_identification_from_method(candidate)

    assert len(threats) >= 3
    assert set(threats).issubset(set(mitigations.keys()))
