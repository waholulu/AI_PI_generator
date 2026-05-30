from agents.candidate_composer import compose_candidates
from agents.method_data_compatibility import (
    extract_affordances,
    screen_methods,
    select_methods_for_candidate,
)
from models.candidate_composer_schema import ComposeRequest, ComposedCandidate


def _candidate(**kwargs) -> ComposedCandidate:
    base = {
        "candidate_id": "tte_001",
        "template_id": "built_environment_health_tte_v1",
        "exposure_family": "walkability",
        "exposure_source": "EPA_National_Walkability_Index",
        "outcome_family": "physical_inactivity",
        "outcome_source": "CDC_PLACES",
        "unit_of_analysis": "census_tract",
        "join_plan": {"boundary_source": ["TIGER_Lines"], "controls": ["ACS"], "join_key": "GEOID"},
        "method_template": "target_trial_emulation_overlap_weighting",
        "claim_strength": "quasi_causal",
        "key_threats": ["baseline_confounding", "limited_common_support", "immortal_time_bias"],
        "mitigations": {
            "baseline_confounding": "m1",
            "limited_common_support": "m2",
            "immortal_time_bias": "m3",
        },
    }
    base.update(kwargs)
    return ComposedCandidate(**base)


def test_cross_sectional_walkability_supports_tte_but_not_strong_qe_methods() -> None:
    candidate = _candidate()

    affordances = extract_affordances(candidate)
    screening = screen_methods(affordances)

    assert screening["target_trial_emulation_overlap_weighting"]["status"] == "eligible"
    assert screening["target_trial_emulation_aipw"]["status"] == "eligible"
    assert screening["target_trial_emulation_tmle"]["status"] == "eligible"
    assert screening["regression_discontinuity"]["status"] == "rejected"
    assert screening["instrumental_variable"]["status"] == "rejected"
    assert screening["diff_in_diff_event_study"]["status"] == "rejected"
    assert screening["synthetic_control"]["status"] == "rejected"


def test_panel_exposure_without_intervention_still_rejects_did_its() -> None:
    candidate = _candidate(
        exposure_family="nighttime_lights",
        exposure_source="VIIRS",
        outcome_family="cardiovascular_disease",
    )

    affordances = extract_affordances(candidate)
    screening = screen_methods(affordances)

    assert affordances["multi_year_panel_available"] is True
    assert screening["target_trial_emulation_overlap_weighting"]["status"] == "eligible"
    assert screening["diff_in_diff_event_study"]["status"] == "rejected"
    assert screening["interrupted_time_series"]["status"] == "rejected"
    assert screening["instrumental_variable"]["status"] == "rejected"


def test_declared_cutoff_promotes_rd_over_tte() -> None:
    candidate = _candidate(
        join_plan={
            "boundary_source": ["TIGER_Lines"],
            "controls": ["ACS"],
            "join_key": "GEOID",
            "cutoff_variable": "eligibility_score_threshold",
        }
    )
    method_specs = {
        c.method_template: {"claim_strength": c.claim_strength}
        for c in compose_candidates(
            ComposeRequest(
                template_id="built_environment_health_tte",
                domain_input="Built environment and health",
                max_candidates=1,
            )
        )
    }
    method_specs.update(
        {
            "regression_discontinuity": {"claim_strength": "quasi_causal"},
            "target_trial_emulation_overlap_weighting": {"claim_strength": "quasi_causal"},
        }
    )

    selected = select_methods_for_candidate(candidate, method_specs)

    assert selected["methods"]["regression_discontinuity"]["status"] == "eligible"
    assert selected["primary_method"] == "regression_discontinuity"
