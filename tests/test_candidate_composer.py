from agents.candidate_composer import compose_candidates
from models.candidate_composer_schema import ComposeRequest

TTE_METHODS = {
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
}


def test_candidate_composer_generates_candidates() -> None:
    req = ComposeRequest(
        template_id="built_environment_health",
        domain_input="Built environment and health",
        max_candidates=40,
        enable_experimental=False,
    )
    candidates = compose_candidates(req)

    assert len(candidates) >= 20
    assert all(c.exposure_source for c in candidates)
    assert all(c.outcome_source for c in candidates)
    assert all(c.method_template for c in candidates)
    assert all(c.key_threats for c in candidates)
    assert all(c.mitigations for c in candidates)
    assert all(c.automation_risk in {"low", "medium"} for c in candidates)
    assert all("experimental" not in c.technology_tags for c in candidates)


def test_candidate_composer_respects_experimental_toggle() -> None:
    req = ComposeRequest(
        template_id="built_environment_health",
        domain_input="Built environment and health",
        max_candidates=60,
        enable_experimental=False,
    )
    candidates = compose_candidates(req)

    assert all("streetview" not in c.exposure_family for c in candidates)
    assert all(c.claim_strength == "associational" for c in candidates)
    assert all(isinstance(c.cloud_safe, bool) for c in candidates)


def test_experimental_template_generates_streetview_candidates() -> None:
    req = ComposeRequest(
        template_id="built_environment_health_experimental",
        domain_input="Built environment and health",
        max_candidates=10,
        enable_experimental=True,
        preferred_technology=["streetview_cv"],
    )
    candidates = compose_candidates(req)

    assert candidates
    assert all(c.exposure_family == "streetview_built_form" for c in candidates)
    assert all("experimental" in c.technology_tags for c in candidates)
    assert all("vision" in c.technology_tags for c in candidates)
    assert all("streetview_cv" in c.technology_tags for c in candidates)


def test_tte_template_screens_methods_and_selects_one_per_xy() -> None:
    req = ComposeRequest(
        template_id="built_environment_health_tte",
        domain_input="Built environment and health",
        max_candidates=40,
        enable_experimental=False,
    )
    candidates = compose_candidates(req)

    assert candidates
    pairs = [(c.exposure_family, c.outcome_family) for c in candidates]
    assert len(pairs) == len(set(pairs))
    assert all(c.claim_strength == "quasi_causal" for c in candidates)
    assert all(c.exposure_source for c in candidates)
    assert all(c.outcome_source for c in candidates)
    assert all(c.key_threats for c in candidates)
    assert all(c.mitigations for c in candidates)
    assert all(TTE_METHODS.issubset(set(c.method_screening["methods"].keys())) for c in candidates)
    assert all(
        c.method_screening["methods"][c.method_template]["status"] == "eligible"
        for c in candidates
    )
    assert all(
        c.method_screening["methods"]["instrumental_variable"]["status"] == "rejected"
        for c in candidates
    )
