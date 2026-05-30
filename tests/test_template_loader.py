from agents.research_template_loader import load_research_template, validate_template_sources

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


def test_template_loads() -> None:
    template = load_research_template("built_environment_health")

    assert template["template_id"] == "built_environment_health_v1"
    assert "allowed_exposure_families" in template
    assert len(template["allowed_exposure_families"]) >= 10


def test_template_sources_exist_in_registry() -> None:
    template = load_research_template("built_environment_health")
    missing = validate_template_sources(template)

    assert missing == []


def test_tte_template_loads_and_declares_quasi_causal_methods() -> None:
    template = load_research_template("built_environment_health_tte")

    assert template["template_id"] == "built_environment_health_tte_v1"
    assert template["kind"] == "quasi_causal_research"
    assert template["default_claim_strength"] == "quasi_causal"
    assert TTE_METHODS.issubset(set(template["allowed_methods"].keys()))


def test_tte_template_sources_exist_in_registry() -> None:
    template = load_research_template("built_environment_health_tte")
    missing = validate_template_sources(template)

    assert missing == []
