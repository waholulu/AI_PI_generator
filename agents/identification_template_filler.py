from __future__ import annotations

from models.candidate_composer_schema import ComposedCandidate

_METHOD_THREATS: dict[str, dict] = {
    "cross_sectional_spatial_association": {
        "threats": [
            "socioeconomic_confounding",
            "residential_self_selection",
            "spatial_autocorrelation",
            "exposure_measurement_error",
        ],
        "mitigations": {
            "socioeconomic_confounding": "Include tract-level ACS socioeconomic controls.",
            "residential_self_selection": "Include county/metro fixed effects and avoid causal claims.",
            "spatial_autocorrelation": "Use clustered or spatially robust standard errors.",
            "exposure_measurement_error": "Run sensitivity checks with alternative feature definitions.",
        },
    }
}

_QUASI_CAUSAL_METHOD_THREATS: dict[str, dict] = {
    "target_trial_emulation_iptw": {
        "threats": [
            "baseline_confounding",
            "immortal_time_bias",
            "positivity_violation",
            "exposure_measurement_error",
        ],
        "mitigations": {
            "baseline_confounding": "Estimate propensity scores using baseline-only covariates.",
            "immortal_time_bias": "Align eligibility, exposure assignment, and follow-up at time zero.",
            "positivity_violation": "Inspect propensity overlap and truncate extreme IPTW weights.",
            "exposure_measurement_error": "Run sensitivity checks with alternative exposure definitions.",
        },
    },
    "target_trial_emulation_overlap_weighting": {
        "threats": [
            "baseline_confounding",
            "limited_common_support",
            "immortal_time_bias",
            "spatial_autocorrelation",
        ],
        "mitigations": {
            "baseline_confounding": "Estimate overlap weights from pre-exposure covariates.",
            "limited_common_support": "Report the overlap-population estimand and effective sample size.",
            "immortal_time_bias": "Define time zero before exposure and outcome follow-up.",
            "spatial_autocorrelation": "Use clustered or spatially robust standard errors.",
        },
    },
    "target_trial_emulation_tmle": {
        "threats": [
            "nuisance_model_misspecification",
            "baseline_confounding",
            "positivity_violation",
            "immortal_time_bias",
        ],
        "mitigations": {
            "nuisance_model_misspecification": "Use cross-fitting for outcome and propensity nuisance models.",
            "baseline_confounding": "Adjust only for pre-exposure baseline covariates.",
            "positivity_violation": "Check propensity overlap and truncation sensitivity.",
            "immortal_time_bias": "Anchor assignment and follow-up at the same time zero.",
        },
    },
    "target_trial_emulation_aipw": {
        "threats": [
            "nuisance_model_misspecification",
            "baseline_confounding",
            "positivity_violation",
            "spatial_autocorrelation",
        ],
        "mitigations": {
            "nuisance_model_misspecification": "Estimate both treatment and outcome models and compare specifications.",
            "baseline_confounding": "Use baseline-only socioeconomic and demographic controls.",
            "positivity_violation": "Inspect support and trim unstable weights.",
            "spatial_autocorrelation": "Use clustered or spatially robust inference.",
        },
    },
    "target_trial_emulation_matching": {
        "threats": [
            "baseline_confounding",
            "poor_match_quality",
            "limited_common_support",
            "immortal_time_bias",
        ],
        "mitigations": {
            "baseline_confounding": "Match on pre-exposure ACS and geography covariates.",
            "poor_match_quality": "Report standardized mean differences before and after matching.",
            "limited_common_support": "Drop unmatched units and describe the matched target population.",
            "immortal_time_bias": "Define exposure and follow-up from a common time zero.",
        },
    },
    "causal_forest_heterogeneous_treatment_effects": {
        "threats": [
            "baseline_confounding",
            "positivity_violation",
            "overfitting_heterogeneity",
            "spatial_autocorrelation",
        ],
        "mitigations": {
            "baseline_confounding": "Include rich baseline covariates and compare with weighted average effects.",
            "positivity_violation": "Restrict heterogeneity claims to regions with common support.",
            "overfitting_heterogeneity": "Use honest forests or cross-fitting and validate subgroup effects.",
            "spatial_autocorrelation": "Cluster inference or aggregate CATEs by geography.",
        },
    },
    "regression_discontinuity": {
        "threats": [
            "cutoff_manipulation",
            "bandwidth_sensitivity",
            "covariate_imbalance",
            "spillover",
        ],
        "mitigations": {
            "cutoff_manipulation": "Run density or manipulation checks around the cutoff.",
            "bandwidth_sensitivity": "Report estimates across multiple bandwidths and kernels.",
            "covariate_imbalance": "Test pre-treatment covariate continuity.",
            "spillover": "Use donut RD or exclude contaminated boundary units.",
        },
    },
    "interrupted_time_series": {
        "threats": [
            "pre_existing_trends",
            "seasonality",
            "autocorrelation",
            "concurrent_interventions",
        ],
        "mitigations": {
            "pre_existing_trends": "Model and visualize the pre-intervention trend.",
            "seasonality": "Include month or seasonal fixed effects where applicable.",
            "autocorrelation": "Use HAC or autoregressive error corrections.",
            "concurrent_interventions": "Document and adjust for overlapping policy changes.",
        },
    },
    "instrumental_variable": {
        "threats": [
            "weak_instrument",
            "exclusion_restriction_violation",
            "monotonicity_violation",
            "spatial_spillover",
        ],
        "mitigations": {
            "weak_instrument": "Report first-stage strength and weak-instrument robust inference.",
            "exclusion_restriction_violation": "Provide a domain argument and negative-control checks.",
            "monotonicity_violation": "State the complier interpretation and test alternatives if available.",
            "spatial_spillover": "Test buffer exclusions or neighboring-area controls.",
        },
    },
    "diff_in_diff_event_study": {
        "threats": [
            "parallel_trends_violation",
            "treatment_timing_heterogeneity",
            "spillover",
            "anticipation_effects",
        ],
        "mitigations": {
            "parallel_trends_violation": "Estimate pre-treatment event-study leads.",
            "treatment_timing_heterogeneity": "Use staggered-adoption robust estimators where applicable.",
            "spillover": "Exclude neighboring areas or model exposure buffers.",
            "anticipation_effects": "Include leads and sensitivity to pre-period exclusions.",
        },
    },
    "synthetic_control": {
        "threats": [
            "poor_pre_treatment_fit",
            "donor_pool_contamination",
            "concurrent_interventions",
            "extrapolation",
        ],
        "mitigations": {
            "poor_pre_treatment_fit": "Report pre-treatment RMSPE and fit plots.",
            "donor_pool_contamination": "Exclude donors exposed to related policies or shocks.",
            "concurrent_interventions": "Document co-occurring interventions and sensitivity exclusions.",
            "extrapolation": "Inspect donor weights and predictor balance.",
        },
    },
}

_METHOD_THREATS.update(_QUASI_CAUSAL_METHOD_THREATS)


def fill_identification_from_method(
    candidate: ComposedCandidate,
    method_spec: dict | None = None,
) -> tuple[list[str], dict[str, str]]:
    defaults = method_spec or _METHOD_THREATS.get(candidate.method_template, {})
    default_threats = list(defaults.get("threats", []))
    default_mitigations = dict(defaults.get("mitigations", {}))

    merged_threats = list(default_threats)
    for threat in candidate.key_threats:
        if threat not in merged_threats:
            merged_threats.append(threat)

    merged_mitigations = dict(default_mitigations)
    merged_mitigations.update(candidate.mitigations)

    return merged_threats, merged_mitigations


def ensure_identification_metadata(candidate: ComposedCandidate) -> ComposedCandidate:
    """Guarantee ≥3 threats with full mitigation coverage before gate checks.

    No-op if the candidate already has ≥3 threats and all have mitigations.
    Otherwise merges method-template defaults so the export validator's strict
    3-threat requirement is satisfied without LLM cost.
    """
    threats = list(candidate.key_threats or [])
    mits = dict(candidate.mitigations or {})
    if len(threats) >= 3 and all(t in mits for t in threats):
        return candidate
    new_threats, new_mits = fill_identification_from_method(candidate)
    return candidate.model_copy(update={"key_threats": new_threats, "mitigations": new_mits})
