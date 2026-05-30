from __future__ import annotations

from typing import Any

from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposedCandidate

QUASI_CAUSAL_METHODS = [
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
]

_PRIMARY_PRIORITY = [
    "regression_discontinuity",
    "instrumental_variable",
    "diff_in_diff_event_study",
    "interrupted_time_series",
    "synthetic_control",
    "target_trial_emulation_overlap_weighting",
    "target_trial_emulation_aipw",
    "target_trial_emulation_tmle",
    "target_trial_emulation_iptw",
    "target_trial_emulation_matching",
]


def _is_machine_readable(registry: SourceRegistry, source_name: str) -> bool:
    spec = registry.get(source_name) or {}
    return bool(spec.get("machine_readable"))


def _coverage_span(registry: SourceRegistry, source_name: str) -> int:
    spec = registry.get(source_name) or {}
    try:
        start = int(spec.get("coverage_year_min") or 0)
        end = int(spec.get("coverage_year_max") or 0)
    except (TypeError, ValueError):
        return 0
    if start <= 0 or end <= 0:
        return 0
    return max(0, end - start)


def _cross_sectional_only(registry: SourceRegistry, source_name: str) -> bool:
    profile = registry.get_profile(source_name)
    if profile and profile.temporal_coverage:
        return bool(profile.temporal_coverage.cross_sectional_only)
    spec = registry.get(source_name) or {}
    return _coverage_span(registry, source_name) == 0 and bool(spec.get("coverage_year_min"))


def _has_any_control(registry: SourceRegistry, controls: list[str]) -> bool:
    for control in controls:
        spec = registry.get(control) or {}
        if "control" in (spec.get("roles") or []) and bool(spec.get("machine_readable")):
            return True
    return False


def _has_join_path(candidate: ComposedCandidate, registry: SourceRegistry) -> bool:
    boundary_sources = candidate.join_plan.get("boundary_source") or []
    for source in boundary_sources:
        spec = registry.get(source) or {}
        if "boundary" in (spec.get("roles") or []):
            return True
    return bool(registry.resolve(candidate.outcome_source))


def extract_affordances(
    candidate: ComposedCandidate,
    registry: SourceRegistry | None = None,
) -> dict[str, bool]:
    registry = registry or SourceRegistry.load()
    join_plan = candidate.join_plan or {}
    controls = list(join_plan.get("controls") or [])

    exp_resolved = registry.resolve(candidate.exposure_source) is not None
    out_resolved = registry.resolve(candidate.outcome_source) is not None
    data_available = (
        exp_resolved
        and out_resolved
        and _is_machine_readable(registry, candidate.exposure_source)
        and _is_machine_readable(registry, candidate.outcome_source)
    )
    baseline_covariates = _has_any_control(registry, controls)
    joinable = _has_join_path(candidate, registry)

    exp_span = _coverage_span(registry, candidate.exposure_source)
    out_span = _coverage_span(registry, candidate.outcome_source)
    multi_year_panel = (
        exp_span >= 2
        and out_span >= 2
        and not _cross_sectional_only(registry, candidate.exposure_source)
        and not _cross_sectional_only(registry, candidate.outcome_source)
    )

    return {
        "data_available": data_available,
        "joinable": joinable,
        "baseline_covariates_available": baseline_covariates,
        "multi_year_panel_available": multi_year_panel,
        "intervention_or_policy_time_available": bool(
            join_plan.get("intervention_time")
            or join_plan.get("policy_date")
            or join_plan.get("shock_date")
        ),
        "cutoff_or_threshold_available": bool(
            join_plan.get("cutoff_variable")
            or join_plan.get("assignment_score")
            or join_plan.get("threshold")
        ),
        "plausible_instrument_available": bool(
            join_plan.get("instrument")
            or join_plan.get("instrument_source")
            or join_plan.get("instrument_rationale")
        ),
        "treated_control_units_available": bool(
            join_plan.get("treated_control_units")
            or (join_plan.get("treated_units") and join_plan.get("control_units"))
        ),
        "donor_pool_available": bool(join_plan.get("donor_pool")),
        "common_support_plausible": data_available and joinable and baseline_covariates,
    }


def _result(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def screen_methods(
    affordances: dict[str, bool],
    methods: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    methods = methods or QUASI_CAUSAL_METHODS
    out: dict[str, dict[str, str]] = {}

    base_ok = (
        affordances["data_available"]
        and affordances["joinable"]
        and affordances["baseline_covariates_available"]
        and affordances["common_support_plausible"]
    )

    for method in methods:
        if method in {
            "target_trial_emulation_iptw",
            "target_trial_emulation_overlap_weighting",
            "target_trial_emulation_tmle",
            "target_trial_emulation_aipw",
            "target_trial_emulation_matching",
        }:
            out[method] = (
                _result("eligible", "baseline covariates and common support are available")
                if base_ok
                else _result("rejected", "missing baseline covariates, join path, or common support")
            )
        elif method == "causal_forest_heterogeneous_treatment_effects":
            out[method] = (
                _result("review", "usable as a heterogeneity extension after a primary effect estimate")
                if base_ok
                else _result("rejected", "missing baseline covariates or common support")
            )
        elif method == "regression_discontinuity":
            out[method] = (
                _result("eligible", "cutoff or assignment score is declared")
                if affordances["cutoff_or_threshold_available"]
                else _result("rejected", "no cutoff or assignment score declared")
            )
        elif method == "instrumental_variable":
            out[method] = (
                _result("eligible", "instrument and rationale are declared")
                if affordances["plausible_instrument_available"]
                else _result("rejected", "no named instrument declared")
            )
        elif method == "diff_in_diff_event_study":
            out[method] = (
                _result("eligible", "multi-year panel and treated/control structure are declared")
                if affordances["multi_year_panel_available"]
                and affordances["treated_control_units_available"]
                else _result("rejected", "no treated/control panel structure declared")
            )
        elif method == "interrupted_time_series":
            out[method] = (
                _result("eligible", "multi-year panel and intervention time are declared")
                if affordances["multi_year_panel_available"]
                and affordances["intervention_or_policy_time_available"]
                else _result("rejected", "no intervention time with multi-year outcome panel declared")
            )
        elif method == "synthetic_control":
            out[method] = (
                _result("eligible", "intervention time, donor pool, and pre-period panel are declared")
                if affordances["multi_year_panel_available"]
                and affordances["intervention_or_policy_time_available"]
                and affordances["donor_pool_available"]
                else _result("rejected", "no intervention time, donor pool, and pre-period panel declared")
            )
    return out


def select_methods_for_candidate(
    candidate: ComposedCandidate,
    method_specs: dict[str, Any],
    registry: SourceRegistry | None = None,
) -> dict[str, Any]:
    method_names = [m for m in QUASI_CAUSAL_METHODS if m in method_specs]
    affordances = extract_affordances(candidate, registry)
    methods = screen_methods(affordances, method_names)

    primary = ""
    for method in _PRIMARY_PRIORITY:
        if methods.get(method, {}).get("status") == "eligible":
            primary = method
            break
    if not primary:
        for method, info in methods.items():
            if info.get("status") == "review":
                primary = method
                break
    if not primary and method_names:
        primary = method_names[0]

    secondary = ""
    if methods.get("causal_forest_heterogeneous_treatment_effects", {}).get("status") in {
        "eligible",
        "review",
    } and primary != "causal_forest_heterogeneous_treatment_effects":
        secondary = "causal_forest_heterogeneous_treatment_effects"
    else:
        for method, info in methods.items():
            if method != primary and info.get("status") == "eligible":
                secondary = method
                break

    return {
        "affordances": affordances,
        "methods": methods,
        "primary_method": primary,
        "secondary_method": secondary,
        "primary_reason": methods.get(primary, {}).get("reason", ""),
    }
