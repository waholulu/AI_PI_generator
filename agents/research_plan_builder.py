from __future__ import annotations

from typing import Any

from agents.source_registry import SourceRegistry
from models.research_plan_schema import (
    DataSourceSpec,
    FeasibilitySpec,
    IdentificationSpec,
    ResearchPlan,
    VariableSpec,
)

_QUASI_CAUSAL_METHODS = {
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


def _as_nonempty(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _claim_strength(candidate: dict[str, Any], method: str) -> str:
    value = str(candidate.get("claim_strength") or "").strip()
    allowed = {"descriptive", "associational", "quasi_causal", "causal"}
    if value in allowed:
        return value
    if method in _QUASI_CAUSAL_METHODS:
        return "quasi_causal"
    return "associational"


def _candidate_data_sources(candidate: dict[str, Any]) -> list[DataSourceSpec]:
    registry = SourceRegistry.load()
    raw_sources = candidate.get("data_sources")
    if not isinstance(raw_sources, list):
        raw_sources = []
        raw_sources.extend(
            [
                {"name": candidate.get("exposure_source"), "role": "exposure", "variable_family": candidate.get("exposure_variable")},
                {"name": candidate.get("outcome_source"), "role": "outcome", "variable_family": candidate.get("outcome_variable")},
            ]
        )
        join_plan = candidate.get("join_plan") or {}
        for name in join_plan.get("controls", []) or []:
            raw_sources.append({"name": name, "role": "control"})
        for name in join_plan.get("boundary_source", []) or []:
            raw_sources.append({"name": name, "role": "boundary"})
        raw_sources.extend([{"name": name} for name in candidate.get("declared_sources", [])])
    specs: list[DataSourceSpec] = []
    seen: set[str] = set()
    for source in raw_sources:
        if isinstance(source, str):
            source = {"name": source}
        name = _as_nonempty(source.get("name") or source.get("source"), "Unknown data source")
        if not name or name == "Unknown data source":
            continue
        canonical = registry.resolve(name) or name
        if canonical in seen:
            continue
        seen.add(canonical)

        registry_spec = registry.get(canonical)
        if registry_spec:
            specs.append(
                registry.enrich_data_source_from_registry(
                    source_id=canonical,
                    role=source.get("role"),
                    variable_family=source.get("variable_family"),
                )
            )
        else:
            specs.append(
                DataSourceSpec(
                    name=name,
                    source_type=str(source.get("source_type") or "unknown"),
                    access_url=str(source.get("access_url") or source.get("url") or ""),
                    documentation_url=str(source.get("documentation_url") or ""),
                    license=str(source.get("license") or ""),
                    expected_format=str(source.get("expected_format") or source.get("format") or ""),
                    access_notes=str(source.get("access_notes") or source.get("accessibility") or ""),
                )
            )
    if not specs:
        specs = [DataSourceSpec(name="Unspecified public source", source_type="unknown")]
    return specs


def _fallback_queries(project_title: str, exposure: str, outcome: str, geography: str) -> list[str]:
    return [
        f"{project_title} {geography}".strip(),
        f"{exposure} {outcome} identification strategy".strip(),
        f"{exposure} {outcome} open data {geography}".strip(),
    ]


def build_research_plan_from_candidate(
    candidate: dict[str, Any],
    evaluation: dict[str, Any] | None,
    run_id: str,
) -> ResearchPlan:
    title = _as_nonempty(candidate.get("title"), "Research Topic")
    exposure_name = _as_nonempty(candidate.get("exposure_variable"), "Exposure variable")
    outcome_name = _as_nonempty(candidate.get("outcome_variable"), "Outcome variable")
    geography = _as_nonempty(candidate.get("geography"), "Unknown geography")
    method = _as_nonempty(candidate.get("method") or candidate.get("method_template"), "descriptive")
    claim_strength = _claim_strength(candidate, method)

    question = _as_nonempty(
        candidate.get("research_question"),
        f"How does {exposure_name} affect {outcome_name} in {geography}?",
    )
    rationale = _as_nonempty(
        candidate.get("brief_rationale") or candidate.get("contribution"),
        f"Assess whether {exposure_name} and {outcome_name} are empirically linked.",
    )

    data_sources = _candidate_data_sources(candidate)
    exposure = VariableSpec(
        name=exposure_name,
        family=str(candidate.get("exposure_family") or candidate.get("exposure_variable_family") or candidate.get("exposure_variable") or ""),
        definition=str(candidate.get("exposure_definition") or ""),
        measurement_proxy=str(candidate.get("exposure_proxy") or ""),
        spatial_unit=str(candidate.get("spatial_unit") or ""),
        temporal_frequency=str(candidate.get("frequency") or ""),
        data_source_names=[d.name for d in data_sources],
    )
    outcome = VariableSpec(
        name=outcome_name,
        family=str(candidate.get("outcome_family") or candidate.get("outcome_variable_family") or candidate.get("outcome_variable") or ""),
        definition=str(candidate.get("outcome_definition") or ""),
        measurement_proxy=str(candidate.get("outcome_proxy") or ""),
        spatial_unit=str(candidate.get("spatial_unit") or ""),
        temporal_frequency=str(candidate.get("frequency") or ""),
        data_source_names=[d.name for d in data_sources],
    )
    identification = IdentificationSpec(
        primary_method=method,
        causal_claim_strength=claim_strength,
        key_threats=list(candidate.get("key_threats") or []),
        mitigations=dict(candidate.get("mitigations") or {}),
    )

    literature_queries = candidate.get("literature_queries") or _fallback_queries(
        title, exposure_name, outcome_name, geography
    )
    if claim_strength == "causal":
        default_hypothesis = f"H1: {exposure_name} has a causal effect on {outcome_name}."
    elif claim_strength == "quasi_causal":
        default_hypothesis = (
            f"H1: {exposure_name} has a quasi-causal effect on {outcome_name}."
        )
    else:
        default_hypothesis = f"H1: {exposure_name} is associated with {outcome_name}."
    hypotheses = list(candidate.get("hypotheses") or [default_hypothesis])

    eval_dict = evaluation or {}
    overall = str(eval_dict.get("overall_verdict") or "warning")
    overall = overall if overall in {"pass", "warning", "fail"} else "warning"
    feasibility = FeasibilitySpec(
        overall_verdict=overall,
        data_verdict=str(eval_dict.get("data_access_verdict") or "warning"),
        novelty_verdict=str(eval_dict.get("novelty_verdict") or "unknown"),
        identification_verdict=str(eval_dict.get("identification_verdict") or "warning"),
        main_risks=list(eval_dict.get("reasons") or []),
    )

    plan = ResearchPlan(
        run_id=run_id,
        project_title=title,
        research_question=question,
        short_rationale=rationale,
        population=str(candidate.get("population") or ""),
        geography=geography,
        time_window=str(candidate.get("time_window") or ""),
        unit_of_analysis=str(candidate.get("unit_of_analysis") or candidate.get("spatial_unit") or ""),
        exposure=exposure,
        outcome=outcome,
        identification=identification,
        data_sources=data_sources,
        literature_queries=list(literature_queries)[:8],
        hypotheses=hypotheses,
        feasibility=feasibility,
        selected_candidate_id=str(
            candidate.get("topic_id")
            or candidate.get("reflection_trace_id")
            or candidate.get("candidate_id")
            or ""
        ),
    )

    # Contract minimums.
    if len(plan.literature_queries) < 3:
        plan.literature_queries.extend(
            _fallback_queries(plan.project_title, plan.exposure.name, plan.outcome.name, plan.geography)
        )
        plan.literature_queries = plan.literature_queries[:3]
    if len(plan.data_sources) < 1:
        plan.data_sources = [DataSourceSpec(name="Unspecified public source")]
    return plan
