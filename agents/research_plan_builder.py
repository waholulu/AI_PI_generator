from __future__ import annotations

from typing import Any

from models.research_plan_schema import (
    DataSourceSpec,
    FeasibilitySpec,
    IdentificationSpec,
    ResearchPlan,
    VariableSpec,
)


def _as_nonempty(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _candidate_data_sources(candidate: dict[str, Any]) -> list[DataSourceSpec]:
    raw_sources = candidate.get("data_sources")
    if not isinstance(raw_sources, list):
        raw_sources = [{"name": name} for name in candidate.get("declared_sources", [])]
    specs: list[DataSourceSpec] = []
    for source in raw_sources:
        if isinstance(source, str):
            source = {"name": source}
        name = _as_nonempty(source.get("name") or source.get("source"), "Unknown data source")
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
    method = _as_nonempty(candidate.get("method"), "descriptive")

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
        definition=str(candidate.get("exposure_definition") or ""),
        measurement_proxy=str(candidate.get("exposure_proxy") or ""),
        spatial_unit=str(candidate.get("spatial_unit") or ""),
        temporal_frequency=str(candidate.get("frequency") or ""),
        data_source_names=[d.name for d in data_sources],
    )
    outcome = VariableSpec(
        name=outcome_name,
        definition=str(candidate.get("outcome_definition") or ""),
        measurement_proxy=str(candidate.get("outcome_proxy") or ""),
        spatial_unit=str(candidate.get("spatial_unit") or ""),
        temporal_frequency=str(candidate.get("frequency") or ""),
        data_source_names=[d.name for d in data_sources],
    )
    identification = IdentificationSpec(
        primary_method=method,
        key_threats=list(candidate.get("key_threats") or []),
        mitigations=dict(candidate.get("mitigations") or {}),
    )

    literature_queries = candidate.get("literature_queries") or _fallback_queries(
        title, exposure_name, outcome_name, geography
    )
    hypotheses = list(candidate.get("hypotheses") or [f"H1: {exposure_name} is associated with {outcome_name}."])

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
