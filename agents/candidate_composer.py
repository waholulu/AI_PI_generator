from __future__ import annotations

from itertools import product

from agents.research_template_loader import load_research_template
from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposeRequest, ComposedCandidate


def _pick_source(preferred_sources: list[str], registry: SourceRegistry, allow_experimental: bool) -> str | None:
    for src in preferred_sources:
        sid = registry.resolve(src)
        if sid is None:
            continue
        spec = registry.sources.get(sid, {})
        tier = spec.get("tier", "stable")
        if tier == "experimental" and not allow_experimental:
            continue
        return sid
    return None


def _risk_for_source(source_spec: dict) -> str:
    if source_spec.get("tier") == "experimental":
        return "high"
    if source_spec.get("auth_required") or source_spec.get("cost_required"):
        return "medium"
    return "low"


def compose_candidates(req: ComposeRequest) -> list[ComposedCandidate]:
    template = load_research_template(req.template_id)
    registry = SourceRegistry.load()

    exposures = template.get("allowed_exposure_families", {})
    outcomes = template.get("allowed_outcome_families", {})
    methods = template.get("allowed_methods", {})
    default_unit = template.get("default_unit_of_analysis", "census_tract")

    candidates: list[ComposedCandidate] = []
    serial = 1

    for exp_name, out_name, method_name in product(exposures.keys(), outcomes.keys(), methods.keys()):
        exp_spec = exposures[exp_name]
        out_spec = outcomes[out_name]
        method_spec = methods[method_name]

        exp_source = _pick_source(exp_spec.get("preferred_sources", []), registry, req.enable_experimental)
        out_source = _pick_source(out_spec.get("preferred_sources", []), registry, True)
        if not exp_source or not out_source:
            continue

        exp_source_spec = registry.sources.get(exp_source, {})
        out_source_spec = registry.sources.get(out_source, {})

        risk_candidates = [_risk_for_source(exp_source_spec), _risk_for_source(out_source_spec)]
        risk = "high" if "high" in risk_candidates else ("medium" if "medium" in risk_candidates else "low")

        tech_tags = []
        if "osmnx" in exp_source.lower():
            tech_tags.append("osmnx")
        if exp_spec.get("tier") == "experimental":
            tech_tags.append("experimental")

        candidates.append(
            ComposedCandidate(
                candidate_id=f"beh_{serial:03d}",
                template_id=template.get("template_id", req.template_id),
                exposure_family=exp_name,
                exposure_source=exp_source,
                exposure_variables=list(exp_spec.get("derived_features", []))[:5],
                outcome_family=out_name,
                outcome_source=out_source,
                outcome_variables=[out_name],
                unit_of_analysis=default_unit,
                join_plan={
                    "boundary_source": template.get("default_boundary_source", ["TIGER_Lines"]),
                    "controls": template.get("default_controls", ["ACS"]),
                    "join_key": "GEOID",
                },
                method_template=method_name,
                key_threats=list(method_spec.get("key_threats", [])),
                mitigations=dict(method_spec.get("mitigations", {})),
                technology_tags=tech_tags,
                automation_risk=risk,
            )
        )
        serial += 1
        if len(candidates) >= req.max_candidates:
            break

    return candidates
