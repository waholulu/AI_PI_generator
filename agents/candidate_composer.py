from __future__ import annotations

from itertools import product

from agents.research_template_loader import load_research_template
from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposeRequest, ComposedCandidate


def _pick_source(preferred_sources: list[str], registry: SourceRegistry, req: ComposeRequest) -> str | None:
    for src in preferred_sources:
        sid = registry.resolve(src)
        if sid is None:
            continue

        spec = registry.sources.get(sid, {})
        tier = str(spec.get("tier", "stable"))

        if tier == "experimental" and not req.enable_experimental:
            continue
        if tier == "tier2" and not req.enable_tier2:
            continue
        if req.no_paid_api and spec.get("cost_required"):
            continue
        if req.no_manual_download and spec.get("source_type") == "manual_download":
            continue

        return sid
    return None


def _risk_for_source(source_spec: dict) -> str:
    if source_spec.get("tier") == "experimental":
        return "high"
    if source_spec.get("auth_required") or source_spec.get("cost_required"):
        return "medium"
    return "low"


def _technology_tags(exp_source: str, exp_family: str, exp_spec: dict) -> list[str]:
    tags: list[str] = []
    source = exp_source.lower()
    if "osmnx" in source:
        tags.append("osmnx")
    if any(k in source for k in ["nlcd", "viirs", "enviroatlas"]):
        tags.append("remote_sensing")
    if "gtfs" in source:
        tags.append("mobility")
    if exp_spec.get("tier") == "experimental" or exp_family == "streetview_built_form":
        tags.append("experimental")
    return tags


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

        if exp_spec.get("tier") == "experimental" and not req.enable_experimental:
            continue

        exp_source = _pick_source(exp_spec.get("preferred_sources", []), registry, req)
        out_source = _pick_source(out_spec.get("preferred_sources", []), registry, req)
        if not exp_source or not out_source:
            continue

        exp_source_spec = registry.sources.get(exp_source, {})
        out_source_spec = registry.sources.get(out_source, {})

        risk_candidates = [_risk_for_source(exp_source_spec), _risk_for_source(out_source_spec)]
        risk = "high" if "high" in risk_candidates else ("medium" if "medium" in risk_candidates else "low")

        required_secrets: list[str] = []
        if exp_source_spec.get("auth_required") or exp_source_spec.get("cost_required"):
            required_secrets.append(f"{exp_source}:api_key")
        if out_source_spec.get("auth_required") or out_source_spec.get("cost_required"):
            required_secrets.append(f"{out_source}:api_key")

        tech_tags = _technology_tags(exp_source, exp_name, exp_spec)
        cloud_safe = registry.is_cloud_safe(exp_source) and registry.is_cloud_safe(out_source)

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
                claim_strength=method_spec.get("claim_strength", template.get("default_claim_strength", "associational")),
                key_threats=list(method_spec.get("key_threats", [])),
                mitigations=dict(method_spec.get("mitigations", {})),
                technology_tags=tech_tags,
                required_secrets=required_secrets,
                automation_risk=risk,
                cloud_safe=cloud_safe,
            )
        )
        serial += 1
        if len(candidates) >= req.max_candidates:
            break

    return candidates
