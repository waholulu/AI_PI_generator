from __future__ import annotations

from agents.research_template_loader import load_research_template
from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposeRequest, ComposedCandidate

# First-batch high-pass (exposure_family, outcome_family) pairs that should always
# be included when their sources resolve successfully. These represent the most data-
# feasible combinations and are generated in round-1 of stratified composition.
_HIGH_PASS_PAIRS: list[tuple[str, str]] = [
    ("street_network", "physical_inactivity"),
    ("street_network", "obesity"),
    ("walkability", "physical_inactivity"),
    ("destination_accessibility", "obesity"),
    ("green_space", "poor_mental_health"),
    ("impervious_surface", "asthma"),
    ("tree_canopy", "asthma"),
    ("transit_access", "physical_inactivity"),
    ("building_density", "physical_inactivity"),
    ("nighttime_lights", "cardiovascular_disease"),
]


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


def _build_candidate(
    serial: int,
    exp_name: str,
    out_name: str,
    method_name: str,
    exp_spec: dict,
    out_spec: dict,
    method_spec: dict,
    registry: SourceRegistry,
    req: ComposeRequest,
    template: dict,
) -> ComposedCandidate | None:
    """Build one candidate or return None if sources can't be resolved."""
    exp_source = _pick_source(exp_spec.get("preferred_sources", []), registry, req)
    out_source = _pick_source(out_spec.get("preferred_sources", []), registry, req)
    if not exp_source or not out_source:
        return None

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

    if risk == "high" or required_secrets:
        shortlist = "review"
    else:
        shortlist = "ready"

    if req.no_paid_api and (
        exp_source_spec.get("cost_required") or out_source_spec.get("cost_required")
    ):
        shortlist = "blocked"
    elif req.automation_risk_tolerance == "low_only" and risk in {"medium", "high"}:
        shortlist = "review"
    elif req.automation_risk_tolerance == "low_medium" and risk == "high":
        shortlist = "review"

    return ComposedCandidate(
        candidate_id=f"beh_{serial:03d}",
        template_id=template.get("template_id", req.template_id),
        exposure_family=exp_name,
        exposure_source=exp_source,
        exposure_variables=list(exp_spec.get("derived_features", []))[:5],
        outcome_family=out_name,
        outcome_source=out_source,
        outcome_variables=[out_name],
        unit_of_analysis=template.get("default_unit_of_analysis", "census_tract"),
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
        initial_shortlist_status=shortlist,
    )


def compose_candidates(req: ComposeRequest) -> list[ComposedCandidate]:
    template = load_research_template(req.template_id)
    registry = SourceRegistry.load()

    exposures = template.get("allowed_exposure_families", {})
    outcomes = template.get("allowed_outcome_families", {})
    methods = template.get("allowed_methods", {})

    # Filter out experimental families when not enabled
    eligible_exposures = {
        k: v for k, v in exposures.items()
        if not (
            (v.get("tier") == "experimental" or k == "streetview_built_form")
            and not req.enable_experimental
        ) and not (
            k == "streetview_built_form"
            and "streetview_cv" not in req.preferred_technology
        )
    }

    # Resolve the first usable method (deepvision methods filtered when not enabled)
    eligible_methods = {
        k: v for k, v in methods.items()
        if not ("deepvision" in k and "deep_learning" not in req.preferred_technology)
    }

    candidates: list[ComposedCandidate] = []
    serial = 1
    seen: set[tuple[str, str, str]] = set()

    def _add(exp_name: str, out_name: str, method_name: str) -> bool:
        nonlocal serial
        key = (exp_name, out_name, method_name)
        if key in seen:
            return False
        if exp_name not in eligible_exposures or out_name not in outcomes or method_name not in eligible_methods:
            return False
        seen.add(key)
        c = _build_candidate(
            serial, exp_name, out_name, method_name,
            eligible_exposures[exp_name], outcomes[out_name], eligible_methods[method_name],
            registry, req, template,
        )
        if c is None:
            return False
        candidates.append(c)
        serial += 1
        return True

    method_name = next(iter(eligible_methods), None)
    if method_name is None:
        return []

    # Round 1: high-pass pairs — always attempt these first
    for exp_name, out_name in _HIGH_PASS_PAIRS:
        if len(candidates) >= req.max_candidates:
            break
        _add(exp_name, out_name, method_name)

    # Round 2: ensure every eligible exposure family has at least one candidate
    outcome_list = list(outcomes.keys())
    for exp_name in eligible_exposures:
        if len(candidates) >= req.max_candidates:
            break
        for out_name in outcome_list:
            if _add(exp_name, out_name, method_name):
                break  # one per family in this round

    # Round 3: fill remaining slots with remaining (exposure, outcome) combos,
    # cycling through outcomes first so coverage is spread across families
    for out_name in outcome_list:
        for exp_name in eligible_exposures:
            if len(candidates) >= req.max_candidates:
                break
            _add(exp_name, out_name, method_name)
        if len(candidates) >= req.max_candidates:
            break

    return candidates
