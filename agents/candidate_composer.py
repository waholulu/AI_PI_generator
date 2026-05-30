from __future__ import annotations

from agents.method_data_compatibility import select_methods_for_candidate
from agents.research_template_loader import load_research_template
from agents.source_registry import SourceRegistry
from agents.task_seed_generator import TaskSeed, TaskSeedGenerator
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


def _pick_source(
    preferred_sources: list[str],
    registry: SourceRegistry,
    req: ComposeRequest,
    expected_role: str = "",
    variable_family: str = "",
) -> str | None:
    for src in preferred_sources:
        sid = registry.resolve(src)
        if sid is None:
            continue

        spec = registry.sources.get(sid, {})
        tier = str(spec.get("tier", "stable"))

        if expected_role and expected_role not in (spec.get("roles") or []):
            continue
        families = spec.get("variable_families") or {}
        if variable_family and variable_family not in families:
            continue
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
    if exp_family == "streetview_built_form" or (
        "street" in source and "image" in source
    ):
        tags.extend(["vision", "streetview_cv"])
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
    task_seed: TaskSeed | None = None,
) -> ComposedCandidate | None:
    """Build one candidate or return None if sources can't be resolved."""
    exp_source = _pick_source(
        exp_spec.get("preferred_sources", []), registry, req,
        expected_role="exposure", variable_family=exp_name,
    )
    out_source = _pick_source(
        out_spec.get("preferred_sources", []), registry, req,
        expected_role="outcome", variable_family=out_name,
    )
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
        outcome_task_id=task_seed.task_id if task_seed else None,
        outcome_task_label=task_seed.task_label if task_seed else None,
        outcome_task_description=task_seed.task_description if task_seed else None,
        outcome_task_modality=task_seed.modality if task_seed else None,
        outcome_task_dataset_hint=task_seed.dataset_hint if task_seed else None,
        outcome_task_domain_input=req.domain_input if task_seed else None,
    )


def _compose_training_research(
    req: ComposeRequest,
    template: dict,
    registry: SourceRegistry,
) -> list[ComposedCandidate]:
    """Training-research path: outcome axis = domain-derived tasks.

    The static `allowed_outcome_families` block (task_accuracy /
    instruction_following / generation_quality) becomes the *metric family*
    palette; the concrete Y axis comes from `TaskSeedGenerator`. Each
    candidate keeps `outcome_family` = metric_family (so registry lookups
    and downstream code still work) and carries the task identity in the
    new `outcome_task_*` fields.

    Falls back to the static Cartesian product (legacy behaviour) when the
    task generator returns nothing usable or all task seeds map to metric
    families absent from the template.
    """
    exposures = template.get("allowed_exposure_families", {})
    outcomes = template.get("allowed_outcome_families", {})
    methods = template.get("allowed_methods", {})

    seeds = TaskSeedGenerator().generate(req.domain_input or "")
    # Drop seeds whose metric_family isn't declared by the template — keeps
    # registry lookups honest.
    seeds = [s for s in seeds if s.metric_family in outcomes]
    if not seeds:
        return []

    eligible_exposures = dict(exposures)
    eligible_methods = dict(methods)
    method_name = next(iter(eligible_methods), None)
    if method_name is None:
        return []
    method_spec = eligible_methods[method_name]

    candidates: list[ComposedCandidate] = []
    serial = 1
    seen: set[tuple[str, str]] = set()

    # Round 1: one candidate per (strategy, task) — broad coverage first.
    for seed in seeds:
        for exp_name, exp_spec in eligible_exposures.items():
            if len(candidates) >= req.max_candidates:
                break
            key = (exp_name, seed.task_id)
            if key in seen:
                continue
            out_spec = outcomes.get(seed.metric_family, {})
            c = _build_candidate(
                serial, exp_name, seed.metric_family, method_name,
                exp_spec, out_spec, method_spec,
                registry, req, template, task_seed=seed,
            )
            if c is None:
                continue
            # Make candidate_id include the task so duplicates from
            # round-robin filling can be distinguished.
            c.candidate_id = f"llm_{serial:03d}"
            seen.add(key)
            candidates.append(c)
            serial += 1
        if len(candidates) >= req.max_candidates:
            break

    return candidates


def _template_uses_method_round_robin(template: dict) -> bool:
    return (
        template.get("kind") == "quasi_causal_research"
        or template.get("method_selection") == "round_robin"
    )


def _template_uses_method_compatibility(template: dict) -> bool:
    return template.get("method_selection") == "data_compatibility"


def compose_candidates(req: ComposeRequest) -> list[ComposedCandidate]:
    template = load_research_template(req.template_id)
    registry = SourceRegistry.load()

    # Training-research templates take a separate path that grounds the Y
    # axis in the user's domain_input via TaskSeedGenerator.
    if template.get("kind") == "training_research":
        return _compose_training_research(req, template, registry)

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
    use_method_compatibility = _template_uses_method_compatibility(template)

    def _add(exp_name: str, out_name: str, method_name: str) -> bool:
        nonlocal serial
        key = (
            (exp_name, out_name, "selected_method")
            if use_method_compatibility
            else (exp_name, out_name, method_name)
        )
        if key in seen:
            return False
        if exp_name not in eligible_exposures or out_name not in outcomes or method_name not in eligible_methods:
            return False
        c = _build_candidate(
            serial, exp_name, out_name, method_name,
            eligible_exposures[exp_name], outcomes[out_name], eligible_methods[method_name],
            registry, req, template,
        )
        if c is None:
            return False
        if use_method_compatibility:
            screening = select_methods_for_candidate(c, eligible_methods, registry)
            selected_method = screening.get("primary_method") or method_name
            selected_spec = eligible_methods.get(selected_method, eligible_methods[method_name])
            c = c.model_copy(
                update={
                    "method_template": selected_method,
                    "claim_strength": selected_spec.get(
                        "claim_strength",
                        template.get("default_claim_strength", "quasi_causal"),
                    ),
                    "key_threats": list(selected_spec.get("key_threats", [])),
                    "mitigations": dict(selected_spec.get("mitigations", {})),
                    "method_screening": screening,
                }
            )
        seen.add(key)
        candidates.append(c)
        serial += 1
        return True

    first_method_name = next(iter(eligible_methods), None)
    if first_method_name is None:
        return []
    method_names = (
        list(eligible_methods.keys())
        if _template_uses_method_round_robin(template)
        else [first_method_name]
    )
    if use_method_compatibility:
        method_names = [first_method_name]

    # Round 1: high-pass pairs. Quasi-causal templates use a diagonal
    # round-robin so the early pool covers both X/Y pairs and methods instead
    # of emitting one X/Y pair with every possible estimator first.
    if _template_uses_method_round_robin(template):
        high_pass_rounds = max(len(_HIGH_PASS_PAIRS), len(method_names))
        for i in range(high_pass_rounds):
            if len(candidates) >= req.max_candidates:
                break
            exp_name, out_name = _HIGH_PASS_PAIRS[i % len(_HIGH_PASS_PAIRS)]
            method_name = method_names[i % len(method_names)]
            _add(exp_name, out_name, method_name)
    else:
        for exp_name, out_name in _HIGH_PASS_PAIRS:
            if len(candidates) >= req.max_candidates:
                break
            _add(exp_name, out_name, first_method_name)

    # Round 2: ensure every eligible exposure family has at least one candidate
    outcome_list = list(outcomes.keys())
    for exp_name in eligible_exposures:
        if len(candidates) >= req.max_candidates:
            break
        for out_name in outcome_list:
            added = False
            for method_name in method_names:
                if _add(exp_name, out_name, method_name):
                    added = True
                    break
            if added:
                break  # one per family in this round

    # Round 3: fill remaining slots with remaining (exposure, outcome) combos,
    # cycling through outcomes first so coverage is spread across families
    for out_name in outcome_list:
        for exp_name in eligible_exposures:
            if len(candidates) >= req.max_candidates:
                break
            for method_name in method_names:
                if len(candidates) >= req.max_candidates:
                    break
                _add(exp_name, out_name, method_name)
        if len(candidates) >= req.max_candidates:
            break

    return candidates
