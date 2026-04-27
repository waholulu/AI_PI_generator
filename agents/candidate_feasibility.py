"""Deterministic feasibility precheck for composed candidates (role-based G3 path).

This module implements the **candidate factory** data-availability gate (G3).
It is distinct from the legacy LLM ideation G3 in ``agents/rule_engine.py``
(``check_G3_data_availability``), which checks each topic against the catalog
without role awareness.

Two G3 paths:
  Legacy LLM ideation path → ``rule_engine.check_G3_data_availability``
  Candidate factory path   → ``precheck_candidate`` here (role-based G3)

Always call ``validate_candidate_export_contract`` after precheck — that is the
final strict gate that enforces claude_code_ready policy.

Subchecks run in order:
  source_exists              — exposure and outcome sources resolve in source_registry
  role_coverage              — exposure source has "exposure" role; outcome has "outcome"
  machine_readable           — all declared sources are machine-readable
  spatial_join_path          — a boundary/join source is available; warns if aggregation needed
  time_overlap               — source coverage years overlap the plan's time window
  cloud_automation_feasibility — no experimental sources with auth/cost requirements
  identification_threats     — key_threats non-empty and mitigations cover ≥ 80 %

Overall verdict:
  pass    → all subchecks pass   → shortlist_status = "ready"
  warning → ≥1 warning, no fail  → shortlist_status = "review"
  fail    → ≥1 fail              → shortlist_status = "blocked"
"""

from __future__ import annotations

from agents.logging_config import get_logger
from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposedCandidate

logger = get_logger(__name__)


def _normalize_unit(unit: str) -> str:
    """Strip common census-prefixes so 'census_tract' == 'tract'."""
    return unit.lower().strip().removeprefix("census_")


def _check_aggregation(
    candidate: ComposedCandidate,
    exp_spec: dict,
) -> tuple[str, list[str], list[str]]:
    """Return (status, reasons, repair_suggestions) for spatial aggregation requirement.

    Only activates when the source explicitly declares aggregation_allowed_to, which
    indicates it has data at a fixed spatial resolution (e.g., block_group).
    Sources without this field (e.g., OSMnx, TIGER) are queryable at any scale.
    """
    # aggregation_allowed_to may appear at the top level of the spec or nested
    # inside variable_families.<family_name> (as in source_capabilities.yaml).
    agg_allowed_raw: list[str] = exp_spec.get("aggregation_allowed_to", [])
    if not agg_allowed_raw:
        vf = exp_spec.get("variable_families") or {}
        fam_spec = vf.get(candidate.exposure_family) or {}
        if isinstance(fam_spec, dict):
            agg_allowed_raw = fam_spec.get("aggregation_allowed_to", [])
    if not agg_allowed_raw:
        return "pass", [], []

    source_units = [u.lower() for u in exp_spec.get("spatial_units", [])]
    target_unit = _normalize_unit(candidate.unit_of_analysis)
    normalized_source_units = [_normalize_unit(u) for u in source_units]

    if target_unit in normalized_source_units:
        return "pass", [], []

    normalized_agg = [_normalize_unit(u) for u in agg_allowed_raw]
    if target_unit in normalized_agg:
        source_label = source_units[0] if source_units else "source_unit"
        reason = f"{source_label}_to_{candidate.unit_of_analysis}_aggregation_required"
        return "warning", [reason], ["add_aggregation_plan"]

    reason = (
        f"spatial_unit_mismatch:{source_units}_cannot_serve_{target_unit}"
    )
    return "warning", [reason], []


def precheck_candidate(candidate: ComposedCandidate) -> dict:
    """Run deterministic feasibility checks on a ComposedCandidate.

    Returns a gate_status dict:
      overall          : "pass" | "warning" | "fail"
      subchecks        : {subcheck_name: "pass" | "warning" | "fail"}
      reasons          : [str]
      repair_suggestions : [str]  (deduplicated, order-preserving)
      shortlist_status : "ready" | "review" | "blocked"
    """
    registry = SourceRegistry.load()

    exp_source = candidate.exposure_source
    out_source = candidate.outcome_source
    boundary_sources: list[str] = candidate.join_plan.get("boundary_source", [])
    control_sources: list[str] = candidate.join_plan.get("controls", [])

    subchecks: dict[str, str] = {}
    reasons: list[str] = []
    repairs: list[str] = []

    # ── 1. source_exists ──────────────────────────────────────────────────────
    exp_sid = registry.resolve(exp_source)
    out_sid = registry.resolve(out_source)
    missing_core = []
    if exp_sid is None:
        missing_core.append(exp_source)
        reasons.append("source_not_in_registry")
        repairs.append("replace_exposure_source_from_template")
    if out_sid is None:
        missing_core.append(out_source)
        reasons.append("source_not_in_registry")
        repairs.append("replace_outcome_source_from_template")
    subchecks["source_exists"] = "fail" if missing_core else "pass"

    exp_spec = registry.sources.get(exp_sid, {}) if exp_sid else {}
    out_spec = registry.sources.get(out_sid, {}) if out_sid else {}

    # ── 2. role_coverage ─────────────────────────────────────────────────────
    # Checks that the exposure source carries the "exposure" role and the
    # outcome source carries the "outcome" role. Family names are template
    # concepts and may diverge from registry keys, so only roles are checked.
    role_fails = []
    if not ("exposure" in exp_spec.get("roles", [])):
        role_fails.append("missing_exposure_role_source")
        repairs.append("replace_exposure_source_from_template")
    if not ("outcome" in out_spec.get("roles", [])):
        role_fails.append("missing_outcome_role_source")
        repairs.append("replace_outcome_source_from_template")
    if role_fails:
        subchecks["role_coverage"] = "fail"
        reasons.extend(role_fails)
    else:
        subchecks["role_coverage"] = "pass"

    # ── 3. machine_readable ──────────────────────────────────────────────────
    all_src_names = [exp_source, out_source] + boundary_sources + control_sources
    not_readable: list[str] = []
    for src_name in all_src_names:
        sid = registry.resolve(src_name)
        if sid and not registry.sources.get(sid, {}).get("machine_readable", False):
            not_readable.append(src_name)
    if not_readable:
        subchecks["machine_readable"] = "fail"
        reasons.append("missing_machine_readable_source")
    else:
        subchecks["machine_readable"] = "pass"

    # ── 4. spatial_join_path ─────────────────────────────────────────────────
    boundary_sids = [registry.resolve(b) for b in boundary_sources]
    boundary_specs = [registry.sources.get(sid, {}) for sid in boundary_sids if sid]

    has_explicit_boundary = any("boundary" in s.get("roles", []) for s in boundary_specs)
    # CDC_PLACES, ACS, and TIGER carry built-in geographic keys usable as join anchors
    implicit_anchor_sids = {"TIGER_Lines", "ACS", "CDC_PLACES"}
    has_implicit_boundary = any(
        sid in implicit_anchor_sids
        for sid in ([exp_sid, out_sid] + [s for s in boundary_sids if s])
        if sid
    )

    if not (has_explicit_boundary or has_implicit_boundary):
        subchecks["spatial_join_path"] = "fail"
        reasons.append("missing_join_path")
        repairs.append("add_boundary_source_tiger_lines")
    else:
        agg_status, agg_reasons, agg_repairs = _check_aggregation(candidate, exp_spec)
        subchecks["spatial_join_path"] = agg_status
        reasons.extend(agg_reasons)
        repairs.extend(agg_repairs)

    # ── 5. time_overlap ──────────────────────────────────────────────────────
    # Warn when the source's declared coverage years don't include the template
    # default time window (2016-2024).  This is a soft warning — many sources
    # cover longer windows than listed in the registry.
    exp_start = exp_spec.get("coverage_start_year")
    exp_end = exp_spec.get("coverage_end_year")
    PLAN_START, PLAN_END = 2016, 2024
    if exp_start and exp_end:
        try:
            if int(exp_end) < PLAN_START or int(exp_start) > PLAN_END:
                subchecks["time_overlap"] = "warning"
                reasons.append("time_overlap_insufficient")
            else:
                subchecks["time_overlap"] = "pass"
        except (ValueError, TypeError):
            subchecks["time_overlap"] = "pass"
    else:
        subchecks["time_overlap"] = "pass"

    # ── 6. cloud_automation_feasibility ──────────────────────────────────────
    all_specs: list[dict] = [exp_spec, out_spec] + boundary_specs
    for ctrl in control_sources:
        sid = registry.resolve(ctrl)
        if sid:
            all_specs.append(registry.sources.get(sid, {}))

    experimental_with_cost = any(
        s.get("tier") == "experimental" and (s.get("auth_required") or s.get("cost_required"))
        for s in all_specs if s
    )
    experimental_only = any(s.get("tier") == "experimental" for s in all_specs if s)

    if experimental_with_cost:
        subchecks["cloud_automation_feasibility"] = "warning"
        reasons.append("experimental_source_requires_key")
    elif experimental_only:
        subchecks["cloud_automation_feasibility"] = "warning"
        reasons.append("experimental_source_in_use")
    else:
        subchecks["cloud_automation_feasibility"] = "pass"

    # ── 6. identification_threats ─────────────────────────────────────────────
    threats = candidate.key_threats
    mitigations = candidate.mitigations
    if not threats:
        subchecks["identification_threats"] = "warning"
        reasons.append("missing_identification_threats")
    else:
        covered = len(set(threats) & set(mitigations.keys()))
        ratio = covered / len(threats)
        if ratio >= 0.8:
            subchecks["identification_threats"] = "pass"
        else:
            subchecks["identification_threats"] = "warning"
            reasons.append(f"threat_mitigation_coverage_low:{ratio:.0%}")

    # ── Overall verdict ───────────────────────────────────────────────────────
    has_fail = any(v == "fail" for v in subchecks.values())
    has_warn = any(v == "warning" for v in subchecks.values())

    if has_fail:
        overall = "fail"
        shortlist_status = "blocked"
    elif has_warn:
        overall = "warning"
        shortlist_status = "review"
    else:
        overall = "pass"
        shortlist_status = "ready"

    # Deduplicate repair suggestions while preserving order
    seen: set[str] = set()
    deduped_repairs: list[str] = []
    for r in repairs:
        if r not in seen:
            seen.add(r)
            deduped_repairs.append(r)

    logger.debug(
        "precheck_candidate %s → %s (subchecks=%s)",
        candidate.candidate_id, overall, subchecks,
    )

    return {
        "gate_id": "G3_candidate",
        "passed": overall != "fail",
        "overall": overall,
        "subchecks": subchecks,
        "reasons": reasons,
        "repair_suggestions": deduped_repairs,
        "shortlist_status": shortlist_status,
    }
