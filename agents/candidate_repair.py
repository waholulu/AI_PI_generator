"""Deterministic repair loop for composed candidates (Step 3).

After precheck_candidate() identifies warnings or failures, repair_candidate()
applies deterministic fixes and re-runs precheck (up to max_rounds times).

Repair rules (applied in a single round):
  canonicalize_source_name         — resolve aliases to canonical registry IDs
  replace_exposure_source_from_template — swap unknown/wrong-role exposure source
  replace_outcome_source_from_template  — swap unknown/wrong-role outcome source
  add_boundary_source_tiger_lines  — fill empty boundary_source with TIGER_Lines
  add_default_controls             — fill empty controls with ACS
  add_aggregation_plan             — document spatial aggregation step in join_plan
  fill_threats_from_method_template — fill key_threats + mitigations from method

Invariants:
  - Paid-API sources (cost_required=True) are never promoted to 'ready' ('blocked')
  - High automation_risk candidates stay 'blocked'
  - Auth-required sources are documented in required_secrets; status <= 'review'
  - repair_history entries are appended for every action taken, including blocks
"""
from __future__ import annotations

import copy

from agents.candidate_feasibility import precheck_candidate
from agents.identification_template_filler import fill_identification_from_method
from agents.logging_config import get_logger
from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposedCandidate

logger = get_logger(__name__)

# ── Static lookup tables ──────────────────────────────────────────────────────

# Stable (non-experimental) canonical fallback per exposure family.
# None means no stable fallback exists (stays blocked).
_EXPOSURE_FALLBACK: dict[str, str | None] = {
    "walkability": "EPA_National_Walkability_Index",
    "street_network": "OSMnx_OpenStreetMap",
    "streetview_built_form": None,   # experimental only; no stable fallback
    "greenery_visibility": None,     # experimental only
}

_OUTCOME_FALLBACK: dict[str, str] = {
    "physical_inactivity": "CDC_PLACES",
    "obesity": "CDC_PLACES",
    "diabetes": "CDC_PLACES",
    "asthma": "CDC_PLACES",
    "poor_mental_health": "CDC_PLACES",
}
_DEFAULT_OUTCOME_FALLBACK = "CDC_PLACES"


# ── History entry builder ─────────────────────────────────────────────────────

def _entry(
    candidate_id: str,
    round_num: int,
    issue: str,
    action: str,
    before: dict,
    after: dict,
    result: str,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "round": round_num,
        "issue": issue,
        "action": action,
        "before": before,
        "after": after,
        "result": result,
    }


# ── Step 0: Canonicalize source names ────────────────────────────────────────

def _canonicalize(
    candidate: ComposedCandidate,
    registry: SourceRegistry,
) -> tuple[ComposedCandidate, list[dict]]:
    """Resolve source aliases to canonical registry IDs. Applied before repair rounds."""
    cid = candidate.candidate_id
    updates: dict = {}
    history: list[dict] = []

    exp_sid = registry.resolve(candidate.exposure_source)
    if exp_sid and exp_sid != candidate.exposure_source:
        history.append(_entry(
            cid, 0, "non_canonical_source_name", "canonicalize_source_name",
            {"exposure_source": candidate.exposure_source},
            {"exposure_source": exp_sid},
            "canonicalized",
        ))
        updates["exposure_source"] = exp_sid

    out_sid = registry.resolve(candidate.outcome_source)
    if out_sid and out_sid != candidate.outcome_source:
        history.append(_entry(
            cid, 0, "non_canonical_source_name", "canonicalize_source_name",
            {"outcome_source": candidate.outcome_source},
            {"outcome_source": out_sid},
            "canonicalized",
        ))
        updates["outcome_source"] = out_sid

    old_jp = candidate.join_plan
    new_jp = copy.deepcopy(old_jp)
    jp_changed = False

    old_boundary = old_jp.get("boundary_source", [])
    new_boundary = [
        (lambda s: s if s and s == b else b)(registry.resolve(b))
        for b in old_boundary
    ]
    # Rewrite: resolve each, keep original if unresolved
    new_boundary = []
    for b in old_boundary:
        sid = registry.resolve(b)
        new_boundary.append(sid if (sid and sid != b) else b)
    if new_boundary != old_boundary:
        new_jp["boundary_source"] = new_boundary
        jp_changed = True

    old_controls = old_jp.get("controls", [])
    new_controls = []
    for c in old_controls:
        sid = registry.resolve(c)
        new_controls.append(sid if (sid and sid != c) else c)
    if new_controls != old_controls:
        new_jp["controls"] = new_controls
        jp_changed = True

    if jp_changed:
        updates["join_plan"] = new_jp
        history.append(_entry(
            cid, 0, "non_canonical_source_name", "canonicalize_source_name",
            {
                "join_plan_sources": {
                    "boundary_source": old_boundary,
                    "controls": old_controls,
                }
            },
            {
                "join_plan_sources": {
                    "boundary_source": new_boundary,
                    "controls": new_controls,
                }
            },
            "canonicalized",
        ))

    if updates:
        candidate = candidate.model_copy(update=updates)
    return candidate, history


# ── Repair round ──────────────────────────────────────────────────────────────

def _repair_round(
    candidate: ComposedCandidate,
    gate_status: dict,
    registry: SourceRegistry,
    round_num: int,
) -> tuple[ComposedCandidate, list[dict]]:
    """Apply all applicable deterministic repairs for one round.

    Returns the (possibly updated) candidate and a list of history entries for
    this round. An empty history list means nothing changed.
    """
    cid = candidate.candidate_id
    subchecks = gate_status.get("subchecks", {})
    reasons = gate_status.get("reasons", [])
    history: list[dict] = []
    updates: dict = {}

    # Shared mutable join_plan copy; all join_plan edits go through new_jp.
    new_jp = copy.deepcopy(candidate.join_plan)
    jp_changed = False

    # ── 1. Replace missing/unknown exposure source ────────────────────────────
    if (
        subchecks.get("source_exists") == "fail"
        and "source_not_in_registry" in reasons
        and registry.resolve(candidate.exposure_source) is None
    ):
        fallback = _EXPOSURE_FALLBACK.get(candidate.exposure_family)
        if fallback:
            history.append(_entry(
                cid, round_num, "exposure_source_not_found",
                "replace_exposure_source_from_template",
                {"exposure_source": candidate.exposure_source},
                {"exposure_source": fallback},
                "repaired",
            ))
            updates["exposure_source"] = fallback

    # ── 2. Replace missing/unknown outcome source ─────────────────────────────
    if (
        subchecks.get("source_exists") == "fail"
        and "source_not_in_registry" in reasons
        and registry.resolve(candidate.outcome_source) is None
    ):
        fallback = _OUTCOME_FALLBACK.get(
            candidate.outcome_family, _DEFAULT_OUTCOME_FALLBACK
        )
        history.append(_entry(
            cid, round_num, "outcome_source_not_found",
            "replace_outcome_source_from_template",
            {"outcome_source": candidate.outcome_source},
            {"outcome_source": fallback},
            "repaired",
        ))
        updates["outcome_source"] = fallback

    # ── 3. Replace exposure source with wrong role ────────────────────────────
    if (
        subchecks.get("role_coverage") == "fail"
        and "missing_exposure_role_source" in reasons
    ):
        current_exp = updates.get("exposure_source", candidate.exposure_source)
        fallback = _EXPOSURE_FALLBACK.get(candidate.exposure_family)
        if fallback and fallback != current_exp:
            history.append(_entry(
                cid, round_num, "missing_exposure_role_source",
                "replace_exposure_source_from_template",
                {"exposure_source": current_exp},
                {"exposure_source": fallback},
                "repaired",
            ))
            updates["exposure_source"] = fallback

    # ── 4. Replace outcome source with wrong role ─────────────────────────────
    if (
        subchecks.get("role_coverage") == "fail"
        and "missing_outcome_role_source" in reasons
    ):
        current_out = updates.get("outcome_source", candidate.outcome_source)
        fallback = _OUTCOME_FALLBACK.get(
            candidate.outcome_family, _DEFAULT_OUTCOME_FALLBACK
        )
        if fallback != current_out:
            history.append(_entry(
                cid, round_num, "missing_outcome_role_source",
                "replace_outcome_source_from_template",
                {"outcome_source": current_out},
                {"outcome_source": fallback},
                "repaired",
            ))
            updates["outcome_source"] = fallback

    # ── 5. Add TIGER_Lines — gate-driven (spatial_join_path = fail) ──────────
    if (
        subchecks.get("spatial_join_path") == "fail"
        and "missing_join_path" in reasons
    ):
        old_boundary = new_jp.get("boundary_source", [])
        if "TIGER_Lines" not in old_boundary:
            new_boundary = old_boundary + ["TIGER_Lines"]
            history.append(_entry(
                cid, round_num, "missing_boundary_source",
                "add_boundary_source_tiger_lines",
                {"boundary_source": old_boundary},
                {"boundary_source": new_boundary},
                "repaired",
            ))
            new_jp["boundary_source"] = new_boundary
            jp_changed = True

    # ── 6. Add TIGER_Lines — proactive (boundary_source field is empty) ───────
    elif not new_jp.get("boundary_source"):
        history.append(_entry(
            cid, round_num, "missing_boundary_source",
            "add_boundary_source_tiger_lines",
            {"boundary_source": []},
            {"boundary_source": ["TIGER_Lines"]},
            "normalized",
        ))
        new_jp["boundary_source"] = ["TIGER_Lines"]
        jp_changed = True

    # ── 7. Add ACS controls — proactive (controls field is empty) ─────────────
    if not new_jp.get("controls"):
        history.append(_entry(
            cid, round_num, "missing_control_source",
            "add_default_controls",
            {"controls": []},
            {"controls": ["ACS"]},
            "normalized",
        ))
        new_jp["controls"] = ["ACS"]
        jp_changed = True

    # ── 8. Document aggregation plan (warning, not fixable but documentable) ──
    if (
        subchecks.get("spatial_join_path") == "warning"
        and any("aggregation_required" in r for r in reasons)
        and "aggregation_plan" not in new_jp
    ):
        agg_plan = f"aggregate_from_source_resolution_to_{candidate.unit_of_analysis}"
        history.append(_entry(
            cid, round_num, "aggregation_required",
            "add_aggregation_plan",
            {"aggregation_plan": None},
            {"aggregation_plan": agg_plan},
            "warning_accepted",
        ))
        new_jp["aggregation_plan"] = agg_plan
        jp_changed = True

    # ── 9. Fill identification threats from method template ───────────────────
    if subchecks.get("identification_threats") == "warning":
        new_threats, new_mits = fill_identification_from_method(candidate)
        if new_threats != list(candidate.key_threats) or new_mits != dict(candidate.mitigations):
            history.append(_entry(
                cid, round_num, "missing_identification_threats",
                "fill_threats_from_method_template",
                {
                    "key_threats": list(candidate.key_threats),
                    "mitigations": dict(candidate.mitigations),
                },
                {
                    "key_threats": new_threats,
                    "mitigations": new_mits,
                },
                "repaired",
            ))
            updates["key_threats"] = new_threats
            updates["mitigations"] = new_mits

    if jp_changed:
        updates["join_plan"] = new_jp

    if updates:
        candidate = candidate.model_copy(update=updates)
    return candidate, history


# ── Post-repair: document irreparable constraints ─────────────────────────────

def _document_irrepairables(
    candidate: ComposedCandidate,
    registry: SourceRegistry,
) -> list[dict]:
    """Return history entries for sources that cannot be automatically repaired.

    Covers paid APIs and auth-required experimental sources, so the caller can
    see WHY a candidate stays blocked/review even after repair rounds complete.
    """
    cid = candidate.candidate_id
    entries: list[dict] = []
    seen: set[str] = set()

    all_src_names = [
        candidate.exposure_source,
        candidate.outcome_source,
        *candidate.join_plan.get("boundary_source", []),
        *candidate.join_plan.get("controls", []),
    ]
    for name in all_src_names:
        sid = registry.resolve(name)
        if not sid or sid in seen:
            continue
        seen.add(sid)
        spec = registry.sources.get(sid, {})

        if spec.get("cost_required"):
            entries.append(_entry(
                cid, 0, "paid_source_not_allowed",
                "keep_blocked",
                {"source": sid, "cost_required": True},
                {"shortlist_status": "blocked"},
                "blocked",
            ))
        elif spec.get("auth_required"):
            entries.append(_entry(
                cid, 0, "experimental_source_requires_key",
                "mark_high_risk",
                {"source": sid, "auth_required": True},
                {"shortlist_status": "review"},
                "review",
            ))

    return entries


# ── Shortlist status resolution ───────────────────────────────────────────────

def _compute_shortlist_status(
    candidate: ComposedCandidate,
    gate_status: dict,
    registry: SourceRegistry,
) -> tuple[str, list[str]]:
    """Return (shortlist_status, required_secrets) applying override rules.

    Rules (in priority order):
      overall=fail OR paid_api OR automation_risk=high  → blocked
      overall=warning OR automation_risk=medium OR auth_required  → review
      overall=pass AND no cost/auth AND risk=low         → ready
    """
    all_src_names = [
        candidate.exposure_source,
        candidate.outcome_source,
        *candidate.join_plan.get("boundary_source", []),
        *candidate.join_plan.get("controls", []),
    ]

    has_paid = False
    required_secrets: list[str] = []
    seen: set[str] = set()
    for name in all_src_names:
        sid = registry.resolve(name)
        if not sid or sid in seen:
            continue
        seen.add(sid)
        spec = registry.sources.get(sid, {})
        if spec.get("cost_required"):
            has_paid = True
            required_secrets.append(f"{sid}:api_key")
        elif spec.get("auth_required"):
            required_secrets.append(f"{sid}:api_key")

    overall = gate_status["overall"]
    automation_risk = candidate.automation_risk

    if overall == "fail" or has_paid or automation_risk == "high":
        shortlist_status = "blocked"
    elif overall == "warning" or automation_risk == "medium" or required_secrets:
        shortlist_status = "review"
    else:
        shortlist_status = "ready"

    return shortlist_status, required_secrets


# ── Public API ────────────────────────────────────────────────────────────────

def repair_candidate(
    candidate: ComposedCandidate,
    gate_status: dict,
    max_rounds: int = 2,
) -> tuple[ComposedCandidate, dict, list[dict]]:
    """Apply deterministic repairs to a candidate with gate warnings or failures.

    Args:
        candidate:   The ComposedCandidate to repair.
        gate_status: Initial gate_status dict from precheck_candidate().
        max_rounds:  Maximum repair rounds (default 2).

    Returns:
        repaired_candidate — ComposedCandidate with all repairs applied
        new_gate_status    — Updated dict with overall/subchecks/reasons/
                             repair_suggestions/shortlist_status/required_secrets
        repair_history     — Ordered list of repair action dicts
    """
    registry = SourceRegistry.load()
    repair_history: list[dict] = []

    # Step 0: Canonicalize source names (non-destructive normalization)
    candidate, canon_repairs = _canonicalize(candidate, registry)
    repair_history.extend(canon_repairs)

    # Re-run precheck after canonicalization (aliases may have changed verdict)
    current_status = precheck_candidate(candidate)

    # Repair rounds
    for round_num in range(1, max_rounds + 1):
        candidate, round_repairs = _repair_round(
            candidate, current_status, registry, round_num
        )
        repair_history.extend(round_repairs)

        if not round_repairs:
            break  # converged; current_status is still valid

        current_status = precheck_candidate(candidate)

    # Document paid/auth constraints that remain after all repairs
    repair_history.extend(_document_irrepairables(candidate, registry))

    # Override shortlist_status with final rules
    shortlist_status, required_secrets = _compute_shortlist_status(
        candidate, current_status, registry
    )
    final_status = {
        **current_status,
        "shortlist_status": shortlist_status,
        "required_secrets": required_secrets,
    }

    logger.debug(
        "repair_candidate %s → overall=%s shortlist=%s repairs=%d",
        candidate.candidate_id,
        final_status["overall"],
        shortlist_status,
        len(repair_history),
    )

    return candidate, final_status, repair_history
