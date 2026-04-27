"""Flag classification and user-facing readiness computation for composed candidates.

Raw gate flags from precheck_candidate() and validate_candidate_export_contract()
are internal lint results. This module converts them into four tiers:

  INFO_FLAGS        — system working as expected; silent to users
  AUTO_FIXABLE_FLAGS — repair loop already fixed these; show as "auto-fixed"
  REVIEW_FLAGS      — needs human attention; show as action reasons
  BLOCKING_FLAGS    — prevents automated execution; always shown

The compute_candidate_readiness() function maps gate_status + repair_history
into a single readiness value:
  "ready"               — all pass, no fixes needed
  "ready_after_auto_fix"— repair loop applied fixes; result is clean
  "needs_review"        — shortlist_status="review"; human attention needed
  "blocked"             — shortlist_status="blocked" or blocking_reasons present
"""
from __future__ import annotations

INFO_FLAGS: frozenset[str] = frozenset({
    "source_alias_resolved",
    "non_canonical_source_name",
    "canonicalize_source_name",
})

AUTO_FIXABLE_FLAGS: frozenset[str] = frozenset({
    "missing_default_controls",
    "missing_boundary_source",
    "missing_identification_threats",
    "aggregation_required",
})

REVIEW_FLAGS: frozenset[str] = frozenset({
    "manual_download_required",
    "semi_automated_source",
    "aggregation_plan_required",
    "experimental_source_in_use",
    "experimental_source_requires_key",
    "causal_assumption_weak",
    "time_overlap_insufficient",
})

BLOCKING_FLAGS: frozenset[str] = frozenset({
    "source_not_in_registry",
    "missing_exposure_role_source",
    "missing_outcome_role_source",
    "missing_machine_readable_source",
    "missing_join_path",
    "paid_source_not_allowed",
    "required_secrets_blocks_ready",
    "high_automation_risk_blocks_ready",
    "missing_threat_mitigation",
    "threat_mitigation_coverage_low",  # colon-suffixed variant from precheck
})

# Flags that directly indicate a data-path failure
_DATA_FAIL_FLAGS: frozenset[str] = frozenset({
    "source_not_in_registry",
    "missing_exposure_role_source",
    "missing_outcome_role_source",
    "missing_machine_readable_source",
    "missing_join_path",
})


def classify_flags(raw_flags: list[str]) -> dict:
    """Classify raw gate flags into four user-facing tiers.

    Unknown flags (not in any set) fall into review_reasons so they
    surface rather than being silently dropped.
    """
    info_notes: list[str] = []
    auto_fixes: list[str] = []
    review_reasons: list[str] = []
    blocking_reasons: list[str] = []

    for flag in raw_flags:
        # Strip any colon-suffixed detail (e.g. "threat_mitigation_coverage_low:40%")
        base = flag.split(":")[0]
        if base in INFO_FLAGS or flag in INFO_FLAGS:
            info_notes.append(flag)
        elif base in AUTO_FIXABLE_FLAGS or flag in AUTO_FIXABLE_FLAGS:
            auto_fixes.append(flag)
        elif base in REVIEW_FLAGS or flag in REVIEW_FLAGS:
            review_reasons.append(flag)
        elif base in BLOCKING_FLAGS or flag in BLOCKING_FLAGS:
            blocking_reasons.append(flag)
        else:
            review_reasons.append(flag)

    return {
        "info_notes": info_notes,
        "auto_fixes": auto_fixes,
        "review_reasons": review_reasons,
        "blocking_reasons": blocking_reasons,
    }


def compute_candidate_readiness(
    candidate_dict: dict,
    gate_status: dict,
    repair_history: list[dict],
) -> dict:
    """Compute a structured user-facing readiness summary.

    Args:
        candidate_dict:  ComposedCandidate.model_dump() or equivalent dict.
        gate_status:     Final gate status from validate_candidate_export_contract().
        repair_history:  Repair entries from repair_candidate().

    Returns a dict with keys:
        readiness              : "ready" | "ready_after_auto_fix" | "needs_review" | "blocked"
        data_status            : "ok" | "failed"
        automation_status      : "full" | "partial" | "blocked"
        identification_status  : descriptive string
        auto_fix_actions       : list of action names applied by the repair loop
        user_visible_reasons   : blocking + review reasons only (no info/auto-fix noise)
        debug_flags            : raw reasons list for diagnostic traces
    """
    blocking = list(gate_status.get("blocking_reasons") or [])
    shortlist = gate_status.get("shortlist_status", "blocked")

    auto_fix_actions = [
        h["action"] for h in repair_history
        if h.get("result") in ("repaired", "normalized", "canonicalized")
    ]

    if blocking or shortlist == "blocked":
        readiness = "blocked"
    elif shortlist == "review":
        readiness = "needs_review"
    elif auto_fix_actions:
        readiness = "ready_after_auto_fix"
    else:
        readiness = "ready"

    # Classify raw reasons for the user_visible_reasons field
    raw_reasons: list[str] = gate_status.get("reasons") or []
    classified = classify_flags(raw_reasons)
    user_visible_reasons = classified["blocking_reasons"] + classified["review_reasons"]

    # Identification quality
    claim_strength = candidate_dict.get("claim_strength", "associational")
    threats = list(candidate_dict.get("key_threats") or [])
    mitigations = dict(candidate_dict.get("mitigations") or {})
    threats_complete = len(threats) >= 3 and all(t in mitigations for t in threats)

    if claim_strength == "causal":
        identification_status = (
            "documented_causal" if threats_complete else "causal_claim_underdocumented"
        )
    else:
        identification_status = (
            "documented_associational" if threats_complete else "associational_threats_partial"
        )

    # Automation reachability
    risk = candidate_dict.get("automation_risk", "low")
    required_secrets = (
        gate_status.get("required_secrets")
        or candidate_dict.get("required_secrets")
        or []
    )
    if blocking or shortlist == "blocked" or risk == "high":
        automation_status = "blocked"
    elif risk == "medium" or required_secrets:
        automation_status = "partial"
    else:
        automation_status = "full"

    # Data path health
    data_ok = not any(
        (flag.split(":")[0] in _DATA_FAIL_FLAGS or flag in _DATA_FAIL_FLAGS)
        for flag in raw_reasons
    )

    return {
        "readiness": readiness,
        "data_status": "ok" if data_ok else "failed",
        "automation_status": automation_status,
        "identification_status": identification_status,
        "auto_fix_actions": auto_fix_actions,
        "user_visible_reasons": user_visible_reasons,
        "debug_flags": raw_reasons,
    }
