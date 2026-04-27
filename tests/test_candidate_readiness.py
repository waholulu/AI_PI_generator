"""Tests for candidate_flag_classifier: classify_flags() and compute_candidate_readiness().

Verifies that:
  - source_alias_resolved and related canonicalization flags are classified as info_notes
  - missing_identification_threats is auto_fixable, not blocking
  - blocking flags (missing_exposure_role_source etc.) land in blocking_reasons
  - compute_candidate_readiness() correctly maps gate_status + repair_history → readiness
"""
from __future__ import annotations

import pytest

from agents.candidate_flag_classifier import (
    classify_flags,
    compute_candidate_readiness,
)


# ── classify_flags ─────────────────────────────────────────────────────────────

def test_source_alias_flags_are_info() -> None:
    """Canonicalization flags must never surface as review or blocking."""
    result = classify_flags([
        "source_alias_resolved",
        "non_canonical_source_name",
        "canonicalize_source_name",
    ])
    assert result["info_notes"] == [
        "source_alias_resolved",
        "non_canonical_source_name",
        "canonicalize_source_name",
    ]
    assert result["blocking_reasons"] == []
    assert result["review_reasons"] == []
    assert result["auto_fixes"] == []


def test_missing_identification_threats_is_auto_fixable() -> None:
    """missing_identification_threats should be auto_fixable, not blocking."""
    result = classify_flags(["missing_identification_threats"])
    assert result["auto_fixes"] == ["missing_identification_threats"]
    assert result["blocking_reasons"] == []
    assert result["review_reasons"] == []


def test_blocking_flags_land_in_blocking() -> None:
    """Hard-blocking flags must appear in blocking_reasons."""
    flags = [
        "missing_exposure_role_source",
        "missing_outcome_role_source",
        "source_not_in_registry",
        "missing_machine_readable_source",
        "missing_join_path",
        "paid_source_not_allowed",
        "required_secrets_blocks_ready",
        "high_automation_risk_blocks_ready",
    ]
    result = classify_flags(flags)
    assert set(result["blocking_reasons"]) == set(flags)
    assert result["review_reasons"] == []
    assert result["info_notes"] == []
    assert result["auto_fixes"] == []


def test_review_flags_land_in_review() -> None:
    """Review-tier flags must appear in review_reasons."""
    flags = [
        "experimental_source_in_use",
        "time_overlap_insufficient",
        "experimental_source_requires_key",
    ]
    result = classify_flags(flags)
    assert set(result["review_reasons"]) == set(flags)
    assert result["blocking_reasons"] == []


def test_threat_mitigation_coverage_low_is_blocking() -> None:
    """threat_mitigation_coverage_low:40% has a colon suffix; base flag is in BLOCKING_FLAGS."""
    result = classify_flags(["threat_mitigation_coverage_low:40%"])
    assert result["blocking_reasons"] == ["threat_mitigation_coverage_low:40%"]


def test_unknown_flag_falls_into_review() -> None:
    """An unrecognised flag must be surfaced in review_reasons rather than silently dropped."""
    result = classify_flags(["some_future_flag_unknown"])
    assert result["review_reasons"] == ["some_future_flag_unknown"]
    assert result["blocking_reasons"] == []


def test_mixed_flags_classified_correctly() -> None:
    result = classify_flags([
        "non_canonical_source_name",          # info
        "missing_identification_threats",      # auto_fix
        "experimental_source_in_use",          # review
        "missing_exposure_role_source",        # blocking
    ])
    assert "non_canonical_source_name" in result["info_notes"]
    assert "missing_identification_threats" in result["auto_fixes"]
    assert "experimental_source_in_use" in result["review_reasons"]
    assert "missing_exposure_role_source" in result["blocking_reasons"]


# ── compute_candidate_readiness ───────────────────────────────────────────────

def _candidate(
    claim_strength: str = "associational",
    automation_risk: str = "low",
    key_threats: list | None = None,
    mitigations: dict | None = None,
    required_secrets: list | None = None,
) -> dict:
    return {
        "claim_strength": claim_strength,
        "automation_risk": automation_risk,
        "key_threats": key_threats if key_threats is not None else [
            "socioeconomic_confounding",
            "residential_self_selection",
            "spatial_autocorrelation",
        ],
        "mitigations": mitigations if mitigations is not None else {
            "socioeconomic_confounding": "ACS controls",
            "residential_self_selection": "county fixed effects",
            "spatial_autocorrelation": "clustered SEs",
        },
        "required_secrets": required_secrets or [],
    }


def _gate(
    shortlist: str = "ready",
    blocking_reasons: list | None = None,
    reasons: list | None = None,
    required_secrets: list | None = None,
) -> dict:
    return {
        "shortlist_status": shortlist,
        "blocking_reasons": blocking_reasons or [],
        "repairable_warnings": [],
        "reasons": reasons or [],
        "required_secrets": required_secrets or [],
    }


def test_readiness_ready_no_repairs() -> None:
    """Clean candidate with no repairs and no issues → readiness=ready."""
    result = compute_candidate_readiness(_candidate(), _gate(), repair_history=[])
    assert result["readiness"] == "ready"
    assert result["automation_status"] == "full"
    assert result["identification_status"] == "documented_associational"
    assert result["data_status"] == "ok"
    assert result["auto_fix_actions"] == []
    assert result["user_visible_reasons"] == []


def test_readiness_ready_after_auto_fix() -> None:
    """Candidate with repair actions but clean final status → ready_after_auto_fix."""
    repair_history = [
        {"action": "fill_threats_from_method_template", "result": "repaired"},
        {"action": "add_boundary_source_tiger_lines", "result": "normalized"},
    ]
    result = compute_candidate_readiness(_candidate(), _gate(), repair_history)
    assert result["readiness"] == "ready_after_auto_fix"
    assert set(result["auto_fix_actions"]) == {
        "fill_threats_from_method_template",
        "add_boundary_source_tiger_lines",
    }


def test_readiness_needs_review() -> None:
    """shortlist=review → needs_review regardless of repair history."""
    result = compute_candidate_readiness(
        _candidate(),
        _gate(
            shortlist="review",
            reasons=["experimental_source_in_use"],
        ),
        repair_history=[],
    )
    assert result["readiness"] == "needs_review"
    assert "experimental_source_in_use" in result["user_visible_reasons"]


def test_readiness_blocked_by_blocking_reasons() -> None:
    """Non-empty blocking_reasons → blocked, with reasons surfaced to user."""
    result = compute_candidate_readiness(
        _candidate(),
        _gate(
            shortlist="blocked",
            blocking_reasons=["missing_exposure_role_source"],
            reasons=["missing_exposure_role_source"],
        ),
        repair_history=[],
    )
    assert result["readiness"] == "blocked"
    assert result["data_status"] == "failed"
    assert "missing_exposure_role_source" in result["user_visible_reasons"]


def test_readiness_blocked_by_shortlist_status() -> None:
    """shortlist=blocked with no explicit blocking_reasons still gives blocked."""
    result = compute_candidate_readiness(
        _candidate(automation_risk="high"),
        _gate(shortlist="blocked"),
        repair_history=[],
    )
    assert result["readiness"] == "blocked"
    assert result["automation_status"] == "blocked"


def test_automation_status_partial_with_secrets() -> None:
    """Required secrets downgrade automation_status to partial."""
    result = compute_candidate_readiness(
        _candidate(required_secrets=["MAPILLARY_API_KEY"]),
        _gate(required_secrets=["MAPILLARY_API_KEY"]),
        repair_history=[],
    )
    assert result["automation_status"] == "partial"


def test_identification_causal_documented() -> None:
    """Causal claim with ≥3 threats + full mitigations → documented_causal."""
    result = compute_candidate_readiness(
        _candidate(claim_strength="causal"),
        _gate(),
        repair_history=[],
    )
    assert result["identification_status"] == "documented_causal"


def test_identification_causal_underdocumented() -> None:
    """Causal claim with < 3 threats → causal_claim_underdocumented."""
    result = compute_candidate_readiness(
        _candidate(claim_strength="causal", key_threats=["t1"], mitigations={"t1": "m1"}),
        _gate(),
        repair_history=[],
    )
    assert result["identification_status"] == "causal_claim_underdocumented"


def test_canonicalize_repair_counts_as_auto_fix() -> None:
    """result='canonicalized' must be included in auto_fix_actions."""
    repair_history = [
        {"action": "canonicalize_source_name", "result": "canonicalized"},
    ]
    result = compute_candidate_readiness(_candidate(), _gate(), repair_history)
    assert "canonicalize_source_name" in result["auto_fix_actions"]
    assert result["readiness"] == "ready_after_auto_fix"


def test_source_alias_flags_invisible_in_user_reasons() -> None:
    """Canonicalization reasons must not appear in user_visible_reasons."""
    result = compute_candidate_readiness(
        _candidate(),
        _gate(reasons=["non_canonical_source_name"]),
        repair_history=[{"action": "canonicalize_source_name", "result": "canonicalized"}],
    )
    assert "non_canonical_source_name" not in result["user_visible_reasons"]
    assert result["readiness"] == "ready_after_auto_fix"
