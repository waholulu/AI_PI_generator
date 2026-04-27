"""Tests for candidate_repair.repair_candidate().

All tests are mock-only — no API keys or network calls required.
The SourceRegistry reads config/source_capabilities.yaml (project root).

Test matrix:
  Case 1: Alias canonicalization (OSMnx → OSMnx_OpenStreetMap)
  Case 2: Missing boundary source → TIGER_Lines added
  Case 3: Missing controls → ACS added
  Case 4: Missing threats → filled from method template
  Case 5: Unknown outcome source → replaced with CDC_PLACES
  Case 6: Google Street View paid API → stays blocked, history documented
"""
from __future__ import annotations

import pytest

from agents.candidate_feasibility import precheck_candidate
from agents.candidate_repair import repair_candidate
from models.candidate_composer_schema import ComposedCandidate

# ── Test fixtures ──────────────────────────────────────────────────────────────

_STANDARD_THREATS = [
    "socioeconomic_confounding",
    "residential_self_selection",
    "spatial_autocorrelation",
]
_STANDARD_MITIGATIONS = {
    "socioeconomic_confounding": "Include ACS tract-level controls.",
    "residential_self_selection": "Include county fixed effects.",
    "spatial_autocorrelation": "Use spatially robust SEs.",
}
_STD_JOIN_PLAN = {
    "boundary_source": ["TIGER_Lines"],
    "controls": ["ACS"],
    "join_key": "GEOID",
}


def _make(
    candidate_id: str = "test_001",
    template_id: str = "built_environment_health_v1",
    exposure_family: str = "street_network",
    exposure_source: str = "OSMnx_OpenStreetMap",
    outcome_family: str = "physical_inactivity",
    outcome_source: str = "CDC_PLACES",
    unit_of_analysis: str = "census_tract",
    join_plan: dict | None = None,
    method_template: str = "cross_sectional_spatial_association",
    key_threats: list | None = None,
    mitigations: dict | None = None,
    automation_risk: str = "low",
) -> ComposedCandidate:
    return ComposedCandidate(
        candidate_id=candidate_id,
        template_id=template_id,
        exposure_family=exposure_family,
        exposure_source=exposure_source,
        outcome_family=outcome_family,
        outcome_source=outcome_source,
        unit_of_analysis=unit_of_analysis,
        join_plan=join_plan if join_plan is not None else dict(_STD_JOIN_PLAN),
        method_template=method_template,
        key_threats=key_threats if key_threats is not None else list(_STANDARD_THREATS),
        mitigations=mitigations if mitigations is not None else dict(_STANDARD_MITIGATIONS),
        automation_risk=automation_risk,
    )


# ── Case 1: Alias canonicalization ────────────────────────────────────────────

def test_alias_canonicalization():
    """'OSMnx' is a registry alias; repair must update the field to 'OSMnx_OpenStreetMap'."""
    candidate = _make(exposure_source="OSMnx")

    gate_status = precheck_candidate(candidate)
    repaired, new_status, history = repair_candidate(candidate, gate_status)

    assert repaired.exposure_source == "OSMnx_OpenStreetMap", (
        f"Expected canonical ID, got: {repaired.exposure_source}"
    )
    canon_entries = [h for h in history if h["action"] == "canonicalize_source_name"]
    assert canon_entries, "Expected at least one canonicalize_source_name entry"
    # Confirm the before/after is documented
    assert any(
        "exposure_source" in h["before"] and h["before"]["exposure_source"] == "OSMnx"
        for h in canon_entries
    )


# ── Case 2: Missing boundary source → TIGER_Lines added ─────────────────────

def test_missing_boundary_source_gets_tiger_lines():
    """Empty boundary_source must be filled with TIGER_Lines via proactive normalization."""
    candidate = _make(
        join_plan={"boundary_source": [], "controls": ["ACS"], "join_key": "GEOID"}
    )

    gate_status = precheck_candidate(candidate)
    repaired, new_status, history = repair_candidate(candidate, gate_status)

    assert "TIGER_Lines" in repaired.join_plan["boundary_source"], (
        f"Expected TIGER_Lines in boundary_source, got: {repaired.join_plan['boundary_source']}"
    )
    assert any(
        h["action"] == "add_boundary_source_tiger_lines" for h in history
    ), f"Expected add_boundary_source_tiger_lines in history; got: {[h['action'] for h in history]}"
    assert new_status["overall"] in ("pass", "warning"), (
        f"Expected pass or warning after repair, got: {new_status['overall']}"
    )


# ── Case 3: Missing controls → ACS added ─────────────────────────────────────

def test_missing_controls_gets_acs():
    """Empty controls must be filled with ACS via proactive normalization."""
    candidate = _make(
        join_plan={"boundary_source": ["TIGER_Lines"], "controls": [], "join_key": "GEOID"}
    )

    gate_status = precheck_candidate(candidate)
    repaired, new_status, history = repair_candidate(candidate, gate_status)

    assert "ACS" in repaired.join_plan["controls"], (
        f"Expected ACS in controls, got: {repaired.join_plan['controls']}"
    )
    assert any(
        h["action"] == "add_default_controls" for h in history
    ), f"Expected add_default_controls in history; got: {[h['action'] for h in history]}"


# ── Case 4: Missing threats → filled from method template ────────────────────

def test_missing_threats_filled_from_template():
    """Empty key_threats must be filled from the method template with mitigations."""
    candidate = _make(key_threats=[], mitigations={})

    gate_status = precheck_candidate(candidate)
    assert gate_status["subchecks"]["identification_threats"] == "warning", (
        "Precondition: empty threats must cause identification_threats warning"
    )

    repaired, new_status, history = repair_candidate(candidate, gate_status)

    assert len(repaired.key_threats) >= 3, (
        f"Expected ≥3 threats after repair, got: {repaired.key_threats}"
    )
    assert all(t in repaired.mitigations for t in repaired.key_threats), (
        "Every threat must have a mitigation after repair"
    )
    assert any(
        h["action"] == "fill_threats_from_method_template" for h in history
    )
    # After repair, identification_threats should be pass
    assert new_status["subchecks"]["identification_threats"] == "pass", (
        f"Expected identification_threats=pass after filling; got {new_status['subchecks']}"
    )


# ── Case 5: Unknown outcome source → replaced with CDC_PLACES ────────────────

def test_unknown_outcome_source_replaced_with_cdc_places():
    """An unrecognised outcome source must be replaced with CDC_PLACES."""
    candidate = _make(outcome_source="Unknown_Health_Source", outcome_family="obesity")

    gate_status = precheck_candidate(candidate)
    assert gate_status["overall"] == "fail", (
        "Precondition: unknown source must cause fail"
    )

    repaired, new_status, history = repair_candidate(candidate, gate_status)

    assert repaired.outcome_source == "CDC_PLACES", (
        f"Expected CDC_PLACES, got: {repaired.outcome_source}"
    )
    assert any(
        h["action"] == "replace_outcome_source_from_template" for h in history
    ), f"Expected replace_outcome_source_from_template; got: {[h['action'] for h in history]}"
    # After replacement, overall should improve
    assert new_status["overall"] in ("pass", "warning"), (
        f"Expected pass or warning after repair, got: {new_status['overall']}"
    )


# ── Case 6: Paid API source stays blocked ─────────────────────────────────────

def test_google_street_view_paid_api_stays_blocked():
    """Google Street View (cost_required=True) must never be promoted to 'ready'.

    The repair loop must document the reason (paid_source_not_allowed) in
    repair_history and set shortlist_status='blocked'.
    """
    candidate = _make(
        exposure_source="Google_Street_View_Static_API",
        exposure_family="streetview_built_form",
        automation_risk="high",
    )

    gate_status = precheck_candidate(candidate)
    repaired, new_status, history = repair_candidate(candidate, gate_status)

    assert new_status["shortlist_status"] == "blocked", (
        f"Expected blocked for paid API source, got: {new_status['shortlist_status']}"
    )
    assert any(
        h["issue"] in ("paid_source_not_allowed", "experimental_source_requires_key")
        for h in history
    ), (
        "Expected paid_source_not_allowed or experimental_source_requires_key in history; "
        f"got issues: {[h['issue'] for h in history]}"
    )
    # Confirm the exposure source was NOT silently swapped to something else
    assert repaired.exposure_source == "Google_Street_View_Static_API", (
        "Paid source must not be silently replaced (no stable fallback for streetview_built_form)"
    )
