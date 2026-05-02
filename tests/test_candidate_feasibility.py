"""Tests for candidate_feasibility.precheck_candidate().

All tests are mock-only — no API keys or network calls required.
The SourceRegistry reads config/source_capabilities.yaml (project root).
"""
from __future__ import annotations

import pytest

from agents.candidate_feasibility import precheck_candidate
from models.candidate_composer_schema import ComposedCandidate


# ── helpers ───────────────────────────────────────────────────────────────────

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
    exposure_family="street_network",
    exposure_source="OSMnx_OpenStreetMap",
    outcome_family="physical_inactivity",
    outcome_source="CDC_PLACES",
    unit_of_analysis="census_tract",
    join_plan=None,
    key_threats=None,
    mitigations=None,
    automation_risk="low",
) -> ComposedCandidate:
    return ComposedCandidate(
        candidate_id="test_001",
        template_id="built_environment_health_v1",
        exposure_family=exposure_family,
        exposure_source=exposure_source,
        outcome_family=outcome_family,
        outcome_source=outcome_source,
        unit_of_analysis=unit_of_analysis,
        join_plan=join_plan if join_plan is not None else dict(_STD_JOIN_PLAN),
        method_template="cross_sectional_spatial_association",
        key_threats=key_threats if key_threats is not None else list(_STANDARD_THREATS),
        mitigations=mitigations if mitigations is not None else dict(_STANDARD_MITIGATIONS),
        automation_risk=automation_risk,
    )


# ── Case 1: OSMnx + CDC PLACES + ACS + TIGER → overall pass ──────────────────

def test_osmnx_cdc_tiger_acs_pass():
    candidate = _make(
        exposure_family="street_network",
        exposure_source="OSMnx_OpenStreetMap",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
    )
    result = precheck_candidate(candidate)

    assert result["overall"] == "pass", f"Expected pass, got {result}"
    assert result["shortlist_status"] == "ready"
    assert result["subchecks"]["source_exists"] == "pass"
    assert result["subchecks"]["role_coverage"] == "pass"
    assert result["subchecks"]["machine_readable"] == "pass"
    assert result["subchecks"]["spatial_join_path"] == "pass"
    assert result["subchecks"]["cloud_automation_feasibility"] == "pass"
    assert result["subchecks"]["identification_threats"] == "pass"
    assert result["reasons"] == []


# ── Case 2: EPA Walkability + CDC PLACES → warning (aggregation) ──────────────

def test_epa_walkability_cdc_aggregation_warning():
    candidate = _make(
        exposure_family="walkability",
        exposure_source="EPA_National_Walkability_Index",
        outcome_family="obesity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
    )
    result = precheck_candidate(candidate)

    assert result["overall"] == "warning", f"Expected warning, got {result}"
    assert result["shortlist_status"] == "review"
    assert result["subchecks"]["source_exists"] == "pass"
    assert result["subchecks"]["role_coverage"] == "pass"
    assert result["subchecks"]["spatial_join_path"] == "warning"

    agg_reason = next(
        (r for r in result["reasons"] if "aggregation_required" in r), None
    )
    assert agg_reason is not None, (
        f"Expected an aggregation_required reason; got {result['reasons']}"
    )
    assert "block_group" in agg_reason
    assert "add_aggregation_plan" in result["repair_suggestions"]


# ── Case 3: Mapillary (experimental) + CDC PLACES → warning ──────────────────

def test_experimental_mapillary_warning():
    candidate = _make(
        exposure_family="greenery_visibility",
        exposure_source="Mapillary_Street_Images",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        automation_risk="high",
    )
    result = precheck_candidate(candidate)

    assert result["overall"] == "warning", f"Expected warning, got {result}"
    assert result["shortlist_status"] == "review"
    assert result["subchecks"]["cloud_automation_feasibility"] == "warning"
    assert any("experimental" in r for r in result["reasons"])


# ── Case 4: unknown outcome source → fail ────────────────────────────────────

def test_unknown_outcome_source_fail():
    candidate = _make(
        outcome_source="Unknown_Source_XYZ",
    )
    result = precheck_candidate(candidate)

    assert result["overall"] == "fail", f"Expected fail, got {result}"
    assert result["shortlist_status"] == "blocked"
    assert result["subchecks"]["source_exists"] == "fail"
    assert "source_not_in_registry" in result["reasons"]
    assert "replace_outcome_source_from_template" in result["repair_suggestions"]


# ── Case 5: missing key_threats → warning ────────────────────────────────────

def test_missing_threats_warning():
    candidate = _make(
        key_threats=[],
        mitigations={},
    )
    result = precheck_candidate(candidate)

    assert result["overall"] == "warning", f"Expected warning, got {result}"
    assert result["shortlist_status"] == "review"
    assert result["subchecks"]["identification_threats"] == "warning"
    assert "missing_identification_threats" in result["reasons"]


# ── Extra: wrong role for exposure source → role_coverage fail ───────────────

def test_wrong_role_fails():
    # CDC_PLACES has only "outcome" role; using it as exposure should fail
    candidate = _make(
        exposure_source="CDC_PLACES",
        outcome_source="CDC_PLACES",
    )
    result = precheck_candidate(candidate)

    assert result["overall"] == "fail"
    assert result["subchecks"]["role_coverage"] == "fail"
    assert "missing_exposure_role_source" in result["reasons"]


# ── Step 3 new tests ──────────────────────────────────────────────────────────

def test_time_overlap_uses_coverage_year_min_max():
    """time_overlap subcheck reads coverage_year_min/max (not coverage_start/end_year)."""
    # EPA SLD coverage_year_min=2021, coverage_year_max=2021, which is within [2016, 2024]
    # so time_overlap should pass
    candidate = _make(
        exposure_family="density",
        exposure_source="EPA_Smart_Location_Database",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
    )
    result = precheck_candidate(candidate)
    # time_overlap must be checked and should pass (2021 is in 2016-2024 window)
    assert "time_overlap" in result["subchecks"]
    assert result["subchecks"]["time_overlap"] == "pass"
    assert "time_overlap_insufficient" not in result["reasons"]


def test_epa_sld_panel_window_gets_warning():
    """EPA SLD (single year 2021) used in multi-year panel context gets a warning."""
    candidate = _make(
        exposure_family="density",
        exposure_source="EPA_Smart_Location_Database",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
    )
    result = precheck_candidate(candidate)
    # EPA SLD is cross_sectional_only + 2021 < PLAN_END 2024
    # → time_cross_sectional_justified should be warning
    assert "time_cross_sectional_justified" in result["subchecks"]
    assert result["subchecks"]["time_cross_sectional_justified"] == "warning"
    assert any("single_year_source" in r for r in result["reasons"])
    assert "change_design_to_cross_sectional_or_justify_static_exposure" in result["repair_suggestions"]


def test_block_group_to_tract_requires_aggregation_recipe():
    """EPA SLD (block group) used with tract analysis unit passes join_recipe_exists
    because its data catalog profile includes a join recipe."""
    candidate = _make(
        exposure_family="density",
        exposure_source="EPA_Smart_Location_Database",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
    )
    result = precheck_candidate(candidate)
    # EPA SLD has join recipes in data catalog → join_recipe_exists should pass
    assert result["subchecks"].get("join_recipe_exists") == "pass"
    assert result["subchecks"].get("aggregation_method_defined") == "pass"
    assert "missing_join_recipe" not in result["reasons"]
    assert "missing_aggregation_method" not in result["reasons"]


def test_missing_variable_mapping_blocks_claude_ready():
    """Exposure source without variable mapping for the family gets a warning."""
    candidate = _make(
        exposure_family="nonexistent_family_xyz",
        exposure_source="EPA_Smart_Location_Database",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
    )
    result = precheck_candidate(candidate)
    # The exposure family is not in EPA SLD catalog → variable_mapping_exists = warning
    assert result["subchecks"].get("variable_mapping_exists") == "warning"
    assert any("no_variable_mapping_for_nonexistent_family_xyz" in r for r in result["reasons"])


def test_source_without_catalog_profile_skips_new_checks():
    """Sources without a data catalog profile (e.g. OSMnx) pass new subchecks by default."""
    candidate = _make(
        exposure_family="street_network",
        exposure_source="OSMnx_OpenStreetMap",
        outcome_family="physical_inactivity",
        outcome_source="CDC_PLACES",
        unit_of_analysis="census_tract",
    )
    result = precheck_candidate(candidate)
    # All new subchecks should be "pass" for uncatalogued sources
    for new_check in [
        "variable_mapping_exists", "native_grain_known", "target_grain_reachable",
        "join_recipe_exists", "aggregation_method_defined", "time_cross_sectional_justified",
    ]:
        assert result["subchecks"].get(new_check) == "pass", (
            f"{new_check} should be 'pass' for OSMnx (no catalog profile), "
            f"got {result['subchecks'].get(new_check)}"
        )
