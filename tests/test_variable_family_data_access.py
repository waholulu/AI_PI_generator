"""Tests that data_accessibility uses VariableSpec.family for registry matching."""
from __future__ import annotations

from agents.data_accessibility import evaluate_data_sources, summarize_data_access
from models.research_plan_schema import (
    DataSourceSpec,
    FeasibilitySpec,
    IdentificationSpec,
    ResearchPlan,
    VariableSpec,
)


def _make_plan(
    exposure_name: str,
    exposure_family: str,
    outcome_name: str,
    outcome_family: str,
    sources: list[DataSourceSpec],
) -> ResearchPlan:
    return ResearchPlan(
        run_id="test",
        project_title="Test",
        research_question="Test?",
        short_rationale="Test",
        geography="United States",
        time_window="2018-2022",
        unit_of_analysis="census tract",
        exposure=VariableSpec(
            name=exposure_name,
            family=exposure_family,
            spatial_unit="census tract",
            temporal_frequency="annual",
        ),
        outcome=VariableSpec(
            name=outcome_name,
            family=outcome_family,
            spatial_unit="census tract",
            temporal_frequency="annual",
        ),
        identification=IdentificationSpec(primary_method="OLS regression"),
        data_sources=sources,
        feasibility=FeasibilitySpec(),
    )


def _exposure_source(families: list[str]) -> DataSourceSpec:
    return DataSourceSpec(
        name="OSMnx_OpenStreetMap",
        role="exposure",
        source_type="api",
        access_url="https://openstreetmap.org",
        machine_readable=True,
        covers_variable_families=families,
    )


def _outcome_source(families: list[str]) -> DataSourceSpec:
    return DataSourceSpec(
        name="CDC_PLACES",
        role="outcome",
        source_type="api",
        access_url="https://data.cdc.gov",
        machine_readable=True,
        covers_variable_families=families,
    )


def test_family_matches_when_name_differs() -> None:
    """family='street_connectivity' matches even if name is a long display string."""
    plan = _make_plan(
        exposure_name="Street Network Connectivity Index",
        exposure_family="street_connectivity",
        outcome_name="Physical Inactivity Prevalence",
        outcome_family="physical_inactivity",
        sources=[
            _exposure_source(["street_connectivity"]),
            _outcome_source(["physical_inactivity"]),
        ],
    )
    checks = evaluate_data_sources(plan)
    exp_check = next(c for c in checks if "OSMnx" in c.source_name)
    out_check = next(c for c in checks if "CDC" in c.source_name)
    assert exp_check.covers_exposure, "exposure family should match via family field"
    assert out_check.covers_outcome, "outcome family should match via family field"
    verdict, reasons = summarize_data_access(checks)
    assert "missing_exposure_role_source" not in reasons
    assert "missing_outcome_role_source" not in reasons


def test_name_fallback_when_family_empty() -> None:
    """When family is empty, name is used as fallback for matching."""
    plan = _make_plan(
        exposure_name="street_connectivity",
        exposure_family="",
        outcome_name="physical_inactivity",
        outcome_family="",
        sources=[
            _exposure_source(["street_connectivity"]),
            _outcome_source(["physical_inactivity"]),
        ],
    )
    checks = evaluate_data_sources(plan)
    exp_check = next(c for c in checks if "OSMnx" in c.source_name)
    out_check = next(c for c in checks if "CDC" in c.source_name)
    assert exp_check.covers_exposure
    assert out_check.covers_outcome


def test_boundary_source_not_required_to_cover_exposure_or_outcome() -> None:
    """Boundary source (TIGER) should not be flagged for missing exposure/outcome coverage."""
    boundary = DataSourceSpec(
        name="TIGER_Lines",
        role="boundary",
        source_type="api",
        access_url="https://tiger.census.gov",
        machine_readable=True,
        join_keys=["GEOID"],
    )
    plan = _make_plan(
        exposure_name="Street Network Connectivity Index",
        exposure_family="street_connectivity",
        outcome_name="Physical Inactivity Prevalence",
        outcome_family="physical_inactivity",
        sources=[
            _exposure_source(["street_connectivity"]),
            _outcome_source(["physical_inactivity"]),
            boundary,
        ],
    )
    checks = evaluate_data_sources(plan)
    tiger_check = next(c for c in checks if "TIGER" in c.source_name)
    assert tiger_check.verdict in {"pass", "warning"}
    assert "missing_join_path" not in tiger_check.reasons


def test_run_hard_blockers_role_based_flag() -> None:
    """run_hard_blockers with use_role_based_g3=True uses role-based G3."""
    from agents.rule_engine import RuleEngine

    engine = RuleEngine()
    # Build a minimal dummy topic with correct attributes
    from unittest.mock import MagicMock
    topic = MagicMock()
    topic.exposure_X.spatial_unit = "census tract"
    topic.outcome_Y.spatial_unit = "census tract"
    topic.spatial_scope.spatial_unit = "census tract"
    topic.temporal_scope.start_year = 2018
    topic.temporal_scope.end_year = 2022
    topic.identification.primary.value = "ols_regression"

    results = engine.run_hard_blockers(
        topic=topic,
        declared_sources=["OSMnx_OpenStreetMap", "CDC_PLACES", "TIGER_Lines"],
        use_role_based_g3=True,
        exposure_family="street_connectivity",
        outcome_family="physical_inactivity",
    )
    g3 = next(r for r in results if r.gate_id == "G3")
    assert g3.name == "data_availability_role_based", (
        "role-based G3 should be used when use_role_based_g3=True"
    )
