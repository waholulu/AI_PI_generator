"""Tests for the OSMnx feature plan builder.

All tests use fixture mode — no live OSM calls.
"""
from __future__ import annotations

import pytest

from agents.feature_modules.osmnx_features import (
    EXPECTED_FEATURES,
    SMOKE_TEST_GEOGRAPHY,
    build_osmnx_feature_plan,
    build_osmnx_features,
)


def test_feature_plan_contains_expected_features():
    plan = build_osmnx_feature_plan(
        candidate_id="beh_001",
        exposure_family="street_network",
        unit_of_analysis="census_tract",
    )
    for feat in ["intersection_density", "street_length_density", "node_density"]:
        assert feat in plan["expected_features"], f"{feat} missing from plan"


def test_feature_plan_runtime_target():
    plan = build_osmnx_feature_plan(
        candidate_id="beh_001",
        exposure_family="street_network",
        unit_of_analysis="census_tract",
    )
    assert plan["runtime_target_minutes"] <= 10


def test_feature_plan_fallback_policy():
    plan = build_osmnx_feature_plan(
        candidate_id="beh_001",
        exposure_family="street_network",
        unit_of_analysis="census_tract",
    )
    assert plan["fallback"] == "use_fixture_if_osm_call_fails"


def test_feature_plan_default_geography():
    plan = build_osmnx_feature_plan(
        candidate_id="beh_001",
        exposure_family="street_network",
        unit_of_analysis="census_tract",
    )
    assert plan["smoke_test_geography"] == SMOKE_TEST_GEOGRAPHY


def test_feature_plan_custom_geography():
    plan = build_osmnx_feature_plan(
        candidate_id="beh_002",
        exposure_family="street_network",
        unit_of_analysis="county",
        place_name="Somerville, Massachusetts, USA",
    )
    assert plan["smoke_test_geography"] == "Somerville, Massachusetts, USA"


def test_feature_plan_module_reference():
    plan = build_osmnx_feature_plan(
        candidate_id="beh_003",
        exposure_family="street_network",
        unit_of_analysis="census_tract",
    )
    assert plan["module"] == "agents.feature_modules.osmnx_features"
    assert plan["function"] == "build_osmnx_features"


def test_feature_plan_required_extras():
    plan = build_osmnx_feature_plan(
        candidate_id="beh_004",
        exposure_family="street_network",
        unit_of_analysis="census_tract",
    )
    assert "geospatial" in plan["required_extras"]


def test_feature_plan_all_expected_features_present():
    plan = build_osmnx_feature_plan(
        candidate_id="beh_005",
        exposure_family="street_network",
        unit_of_analysis="census_tract",
    )
    for feat in EXPECTED_FEATURES:
        assert feat in plan["expected_features"], f"{feat} not declared in plan"


def test_fixture_features_no_geodataframe():
    """build_osmnx_features with use_fixture=True and no GeoDataFrame."""
    result = build_osmnx_features(
        place_name="Cambridge, Massachusetts, USA",
        boundary_gdf=None,
        unit_id_col="GEOID",
        use_fixture=True,
    )
    assert isinstance(result, dict)
    assert len(result) >= 1
    for unit_id, features in result.items():
        assert "intersection_density" in features
        assert isinstance(features["intersection_density"], float)


def test_fixture_features_values_plausible():
    result = build_osmnx_features(
        place_name="Cambridge, Massachusetts, USA",
        boundary_gdf=None,
        unit_id_col="GEOID",
        use_fixture=True,
    )
    first = next(iter(result.values()))
    assert first["intersection_density"] > 0
    assert first["average_circuity"] >= 1.0
