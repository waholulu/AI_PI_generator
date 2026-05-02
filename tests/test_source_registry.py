"""Tests for the updated SourceRegistry with data catalog support."""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.source_registry import SourceRegistry


# ── Step 1 tests ─────────────────────────────────────────────────────────────

def test_load_data_catalog_sources():
    """SourceRegistry loads profiles from config/data_catalog/sources/."""
    registry = SourceRegistry.load()
    # All 6 catalog sources must be present
    for source_id in [
        "EPA_Smart_Location_Database",
        "EPA_National_Walkability_Index",
        "CDC_PLACES",
        "ACS",
        "TIGER_Lines",
        "Census_LEHD_LODES",
    ]:
        assert registry.resolve(source_id) is not None, f"{source_id} not found in registry"
        assert registry.get_profile(source_id) is not None, f"No profile for {source_id}"


def test_resolve_alias_from_new_catalog():
    """Aliases defined in data catalog YAML files resolve correctly."""
    registry = SourceRegistry.load()
    assert registry.resolve("EPA SLD") == "EPA_Smart_Location_Database"
    assert registry.resolve("SLD") == "EPA_Smart_Location_Database"
    assert registry.resolve("LODES") == "Census_LEHD_LODES"
    assert registry.resolve("CDC Places") == "CDC_PLACES"
    assert registry.resolve("TIGER") == "TIGER_Lines"
    assert registry.resolve("ACS 5-year") == "ACS"


def test_old_source_capabilities_fallback():
    """Sources only in source_capabilities.yaml (not in data catalog) still load."""
    registry = SourceRegistry.load()
    # OSMnx and NLCD are in source_capabilities.yaml but NOT in data_catalog/sources/
    assert registry.resolve("OSMnx_OpenStreetMap") is not None
    assert registry.resolve("NLCD") is not None
    # They should NOT have a rich profile (fallback only)
    assert registry.get_profile("OSMnx_OpenStreetMap") is None
    assert registry.get_profile("NLCD") is None


# ── Step 2 tests ─────────────────────────────────────────────────────────────

def test_epa_sld_is_block_group_native():
    """EPA SLD native unit is census_block_group."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("EPA_Smart_Location_Database")
    assert profile is not None
    assert profile.geography is not None
    assert profile.geography.native_unit == "census_block_group"
    assert registry.get_native_unit("EPA_Smart_Location_Database") == "census_block_group"


def test_epa_sld_is_2021_only():
    """EPA SLD coverage is single year 2021 with cross_sectional_only=True."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("EPA_Smart_Location_Database")
    assert profile is not None
    assert profile.temporal_coverage is not None
    assert profile.temporal_coverage.coverage_year_min == 2021
    assert profile.temporal_coverage.coverage_year_max == 2021
    assert profile.temporal_coverage.cross_sectional_only is True


def test_epa_sld_aggregation_required():
    """EPA SLD requires aggregation from block_group to tract."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("EPA_Smart_Location_Database")
    assert profile.geography.aggregation_required is True
    assert profile.geography.default_aggregation == "population_weighted_mean"
    assert registry.aggregation_required("EPA_Smart_Location_Database") is True
    assert registry.get_default_aggregation_method("EPA_Smart_Location_Database") == "population_weighted_mean"


def test_epa_sld_has_join_recipe():
    """EPA SLD has at least one join recipe defined."""
    registry = SourceRegistry.load()
    recipes = registry.get_join_recipes("EPA_Smart_Location_Database")
    assert len(recipes) >= 1
    assert recipes[0]["left_source"] == "EPA_Smart_Location_Database"


def test_lodes_exposes_od_rac_wac_xwalk():
    """LODES data catalog profile exposes OD, RAC, WAC, XWALK tables."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("Census_LEHD_LODES")
    assert profile is not None
    table_ids = [t.table_id for t in profile.tables]
    for expected_table in ["OD", "RAC", "WAC", "XWALK"]:
        assert expected_table in table_ids, f"Table {expected_table} missing from LODES profile"


def test_lodes_home_and_work_keys_are_distinct():
    """LODES OD table h_geocode and w_geocode are distinct special join keys."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("Census_LEHD_LODES")
    assert profile is not None
    od_table = next((t for t in profile.tables if t.table_id == "OD"), None)
    assert od_table is not None
    special_keys = od_table.special_join_keys
    assert "home_geocode" in special_keys
    assert "workplace_geocode" in special_keys
    assert special_keys["home_geocode"] == "h_geocode"
    assert special_keys["workplace_geocode"] == "w_geocode"
    # They must be different columns
    assert special_keys["home_geocode"] != special_keys["workplace_geocode"]


def test_lodes_has_warning_about_geocode():
    """LODES OD table has warning about not treating h_geocode as generic GEOID."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("Census_LEHD_LODES")
    od_table = next((t for t in profile.tables if t.table_id == "OD"), None)
    assert od_table is not None
    warnings_text = " ".join(od_table.warnings).lower()
    assert "h_geocode" in warnings_text or "do_not_treat" in warnings_text


def test_cdc_places_native_unit_is_tract():
    """CDC PLACES native unit is census_tract."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("CDC_PLACES")
    assert profile is not None
    assert profile.geography is not None
    assert profile.geography.native_unit == "census_tract"


def test_cdc_places_no_aggregation_required():
    """CDC PLACES does not require aggregation (it is natively at tract level)."""
    registry = SourceRegistry.load()
    assert registry.aggregation_required("CDC_PLACES") is False


def test_acs_has_population_weights_family():
    """ACS has the population_weights variable family for SLD aggregation."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("ACS")
    assert profile is not None
    assert "population_weights" in profile.variable_families
    assert "socioeconomic_controls" in profile.variable_families


def test_tiger_lines_has_boundaries_family():
    """TIGER_Lines has a boundaries variable family with geometry."""
    registry = SourceRegistry.load()
    profile = registry.get_profile("TIGER_Lines")
    assert profile is not None
    assert "boundaries" in profile.variable_families


def test_registry_backward_compat_flat_fields():
    """Flat field access (machine_readable, coverage_year_min, etc.) still works."""
    registry = SourceRegistry.load()
    # All catalog sources should have backward-compat flat fields
    for source_id in ["EPA_Smart_Location_Database", "CDC_PLACES", "ACS"]:
        spec = registry.get(source_id)
        assert spec is not None
        assert spec.get("machine_readable") is True
        assert "coverage_year_min" in spec
        assert "coverage_year_max" in spec
        assert "roles" in spec


def test_has_variable_mapping_positive():
    """has_variable_mapping returns True for sources with concrete variables defined."""
    registry = SourceRegistry.load()
    # EPA SLD has density family with real column names
    assert registry.has_variable_mapping("EPA_Smart_Location_Database", "density") is True
    # CDC PLACES has physical_inactivity with real column
    assert registry.has_variable_mapping("CDC_PLACES", "physical_inactivity") is True


def test_has_variable_mapping_negative():
    """has_variable_mapping returns False for missing families."""
    registry = SourceRegistry.load()
    assert registry.has_variable_mapping("EPA_Smart_Location_Database", "nonexistent_family") is False
    # OSMnx has no data catalog profile → no variable mapping
    assert registry.has_variable_mapping("OSMnx_OpenStreetMap", "street_connectivity") is False
