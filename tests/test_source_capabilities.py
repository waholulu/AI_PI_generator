from agents.source_registry import SourceRegistry

_REQUIRED_FIELDS = [
    "roles",
    "machine_readable",
    "auth_required",
    "cost_required",
    "spatial_units",
    "variable_families",
]


def test_all_registry_sources_have_required_fields() -> None:
    registry = SourceRegistry.load()
    for sid, spec in registry.sources.items():
        for field in _REQUIRED_FIELDS:
            assert field in spec, f"{sid} missing required field '{field}'"


def test_source_registry_load_and_resolve_alias() -> None:
    registry = SourceRegistry.load()

    assert "CDC_PLACES" in registry.sources
    assert registry.resolve("OSMnx") == "OSMnx_OpenStreetMap"
    assert registry.resolve("EPA Walkability Index") == "EPA_National_Walkability_Index"


def test_source_registry_list_variable_families_by_role() -> None:
    registry = SourceRegistry.load()
    exposure_families = registry.list_variable_families(role="exposure")

    assert "OSMnx_OpenStreetMap" in exposure_families
    assert "street_connectivity" in exposure_families["OSMnx_OpenStreetMap"]
    assert "CDC_PLACES" not in exposure_families


def test_source_registry_helpers() -> None:
    registry = SourceRegistry.load()

    assert "NLCD" in registry.get_sources_by_variable_family("green_space")
    assert "CDC_PLACES" in registry.get_sources_by_role("outcome")
    assert "CDC_PLACES" in registry.get_machine_readable_sources()
    assert registry.is_cloud_safe("CDC_PLACES") is True
    assert registry.requires_secret("Google_Street_View_Static_API") is True
    assert registry.get_source_tier("VIIRS") == "tier2"
