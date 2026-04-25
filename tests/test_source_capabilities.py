from agents.source_registry import SourceRegistry


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
