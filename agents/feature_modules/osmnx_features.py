"""OSMnx-based street network feature plan builder and live interface.

CI uses fixture mode (no live OSM calls).
Smoke tests and cloud runs use the live interface with a small geography.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

EXPECTED_FEATURES = [
    "intersection_density",
    "street_length_density",
    "node_density",
    "edge_density",
    "average_circuity",
    "amenity_density",
    "park_accessibility",
    "transit_stop_density",
]

SMOKE_TEST_GEOGRAPHY = "Cambridge, Massachusetts, USA"


def build_osmnx_feature_plan(
    candidate_id: str,
    exposure_family: str,
    unit_of_analysis: str,
    place_name: str = SMOKE_TEST_GEOGRAPHY,
    network_type: str = "walk",
) -> dict[str, Any]:
    """Return a machine-readable feature plan for OSMnx-based candidates.

    Does not call OSM; purely declarative.  The implementation_spec_builder
    embeds this plan in the development pack so Claude Code knows exactly
    what to build.
    """
    return {
        "module": "agents.feature_modules.osmnx_features",
        "function": "build_osmnx_features",
        "candidate_id": candidate_id,
        "exposure_family": exposure_family,
        "smoke_test_geography": place_name,
        "network_type": network_type,
        "unit_of_analysis": unit_of_analysis,
        "expected_features": EXPECTED_FEATURES,
        "runtime_target_minutes": 10,
        "fallback": "use_fixture_if_osm_call_fails",
        "required_extras": ["geospatial"],
        "notes": (
            "Run build_osmnx_feature_plan() in CI (fixture mode). "
            "Run build_osmnx_features() for live smoke tests against "
            "Cambridge, MA or a similarly small geography."
        ),
    }


def build_osmnx_features(
    place_name: str,
    boundary_gdf: Any,
    unit_id_col: str,
    network_type: str = "walk",
    use_fixture: bool = False,
) -> dict[str, Any]:
    """Compute tract/block-group level OSMnx street network features.

    Parameters
    ----------
    place_name:
        Nominatim place string (e.g. "Cambridge, Massachusetts, USA").
    boundary_gdf:
        GeoDataFrame of spatial units (tracts/block groups) with CRS set.
    unit_id_col:
        Column in boundary_gdf that holds the unit identifier (e.g. "GEOID").
    network_type:
        OSMnx network type ("walk", "drive", "bike", "all").
    use_fixture:
        If True, skip live OSM call and return a minimal fixture dict.
        Useful for CI and unit tests.

    Returns
    -------
    dict mapping unit_id → feature dict with EXPECTED_FEATURES keys.
    """
    if use_fixture:
        return _fixture_features(boundary_gdf, unit_id_col)

    try:
        import osmnx as ox  # type: ignore[import]
    except ImportError as exc:
        warnings.warn(
            "osmnx is not installed.  Install the 'geospatial' extras or set "
            "use_fixture=True for CI runs.",
            stacklevel=2,
        )
        raise RuntimeError(
            "osmnx not available; install geospatial extras or use use_fixture=True"
        ) from exc

    results: dict[str, dict[str, float]] = {}

    for _, row in boundary_gdf.iterrows():
        unit_id = str(row[unit_id_col])
        try:
            G = ox.graph_from_polygon(row.geometry, network_type=network_type)
            stats = ox.basic_stats(G)
            area_km2 = row.geometry.area / 1e6

            results[unit_id] = {
                "intersection_density": stats.get("intersection_count", 0) / max(area_km2, 1e-6),
                "street_length_density": stats.get("street_length_total", 0) / max(area_km2, 1e-6),
                "node_density": stats.get("n", 0) / max(area_km2, 1e-6),
                "edge_density": stats.get("m", 0) / max(area_km2, 1e-6),
                "average_circuity": stats.get("circuity_avg", float("nan")),
                "amenity_density": float("nan"),
                "park_accessibility": float("nan"),
                "transit_stop_density": float("nan"),
            }
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"OSMnx failed for unit {unit_id}: {exc}", stacklevel=2)
            results[unit_id] = _empty_feature_row()

    return results


def _empty_feature_row() -> dict[str, float]:
    return {feat: float("nan") for feat in EXPECTED_FEATURES}


def _fixture_features(boundary_gdf: Any, unit_id_col: str) -> dict[str, dict[str, float]]:
    """Return plausible synthetic features for each unit in boundary_gdf."""
    import random

    rng = random.Random(42)
    results: dict[str, dict[str, float]] = {}

    if boundary_gdf is None:
        # Called without a real GeoDataFrame (unit-test shortcut)
        return {
            "fixture_tract_001": {
                "intersection_density": 42.3,
                "street_length_density": 18500.0,
                "node_density": 55.1,
                "edge_density": 88.4,
                "average_circuity": 1.04,
                "amenity_density": float("nan"),
                "park_accessibility": float("nan"),
                "transit_stop_density": float("nan"),
            }
        }

    for _, row in boundary_gdf.iterrows():
        unit_id = str(row[unit_id_col])
        results[unit_id] = {
            "intersection_density": round(rng.uniform(20, 80), 2),
            "street_length_density": round(rng.uniform(8000, 30000), 1),
            "node_density": round(rng.uniform(30, 100), 2),
            "edge_density": round(rng.uniform(50, 150), 2),
            "average_circuity": round(rng.uniform(1.0, 1.15), 4),
            "amenity_density": float("nan"),
            "park_accessibility": float("nan"),
            "transit_stop_density": float("nan"),
        }

    return results
