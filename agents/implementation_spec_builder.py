from __future__ import annotations

from agents.feature_modules.osmnx_features import build_osmnx_feature_plan
from models.candidate_composer_schema import ComposedCandidate
from models.implementation_schema import (
    AnalysisStep,
    DataAcquisitionStep,
    FeatureEngineeringStep,
    ImplementationSpec,
)

_SMOKE_GEOGRAPHY = "Cambridge, Massachusetts"
_SMOKE_RUNTIME_MINUTES = 8
_SMOKE_MIN_ROWS = 10


def _acquisition_method(source_name: str) -> tuple[str, list[str]]:
    sid = source_name.lower()
    if "osmnx" in sid:
        return "osmnx", ["data/raw/osmnx_features.parquet"]
    if source_name == "CDC_PLACES":
        return "api", ["data/raw/cdc_places.csv"]
    if source_name == "ACS":
        return "api", ["data/raw/acs_controls.csv"]
    if source_name == "TIGER_Lines":
        return "download", ["data/raw/tiger_tracts.geojson"]
    if source_name in {"NLCD", "EPA_EnviroAtlas"}:
        return "download", ["data/raw/landcover_or_enviroatlas.tif", "data/raw/tract_aggregate.csv"]
    if source_name == "VIIRS":
        return "download", ["data/raw/viirs_nighttime_lights.tif"]
    if source_name == "Microsoft_Building_Footprints":
        return "download", ["data/raw/microsoft_buildings.geojson"]
    if source_name == "GTFS":
        return "download", ["data/raw/gtfs.zip"]
    if source_name == "EPA_National_Walkability_Index":
        return "download", ["data/raw/epa_walkability.csv"]
    return "api", ["data/raw/source_extract.csv"]


def _source_notes(source_name: str) -> str:
    notes = {
        "OSMnx_OpenStreetMap": (
            "Use osmnx.graph_from_place() with network_type='walk'. "
            "In CI, load from fixture data/fixtures/osmnx_cambridge.graphml. "
            "Live call only during smoke test."
        ),
        "CDC_PLACES": (
            "Download tract-level outcomes by GEOID via CDC PLACES API "
            "(https://data.cdc.gov/resource/). Filter to target state FIPS."
        ),
        "ACS": (
            "Pull tract-level control variables (poverty rate, median income, "
            "education) via Census Bureau API or Census Data API wrapper."
        ),
        "TIGER_Lines": (
            "Download tract boundary shapefiles from Census TIGER/Line "
            "(https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html). "
            "Join to all other datasets on GEOID."
        ),
        "NLCD": (
            "Download National Land Cover Database GeoTIFF. "
            "Compute zonal statistics (mean green/impervious fraction) per tract polygon. "
            "Use rasterstats.zonal_stats() with TIGER tract geometry."
        ),
        "EPA_National_Walkability_Index": (
            "Download block-group-level walkability CSV from EPA. "
            "Aggregate to tract level via population-weighted mean before joining."
        ),
        "EPA_EnviroAtlas": (
            "Download EnviroAtlas metrics at block-group resolution. "
            "Aggregate to tract with population-weighted mean."
        ),
        "Microsoft_Building_Footprints": (
            "Download GeoJSON footprints from Microsoft Planetary Computer. "
            "Compute density (footprints per km²) per tract via spatial join."
        ),
        "GTFS": (
            "Download GTFS zip from transit agency open data portal. "
            "Compute stop density and service frequency per tract via spatial join."
        ),
        "VIIRS": (
            "Download VIIRS Annual VNP46A4 GeoTIFF from NASA LAADS DAAC. "
            "Compute mean radiance per tract via zonal statistics."
        ),
    }
    return notes.get(source_name, f"Acquire {source_name} data and join to tract GEOID.")


def build_implementation_spec(candidate: ComposedCandidate) -> ImplementationSpec:
    required_extras = ["geospatial"]
    if "experimental" in candidate.technology_tags:
        required_extras.append("vision")

    exposure_method, exposure_files = _acquisition_method(candidate.exposure_source)
    outcome_method, outcome_files = _acquisition_method(candidate.outcome_source)

    # Default control and boundary sources from join_plan
    control_sources: list[str] = candidate.join_plan.get("controls", ["ACS"])
    boundary_sources: list[str] = candidate.join_plan.get("boundary_source", ["TIGER_Lines"])

    # Attach OSMnx feature plan for candidates using OpenStreetMap street network data
    osmnx_feature_plan: dict | None = None
    if "osmnx" in candidate.technology_tags or "osmnx" in candidate.exposure_source.lower():
        osmnx_feature_plan = build_osmnx_feature_plan(
            candidate_id=candidate.candidate_id,
            exposure_family=candidate.exposure_family,
            unit_of_analysis=candidate.unit_of_analysis,
        )

    feature_step_libs = ["pandas", "geopandas"]
    feature_step_pseudo = (
        f"load exposure ({candidate.exposure_source}) and outcome ({candidate.outcome_source}), "
        f"spatially aggregate to {candidate.unit_of_analysis}, join on GEOID; "
        "merge controls (ACS) and apply boundary (TIGER) for spatial integrity"
    )
    if osmnx_feature_plan:
        feature_step_libs.append("osmnx")
        feature_step_pseudo = (
            f"call build_osmnx_features(place_name=smoke_test_geography, use_fixture=True) "
            f"to get {', '.join(osmnx_feature_plan['expected_features'][:3])} + more; "
            f"join to CDC_PLACES outcome on GEOID; "
            "merge ACS controls and TIGER boundary geometry"
        )

    # Build all 4 acquisition steps: exposure, outcome, control(s), boundary
    acquisition_steps: list[DataAcquisitionStep] = [
        DataAcquisitionStep(
            source_name=candidate.exposure_source,
            source_role="exposure",
            method=exposure_method,
            expected_files=exposure_files,
            required_secrets=candidate.required_secrets,
            cache_policy="cache_processed_only",
        ),
        DataAcquisitionStep(
            source_name=candidate.outcome_source,
            source_role="outcome",
            method=outcome_method,
            expected_files=outcome_files,
            cache_policy="cache_processed_only",
        ),
    ]
    for ctrl in control_sources:
        ctrl_method, ctrl_files = _acquisition_method(ctrl)
        acquisition_steps.append(
            DataAcquisitionStep(
                source_name=ctrl,
                source_role="control",
                method=ctrl_method,
                expected_files=ctrl_files,
                cache_policy="cache_processed_only",
            )
        )
    for bnd in boundary_sources:
        bnd_method, bnd_files = _acquisition_method(bnd)
        acquisition_steps.append(
            DataAcquisitionStep(
                source_name=bnd,
                source_role="boundary",
                method=bnd_method,
                expected_files=bnd_files,
                cache_policy="cache_processed_only",
            )
        )

    smoke_plan = [
        f"Run on {_SMOKE_GEOGRAPHY} (single county, fast CI geography)",
        f"Expected runtime: ≤ {_SMOKE_RUNTIME_MINUTES} minutes for smoke test",
        f"Verify tract_features.csv has ≥ {_SMOKE_MIN_ROWS} rows",
        "Verify outcome column exists and is non-null for ≥ 50% of tracts",
        "On network failure: log warning, write empty placeholder file, do not raise",
        f"OSMnx/fixture policy: load from data/fixtures/ in CI, live call in smoke test",
        "No paid API calls; no raw image storage; no auth-required sources",
    ]

    # Source-specific notes added as a structured extra in feature engineering
    source_notes = {
        src: _source_notes(src)
        for src in (
            [candidate.exposure_source, candidate.outcome_source]
            + control_sources
            + boundary_sources
        )
    }

    return ImplementationSpec(
        candidate_id=candidate.candidate_id,
        cloud_safe=candidate.cloud_safe,
        automation_risk=candidate.automation_risk,
        required_python_extras=required_extras,
        required_secrets=candidate.required_secrets,
        osmnx_feature_plan=osmnx_feature_plan,
        data_acquisition_steps=acquisition_steps,
        feature_engineering_steps=[
            FeatureEngineeringStep(
                step_id="join_features_outcome",
                input_sources=(
                    [candidate.exposure_source, candidate.outcome_source]
                    + control_sources
                    + boundary_sources
                ),
                output_features=candidate.exposure_variables,
                library_tags=feature_step_libs,
                pseudo_code=feature_step_pseudo,
            )
        ],
        analysis_steps=[
            AnalysisStep(
                method=candidate.method_template,
                formula_or_model=(
                    f"{candidate.outcome_family} ~ "
                    f"{' + '.join(candidate.exposure_variables[:2] or ['exposure'])} "
                    f"+ controls | {candidate.unit_of_analysis} FE"
                ),
                robustness_checks=[
                    "alternative_exposure_definitions",
                    "clustered_standard_errors",
                    "spatial_lag_controls",
                ],
            )
        ],
        expected_outputs=[
            "data/processed/tract_features.csv",
            "output/tables/model_summary.csv",
            "output/report/technical_summary.md",
        ],
        smoke_test_plan=smoke_plan,
    )
