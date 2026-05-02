from __future__ import annotations

from agents.feature_modules.osmnx_features import build_osmnx_feature_plan
from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposedCandidate
from models.implementation_schema import (
    AnalysisStep,
    DataAcquisitionStep,
    FeatureEngineeringStep,
    ImplementationSpec,
    SourceUseSpec,
)

_SMOKE_GEOGRAPHY = "Cambridge, Massachusetts"
_SMOKE_RUNTIME_MINUTES = 8
_SMOKE_MIN_ROWS = 10


def _acquisition_method(source_name: str, registry: SourceRegistry | None = None) -> tuple[str, list[str]]:
    """Return (method, expected_files) for a source.

    Prefers the data catalog acquisition spec when available; falls back to
    the original hard-coded mapping for sources without catalog profiles.
    """
    if registry:
        profile = registry.get_profile(source_name)
        if profile and profile.acquisition.method:
            method = profile.acquisition.method
            # Normalise api_download → api for the Literal field
            if method == "api_download":
                method = "api"
            if profile.acquisition.expected_files:
                return method, list(profile.acquisition.expected_files)

    # Hard-coded fallback (preserves original behaviour for uncatalogued sources)
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


def _source_notes(source_name: str, registry: SourceRegistry | None = None) -> str:
    """Return implementation notes for a source.

    Uses data catalog profile notes when available.
    """
    if registry:
        profile = registry.get_profile(source_name)
        if profile:
            parts: list[str] = []
            if profile.acquisition.notes:
                parts.append(profile.acquisition.notes.strip())
            if profile.geography:
                g = profile.geography
                if g.aggregation_required:
                    parts.append(
                        f"Native unit: {g.native_unit}. "
                        f"Aggregate to {', '.join(g.target_units_supported)} "
                        f"using {g.default_aggregation}."
                    )
            if profile.known_limitations:
                parts.append("Known limitations: " + "; ".join(profile.known_limitations[:2]))
            if parts:
                return " ".join(parts)

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


def _build_source_use_spec(
    source_name: str,
    role: str,
    candidate: ComposedCandidate,
    registry: SourceRegistry,
) -> SourceUseSpec:
    """Build a SourceUseSpec from the data catalog profile."""
    profile = registry.get_profile(source_name)
    source_id = registry.resolve(source_name) or source_name

    if role == "exposure":
        raw_cols = _get_raw_columns(source_name, candidate.exposure_family, profile)
        derived = list(candidate.exposure_variables or [])
    elif role == "outcome":
        raw_cols = _get_raw_columns(source_name, candidate.outcome_family, profile)
        derived = list(candidate.outcome_variables or [f"{candidate.outcome_family}_prevalence"])
    elif role == "control":
        raw_cols = ["B17001_002E", "B19013_001E", "B15003_017E", "B01003_001E"]
        derived = ["poverty_rate", "median_income", "pct_no_hs_diploma", "pop_density"]
    else:  # boundary
        raw_cols = ["GEOID", "geometry", "ALAND"]
        derived = ["geometry"]

    native_unit = ""
    target_unit = candidate.unit_of_analysis
    join_recipe: dict | None = None
    aggregation_method = ""
    validation_rules: list[str] = []
    known_limitations: list[str] = []
    acquisition_method = ""
    acquisition_url = ""

    if profile:
        if profile.geography:
            native_unit = profile.geography.native_unit
            aggregation_method = profile.geography.default_aggregation
        if profile.join_recipes:
            join_recipe = profile.join_recipes[0].model_dump()
        if profile.validation_rules:
            validation_rules = [r.rule_id for r in profile.validation_rules[:3]]
        known_limitations = list(profile.known_limitations[:3])
        acquisition_method = profile.acquisition.method
        acquisition_url = profile.acquisition.url

    return SourceUseSpec(
        source_id=source_id,
        role=role,
        native_unit=native_unit,
        target_unit=target_unit,
        raw_columns=raw_cols,
        derived_features=derived,
        acquisition_method=acquisition_method,
        acquisition_url=acquisition_url,
        join_recipe=join_recipe,
        aggregation_method=aggregation_method,
        validation_rules=validation_rules,
        known_limitations=known_limitations,
    )


def _get_raw_columns(
    source_name: str,
    family: str,
    profile,
) -> list[str]:
    """Extract concrete column names from the data catalog profile."""
    if profile is None:
        return []
    vf = profile.variable_families.get(family)
    if not vf:
        return []
    if isinstance(vf, dict):
        variables = vf.get("variables", [])
        cols: list[str] = []
        for v in variables:
            if isinstance(v, dict):
                cols.append(v.get("name", ""))
            elif isinstance(v, str):
                cols.append(v)
        return [c for c in cols if c]
    return []


def _build_data_lineage_plan(
    candidate: ComposedCandidate,
    source_use_specs: list[SourceUseSpec],
) -> dict:
    """Build a structured data lineage plan describing grain conversions."""
    exp_spec = next((s for s in source_use_specs if s.role == "exposure"), None)
    out_spec = next((s for s in source_use_specs if s.role == "outcome"), None)
    ctrl_specs = [s for s in source_use_specs if s.role == "control"]
    bnd_spec = next((s for s in source_use_specs if s.role == "boundary"), None)

    steps: list[dict] = []

    if exp_spec and exp_spec.native_unit and exp_spec.native_unit != exp_spec.target_unit:
        steps.append({
            "step": "aggregate_exposure",
            "source": exp_spec.source_id,
            "from_unit": exp_spec.native_unit,
            "to_unit": exp_spec.target_unit,
            "method": exp_spec.aggregation_method or "population_weighted_mean",
            "weight_source": "ACS B01003_001E",
            "join_recipe": exp_spec.join_recipe.get("recipe_id") if exp_spec.join_recipe else None,
        })

    if out_spec:
        steps.append({
            "step": "acquire_outcome",
            "source": out_spec.source_id,
            "native_unit": out_spec.native_unit or candidate.unit_of_analysis,
            "columns": out_spec.raw_columns[:5],
        })

    for cs in ctrl_specs:
        steps.append({
            "step": "acquire_controls",
            "source": cs.source_id,
            "native_unit": cs.native_unit or candidate.unit_of_analysis,
            "columns": cs.raw_columns[:5],
        })

    steps.append({
        "step": "join_all_to_analysis_unit",
        "analysis_unit": candidate.unit_of_analysis,
        "join_key": "GEOID",
        "boundary_source": bnd_spec.source_id if bnd_spec else "TIGER_Lines",
    })

    return {
        "candidate_id": candidate.candidate_id,
        "analysis_unit": candidate.unit_of_analysis,
        "lineage_steps": steps,
        "final_join_key": "GEOID",
    }


def build_implementation_spec(candidate: ComposedCandidate) -> ImplementationSpec:
    registry = SourceRegistry.load()
    required_extras = ["geospatial"]
    if "experimental" in candidate.technology_tags:
        required_extras.append("vision")

    exposure_method, exposure_files = _acquisition_method(candidate.exposure_source, registry)
    outcome_method, outcome_files = _acquisition_method(candidate.outcome_source, registry)

    control_sources: list[str] = candidate.join_plan.get("controls", ["ACS"])
    boundary_sources: list[str] = candidate.join_plan.get("boundary_source", ["TIGER_Lines"])

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
        ctrl_method, ctrl_files = _acquisition_method(ctrl, registry)
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
        bnd_method, bnd_files = _acquisition_method(bnd, registry)
        acquisition_steps.append(
            DataAcquisitionStep(
                source_name=bnd,
                source_role="boundary",
                method=bnd_method,
                expected_files=bnd_files,
                cache_policy="cache_processed_only",
            )
        )

    # Build SourceUseSpec for each role
    source_use_specs: list[SourceUseSpec] = [
        _build_source_use_spec(candidate.exposure_source, "exposure", candidate, registry),
        _build_source_use_spec(candidate.outcome_source, "outcome", candidate, registry),
    ]
    for ctrl in control_sources:
        source_use_specs.append(_build_source_use_spec(ctrl, "control", candidate, registry))
    for bnd in boundary_sources:
        source_use_specs.append(_build_source_use_spec(bnd, "boundary", candidate, registry))

    data_lineage_plan = _build_data_lineage_plan(candidate, source_use_specs)

    smoke_plan = [
        f"Run on {_SMOKE_GEOGRAPHY} (single county, fast CI geography)",
        f"Expected runtime: ≤ {_SMOKE_RUNTIME_MINUTES} minutes for smoke test",
        f"Verify tract_features.csv has ≥ {_SMOKE_MIN_ROWS} rows",
        "Verify outcome column exists and is non-null for ≥ 50% of tracts",
        "On network failure: log warning, write empty placeholder file, do not raise",
        f"OSMnx/fixture policy: load from data/fixtures/ in CI, live call in smoke test",
        "No paid API calls; no raw image storage; no auth-required sources",
    ]

    source_notes = {
        src: _source_notes(src, registry)
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
        source_use_specs=source_use_specs,
        data_lineage_plan=data_lineage_plan,
    )
