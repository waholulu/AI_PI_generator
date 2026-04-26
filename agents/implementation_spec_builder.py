from __future__ import annotations

from agents.feature_modules.osmnx_features import build_osmnx_feature_plan
from models.candidate_composer_schema import ComposedCandidate
from models.implementation_schema import (
    AnalysisStep,
    DataAcquisitionStep,
    FeatureEngineeringStep,
    ImplementationSpec,
)


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
    return "api", ["data/raw/source_extract.csv"]


def build_implementation_spec(candidate: ComposedCandidate) -> ImplementationSpec:
    required_extras = ["geospatial"]
    if "experimental" in candidate.technology_tags:
        required_extras.append("vision")

    exposure_method, exposure_files = _acquisition_method(candidate.exposure_source)
    outcome_method, outcome_files = _acquisition_method(candidate.outcome_source)

    # Attach OSMnx feature plan for candidates using OpenStreetMap street network data
    osmnx_feature_plan: dict | None = None
    if "osmnx" in candidate.technology_tags or "osmnx" in candidate.exposure_source.lower():
        osmnx_feature_plan = build_osmnx_feature_plan(
            candidate_id=candidate.candidate_id,
            exposure_family=candidate.exposure_family,
            unit_of_analysis=candidate.unit_of_analysis,
        )

    feature_step_libs = ["pandas", "geopandas"]
    feature_step_pseudo = "load exposure and outcome, spatially aggregate to tract, join on GEOID"
    if osmnx_feature_plan:
        feature_step_libs.append("osmnx")
        feature_step_pseudo = (
            f"call build_osmnx_features(place_name=smoke_test_geography, use_fixture=True) "
            f"to get {', '.join(osmnx_feature_plan['expected_features'][:3])} + more; "
            "then join to outcome on GEOID"
        )

    return ImplementationSpec(
        candidate_id=candidate.candidate_id,
        cloud_safe=candidate.cloud_safe,
        automation_risk=candidate.automation_risk,
        required_python_extras=required_extras,
        required_secrets=candidate.required_secrets,
        osmnx_feature_plan=osmnx_feature_plan,
        data_acquisition_steps=[
            DataAcquisitionStep(
                source_name=candidate.exposure_source,
                source_role="exposure",
                method=exposure_method,
                expected_files=exposure_files,
                required_secrets=candidate.required_secrets,
            ),
            DataAcquisitionStep(
                source_name=candidate.outcome_source,
                source_role="outcome",
                method=outcome_method,
                expected_files=outcome_files,
            ),
        ],
        feature_engineering_steps=[
            FeatureEngineeringStep(
                step_id="join_features_outcome",
                input_sources=[candidate.exposure_source, candidate.outcome_source],
                output_features=candidate.exposure_variables,
                library_tags=feature_step_libs,
                pseudo_code=feature_step_pseudo,
            )
        ],
        analysis_steps=[
            AnalysisStep(
                method=candidate.method_template,
                formula_or_model=f"{candidate.outcome_family} ~ {' + '.join(candidate.exposure_variables[:2] or ['exposure'])} + controls",
                robustness_checks=["alternative_exposure_definitions", "clustered_standard_errors"],
            )
        ],
        expected_outputs=[
            "data/processed/tract_features.csv",
            "output/tables/model_summary.csv",
            "output/report/technical_summary.md",
        ],
        smoke_test_plan=[
            "Run on Cambridge, Massachusetts",
            "Verify non-empty tract_features.csv",
            "Verify outcome column exists",
        ],
    )
