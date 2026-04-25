from __future__ import annotations

from models.candidate_composer_schema import ComposedCandidate
from models.implementation_schema import (
    AnalysisStep,
    DataAcquisitionStep,
    FeatureEngineeringStep,
    ImplementationSpec,
)


def build_implementation_spec(candidate: ComposedCandidate) -> ImplementationSpec:
    requires_secrets = []
    if candidate.automation_risk == "high":
        requires_secrets = ["OPTIONAL_API_KEY"]

    required_extras = ["geospatial"]
    if "experimental" in candidate.technology_tags:
        required_extras.append("vision")

    return ImplementationSpec(
        candidate_id=candidate.candidate_id,
        cloud_safe=candidate.automation_risk != "high",
        automation_risk=candidate.automation_risk,
        required_python_extras=required_extras,
        required_secrets=requires_secrets,
        data_acquisition_steps=[
            DataAcquisitionStep(
                source_name=candidate.exposure_source,
                source_role="exposure",
                method="osmnx" if "osmnx" in candidate.exposure_source.lower() else "api",
                expected_files=["data/raw/exposure.parquet"],
                required_secrets=requires_secrets,
            ),
            DataAcquisitionStep(
                source_name=candidate.outcome_source,
                source_role="outcome",
                method="download",
                expected_files=["data/raw/outcome.csv"],
            ),
        ],
        feature_engineering_steps=[
            FeatureEngineeringStep(
                step_id="join_features_outcome",
                input_sources=[candidate.exposure_source, candidate.outcome_source],
                output_features=candidate.exposure_variables,
                library_tags=["pandas", "geopandas"],
                pseudo_code="load exposure and outcome, spatially aggregate to tract, join on GEOID",
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
