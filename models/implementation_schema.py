from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DataAcquisitionStep(BaseModel):
    source_name: str
    source_role: str
    method: Literal["download", "api", "osmnx", "manual_placeholder"]
    url: str = ""
    required_secrets: list[str] = Field(default_factory=list)
    expected_files: list[str] = Field(default_factory=list)
    cache_policy: str = "cache_processed_only"


class FeatureEngineeringStep(BaseModel):
    step_id: str
    input_sources: list[str] = Field(default_factory=list)
    output_features: list[str] = Field(default_factory=list)
    library_tags: list[str] = Field(default_factory=list)
    pseudo_code: str


class AnalysisStep(BaseModel):
    method: str
    formula_or_model: str
    robustness_checks: list[str] = Field(default_factory=list)


class SourceUseSpec(BaseModel):
    """Semantic description of how one data source is used in this candidate.

    Captures native grain, target grain, the aggregation recipe, the actual
    column names, and any known limitations — giving Claude Code enough context
    to write correct acquisition and feature-engineering code without guessing.
    """

    source_id: str
    role: str
    native_unit: str = ""
    target_unit: str = ""
    raw_columns: list[str] = Field(default_factory=list)
    derived_features: list[str] = Field(default_factory=list)
    acquisition_method: str = ""
    acquisition_url: str = ""
    join_recipe: dict | None = None
    aggregation_method: str = ""
    validation_rules: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)


class ImplementationSpec(BaseModel):
    candidate_id: str
    cloud_safe: bool
    automation_risk: Literal["low", "medium", "high"]
    required_python_extras: list[str] = Field(default_factory=list)
    required_secrets: list[str] = Field(default_factory=list)
    # Populated only for OSMnx/street-network candidates; None otherwise
    osmnx_feature_plan: dict | None = None
    data_acquisition_steps: list[DataAcquisitionStep] = Field(default_factory=list)
    feature_engineering_steps: list[FeatureEngineeringStep] = Field(default_factory=list)
    analysis_steps: list[AnalysisStep] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    smoke_test_plan: list[str] = Field(default_factory=list)
    # Data catalog-aware fields
    source_use_specs: list[SourceUseSpec] = Field(default_factory=list)
    data_lineage_plan: dict = Field(default_factory=dict)
