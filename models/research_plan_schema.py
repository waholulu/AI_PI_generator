from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VariableSpec(BaseModel):
    name: str
    definition: str = ""
    measurement_proxy: str = ""
    spatial_unit: str = ""
    temporal_frequency: str = ""
    data_source_names: list[str] = Field(default_factory=list)


class DataSourceSpec(BaseModel):
    name: str
    role: Literal[
        "exposure",
        "outcome",
        "control",
        "boundary",
        "literature",
        "image_provider",
        "feature_provider",
    ] = "control"
    source_type: Literal["api", "download", "registry", "manual", "unknown"] = "unknown"
    access_url: str = ""
    documentation_url: str = ""
    license: str = ""
    expected_format: str = ""
    access_notes: str = ""
    machine_readable: bool = False
    auth_required: bool = False
    cost_required: bool = False
    covers_variable_families: list[str] = Field(default_factory=list)
    spatial_units: list[str] = Field(default_factory=list)
    join_keys: list[str] = Field(default_factory=list)
    technology_tags: list[str] = Field(default_factory=list)


class IdentificationSpec(BaseModel):
    primary_method: str
    causal_claim_strength: Literal["descriptive", "associational", "quasi_causal", "causal"] = (
        "associational"
    )
    key_threats: list[str] = Field(default_factory=list)
    mitigations: dict[str, str] = Field(default_factory=dict)


class FeasibilitySpec(BaseModel):
    overall_verdict: Literal["pass", "warning", "fail"] = "warning"
    data_verdict: Literal["pass", "warning", "fail"] = "warning"
    novelty_verdict: Literal["novel", "partially_overlapping", "already_published", "unknown"] = "unknown"
    identification_verdict: Literal["pass", "warning", "fail"] = "warning"
    main_risks: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    run_id: str
    project_title: str
    research_question: str
    short_rationale: str
    population: str = ""
    geography: str = ""
    time_window: str = ""
    unit_of_analysis: str = ""
    exposure: VariableSpec
    outcome: VariableSpec
    identification: IdentificationSpec
    data_sources: list[DataSourceSpec] = Field(default_factory=list)
    literature_queries: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    feasibility: FeasibilitySpec
    selected_candidate_id: str = ""
