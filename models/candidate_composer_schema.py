from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ComposeRequest(BaseModel):
    template_id: str
    domain_input: str
    max_candidates: int = 40
    enable_tier2: bool = True
    enable_experimental: bool = False
    preferred_technology: list[str] = Field(default_factory=list)


class ComposedCandidate(BaseModel):
    candidate_id: str
    template_id: str
    exposure_family: str
    exposure_source: str
    exposure_variables: list[str] = Field(default_factory=list)
    outcome_family: str
    outcome_source: str
    outcome_variables: list[str] = Field(default_factory=list)
    unit_of_analysis: str
    join_plan: dict = Field(default_factory=dict)
    method_template: str
    key_threats: list[str] = Field(default_factory=list)
    mitigations: dict[str, str] = Field(default_factory=dict)
    technology_tags: list[str] = Field(default_factory=list)
    automation_risk: Literal["low", "medium", "high"] = "low"
