from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ComposeRequest(BaseModel):
    template_id: str
    domain_input: str
    max_candidates: int = 40
    enable_tier2: bool = True
    enable_experimental: bool = False
    no_paid_api: bool = True
    no_manual_download: bool = True
    preferred_technology: list[str] = Field(default_factory=list)
    automation_risk_tolerance: Literal["low_only", "low_medium", "experimental"] = "low_medium"
    # When True, the LLM task-seed generator may propose tasks backed by
    # credentialed datasets (MIMIC, UK Biobank, i2b2, …). Candidates that
    # rely on such data are labelled `requires_credentialing=True` instead
    # of being filtered out. Default False preserves the Colab-runnable,
    # no-sign-up promise.
    allow_credentialed_data: bool = False


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
    claim_strength: str = "associational"
    key_threats: list[str] = Field(default_factory=list)
    mitigations: dict[str, str] = Field(default_factory=dict)
    technology_tags: list[str] = Field(default_factory=list)
    required_secrets: list[str] = Field(default_factory=list)
    automation_risk: Literal["low", "medium", "high"] = "low"
    cloud_safe: bool = True
    # Composer-level shortlist hint; feasibility precheck may override to "blocked"
    initial_shortlist_status: Literal["ready", "review", "blocked"] = "ready"

    # Domain-derived task fields (training_research templates only). When set,
    # outcome_family is the *metric family* (task_accuracy / instruction_following /
    # generation_quality) and these carry the concrete supervised task derived
    # from the user's domain_input. Leave None for spatial templates.
    outcome_task_id: str | None = None
    outcome_task_label: str | None = None
    outcome_task_description: str | None = None
    outcome_task_modality: str | None = None
    outcome_task_dataset_hint: str | None = None
    outcome_task_domain_input: str | None = None
    # Set when the seed references a credentialed dataset (MIMIC, UKB, i2b2,
    # …) and `allow_credentialed_data` opt-in was enabled. UI shows a
    # warning banner and the dev pack includes access-request instructions.
    requires_credentialing: bool = False
    credentialing_note: str | None = None
