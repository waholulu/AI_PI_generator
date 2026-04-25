"""Pydantic request/response models for the Auto-PI REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class StartRunRequest(BaseModel):
    domain_input: str = Field(..., description="Research domain description, e.g. 'GeoAI and Urban Planning'")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for checkpointing/resume. Auto-generated if not provided.")
    template_id: Optional[str] = Field(
        default=None, description="Optional research template id, e.g. 'built_environment_health'."
    )
    technology_options: Optional[dict[str, bool]] = Field(
        default=None, description="Per-run technology feature toggles."
    )
    automation_risk_tolerance: str = Field(
        default="low_medium", description="One of low_only | low_medium | experimental."
    )
    cloud_constraints: Optional[dict[str, bool]] = Field(
        default=None, description="Cloud runtime constraints such as no_paid_api/no_gpu."
    )
    enable_experimental: bool = Field(
        default=False, description="Whether to allow experimental high-risk candidate generation."
    )


class RunStatus(BaseModel):
    run_id: str
    thread_id: str
    domain_input: str
    status: str  # starting | running | awaiting_approval | regenerating | completed | failed | aborted
    current_node: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    degraded_nodes: List[str] = []  # non-empty when any agent fell back due to LLM failure
    regeneration_round: int = 0  # how many times topics have been regenerated at HITL


class RunListItem(BaseModel):
    run_id: str
    domain_input: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None


class LogEntry(BaseModel):
    ts: str
    level: str
    logger: str
    msg: str
    run_id: Optional[str] = None
    node: Optional[str] = None


class OutputFile(BaseModel):
    filename: str
    path: str
    size_bytes: int


class OutputsResponse(BaseModel):
    run_id: str
    files: List[OutputFile]


class ApproveRequest(BaseModel):
    action: str = Field(
        default="select",
        description=(
            "HITL action to take. One of: 'select' (pick a topic and continue) "
            "or 'regenerate' (reject all current topics and re-run ideation + validation)."
        ),
    )
    selected_idea_index: Optional[int] = Field(
        default=None,
        description=(
            "0-based index of the candidate idea to promote to rank-1 before resuming. "
            "Required when action='select'. Ignored when action='regenerate'."
        ),
    )

    @model_validator(mode="after")
    def _validate_action(self):
        if self.action not in ("select", "regenerate"):
            raise ValueError(
                f"action must be 'select' or 'regenerate', got {self.action!r}"
            )
        return self


class ApproveResponse(BaseModel):
    run_id: str
    status: str
    message: str
    selected_idea: Optional[str] = None  # title of the idea that will be researched
    regeneration_round: Optional[int] = None  # present when action was 'regenerate'


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"


class Milestone(BaseModel):
    ts: str
    event: str    # pipeline_started | node_completed | hitl_paused | approved | completed | failed
    detail: str


class TechnologyOptions(BaseModel):
    osmnx: bool = True
    remote_sensing: bool = True
    streetview_cv: bool = False
    deep_learning: bool = False


class CloudConstraints(BaseModel):
    no_paid_api: bool = True
    no_raw_image_storage: bool = True
    no_manual_download: bool = True
    no_gpu: bool = True


class CandidateCard(BaseModel):
    candidate_id: str
    title: str
    research_question: str
    exposure_label: str
    exposure_source: str
    outcome_label: str
    outcome_source: str
    unit_of_analysis: str
    method: str
    claim_strength: str
    technology_tags: list[str] = Field(default_factory=list)
    required_secrets: list[str] = Field(default_factory=list)
    automation_risk: str = "low"
    scores: dict[str, float] = Field(default_factory=dict)
    gate_status: dict[str, Any] = Field(default_factory=dict)
    repair_history: list[dict[str, Any]] = Field(default_factory=list)
    development_pack_status: str = "not_generated"


class DevelopmentPackFile(BaseModel):
    filename: str
    file_type: str
    preview_text: str | None = None
    download_url: str
