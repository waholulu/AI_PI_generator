"""Pydantic request/response models for the Auto-PI REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class StartRunRequest(BaseModel):
    domain_input: str = Field(..., description="Research domain description, e.g. 'GeoAI and Urban Planning'")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for checkpointing/resume. Auto-generated if not provided.")


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
