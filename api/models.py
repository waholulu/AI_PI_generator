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
    status: str  # starting | running | awaiting_approval | completed | failed | aborted
    current_node: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    degraded_nodes: List[str] = []  # non-empty when any agent fell back due to LLM failure


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
    selected_idea_index: Optional[int] = Field(
        default=None,
        description=(
            "0-based index of the candidate idea to promote to rank-1 before resuming. "
            "If omitted or None the current top-1 is kept unchanged."
        ),
    )


class ApproveResponse(BaseModel):
    run_id: str
    status: str
    message: str
    selected_idea: Optional[str] = None  # title of the idea that will be researched


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"


class Milestone(BaseModel):
    ts: str
    event: str    # pipeline_started | node_completed | hitl_paused | approved | completed | failed
    detail: str
