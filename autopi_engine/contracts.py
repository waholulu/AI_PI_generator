from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


RunStatus = Literal[
    "active",
    "awaiting_input",
    "completed",
    "completed_with_warnings",
    "failed",
    "aborted",
]

InputKind = Literal[
    "search_plan",
    "candidate_selection",
    "novelty_search_plan",
    "novelty_assessment",
    "novelty_decision",
    "literature_search_plan",
    "research_memo",
]


class SearchPlan(BaseModel):
    purpose: str = Field(min_length=3)
    queries: list[str] = Field(min_length=1, max_length=12)
    per_query_limit: int = Field(default=10, ge=1, le=50)
    final_limit: int = Field(default=25, ge=1, le=100)
    notes: str = ""

    @field_validator("queries")
    @classmethod
    def _clean_queries(cls, value: list[str]) -> list[str]:
        cleaned = [q.strip() for q in value if q and q.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty query is required")
        return cleaned


class CandidateSelection(BaseModel):
    candidate_id: str = Field(min_length=1)
    rationale: str = ""


class SimilarPaperAssessment(BaseModel):
    title: str
    citation_key: str = ""
    overlap: Literal["low", "medium", "high", "direct"] = "medium"
    reason: str = ""


class NoveltyAssessment(BaseModel):
    verdict: Literal["novel", "partially_overlapping", "already_published", "unavailable"]
    confidence: Literal["low", "medium", "high"] = "medium"
    summary: str = Field(min_length=1)
    similar_papers: list[SimilarPaperAssessment] = Field(default_factory=list)
    recommended_action: Literal["proceed", "revise", "reselect", "abort"] = "proceed"


class NoveltyDecision(BaseModel):
    action: Literal["proceed", "reselect", "abort"]
    rationale: str = ""


class StageEvent(BaseModel):
    stage: str
    status: Literal["completed", "awaiting_input", "failed", "skipped"]
    message: str = ""
    artifacts: list[str] = Field(default_factory=list)
    input_kind: str = ""


class PendingAction(BaseModel):
    kind: InputKind
    prompt: str
    reference_files: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    run_id: str
    domain: str
    template_id: str = "built_environment_health"
    status: RunStatus = "awaiting_input"
    current_stage: str = "field_search_plan"
    pending_action: PendingAction | None = None
    stage_history: list[StageEvent] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    selected_candidate_id: str = ""
    created_at: str
    updated_at: str


class SubmitResult(BaseModel):
    run_id: str
    status: RunStatus
    current_stage: str
    accepted_kind: InputKind
    pending_action: PendingAction | None = None


class AdvanceResult(BaseModel):
    run_id: str
    status: RunStatus
    current_stage: str
    completed_stages: list[str] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


def model_dump_jsonable(model: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(model, BaseModel):
        return model.model_dump(mode="json")
    return model
