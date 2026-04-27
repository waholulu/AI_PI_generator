from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CandidateEvaluation(BaseModel):
    candidate_id: str
    title: str
    rank: int = 0
    schema_valid: bool
    data_registry_verdict: str
    data_access_verdict: str
    novelty_verdict: str
    identification_verdict: str
    contribution_verdict: str
    overall_verdict: Literal["pass", "warning", "fail"]
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
