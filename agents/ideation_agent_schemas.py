"""
Pydantic schemas extracted from ideation_agent.py for Module 1 upgrade.

Re-exported here so existing imports (tests, downstream agents) remain valid
while ideation_agent.py is refactored to a thin router.
"""

from typing import List

from pydantic import BaseModel, Field


class LightCandidateTopic(BaseModel):
    title: str = Field(description="The formal title of the research topic.")
    brief_rationale: str = Field(
        description=(
            "A concise 1-2 sentence rationale explaining why this topic is impactful, novel, "
            "and feasible with public quantitative data. Hint at method type, data source, and gap filled."
        )
    )


class LightCandidateTopicsList(BaseModel):
    candidates: List[LightCandidateTopic]


class TopicScore(BaseModel):
    title: str = Field(description="Title of the topic being scored.")
    score: int = Field(description="Overall quality score out of 100.")
    passed_gates: bool = Field(
        description="Whether the topic passes all hard gates (impact, quantitative, novelty, etc.)."
    )
    rejection_reason: str = Field(
        description="If passed_gates is false, strictly explain which gate it failed and why. Otherwise, empty string.",
        default="",
    )
    rank: int = Field(
        description="Final ranking position among passed topics (1 = best). Set to 0 if passed_gates is false.",
        default=0,
    )


class TopicScoresList(BaseModel):
    scores: List[TopicScore]


class QuantitativeSpecs(BaseModel):
    unit_of_analysis: str = Field(description="Sample unit and spatial/time scale.")
    outcomes: List[str] = Field(description="Clear dependent variables constructable from public data.")
    exposures: List[str] = Field(description="Clear core explanatory/treatment variables.")
    estimand_and_strategy: str = Field(description="Estimand and identification strategy/assumptions.")
    model_family: str = Field(description="Specific, scriptable model family.")
    robustness_checks: List[str] = Field(description="At least 6 scriptable robustness/heterogeneity checks.")
    expected_tables_figures: List[str] = Field(description="Expected table/figure types and required statistics.")


class DataSource(BaseModel):
    name: str = Field(description="Name of the public data source.")
    accessibility: str = Field(description="Why this data is freely accessible without paywalls.")


class RawCandidateTopic(BaseModel):
    title: str = Field(description="The formal title of the research topic (keep exactly as given).")
    impact_evidence: str = Field(description="Realistic justification of impact.")
    novelty_gap_type: str = Field(description="Specific type of gap this fills.")
    publishability: str = Field(description="2-3 specific target journals with brief matching justification.")
    quantitative_specs: QuantitativeSpecs
    data_sources: List[DataSource]


class ResearchPlanSchema(BaseModel):
    project_title: str
    study_type: str
    topic_screening: dict
    research_questions: List[str]
    hypotheses: List[str]
    unit_of_analysis: str
    outcomes: List[dict]
    exposures: List[dict]
    keywords: List[str]
    data_sources: List[dict]
    methodology: dict


class NoveltyQueryPlan(BaseModel):
    queries: List[str] = Field(description="2-3 precise OpenAlex queries for novelty check.")


class PaperOverlapAssessment(BaseModel):
    paper_openalex_id: str = Field(description="OpenAlex ID of the paper being assessed.")
    overlap_score: int = Field(description="0-100 overlap score. 100 means almost identical research contribution.")
    rationale: str = Field(description="Short explanation of overlap judgment.")


class NoveltyAssessment(BaseModel):
    novelty_verdict: str = Field(description="One of: novel, partially_overlapping, already_published.")
    assessments: List[PaperOverlapAssessment] = Field(description="Per-paper overlap assessments.")
