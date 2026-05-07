"""
Thin router for the ideation node.

Routes to:
  - IdeationAgentV0 (legacy): when state["legacy_ideation"] is True
    OR environment variable LEGACY_IDEATION=1
  - IdeationAgentV2 (new): default path

All Pydantic schemas previously defined here have been moved to
agents/ideation_agent_schemas.py and are re-exported below for backward
compatibility with existing tests and downstream importers.
"""

import os
import json
from datetime import datetime, timezone
from uuid import uuid4

from agents.ideation_agent_schemas import (  # noqa: F401  (re-exports for compat)
    DataSource,
    LightCandidateTopic,
    LightCandidateTopicsList,
    NoveltyAssessment,
    NoveltyQueryPlan,
    PaperOverlapAssessment,
    QuantitativeSpecs,
    RawCandidateTopic,
    ResearchPlanSchema,
    TopicScore,
    TopicScoresList,
)
from agents.logging_config import get_logger
from agents.orchestrator import ResearchState
from models.topic_schema import HITLInterruption

logger = get_logger(__name__)


def _emit_emergency_fallback_outputs(state: ResearchState) -> dict:
    """Last-resort offline fallback so tests/cloud runs don't crash without LLM keys."""
    from agents import settings as _settings
    from agents.research_plan_builder import build_research_plan_from_candidate

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    domain = state.get("domain_input", "Built environment and health")
    candidate = {
        "candidate_id": "fallback_001",
        "topic_id": "fallback_001",
        "title": f"{domain}: Street Connectivity and Physical Inactivity",
        "research_question": "Are more connected street networks associated with lower physical inactivity?",
        "exposure_variable": "street_connectivity",
        "outcome_variable": "physical_inactivity",
        "geography": "United States",
        "method": "cross_sectional_spatial_association",
        "data_sources": [
            {"name": "OSMnx_OpenStreetMap", "source_type": "api"},
            {"name": "CDC_PLACES", "source_type": "download"},
            {"name": "ACS", "source_type": "api"},
            {"name": "TIGER_Lines", "source_type": "download"},
        ],
        "brief_rationale": "Fallback deterministic candidate when ideation LLM is unavailable.",
        "rank": 1,
        "evaluation": {"overall_verdict": "warning", "score": 0.0},
    }

    screening = {
        "run_id": run_id,
        "input_mode": "level_2",
        "ideation_mode": "fallback",
        "domain": domain,
        "candidates": [candidate],
        "fallback_reason": "llm_unavailable_and_legacy_fallback_failed",
    }
    screening_path = _settings.topic_screening_path()
    with open(screening_path, "w", encoding="utf-8") as f:
        json.dump(screening, f, indent=2, ensure_ascii=False)

    plan = build_research_plan_from_candidate(candidate, candidate.get("evaluation"), run_id=run_id)
    plan_path = _settings.research_plan_path()
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False)

    context_path = _settings.research_context_path()
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump({"domain": domain, "run_id": run_id, "selected_topic": candidate}, f, indent=2, ensure_ascii=False)

    logger.warning("Emergency ideation fallback emitted deterministic candidate outputs.")
    return {
        "execution_status": "harvesting",
        "candidate_topics_path": screening_path,
        "current_plan_path": plan_path,
        "research_context_path": context_path,
        "degraded_nodes": ["ideation"],
    }


_DEFAULT_TEMPLATE_ID = "built_environment_health"


def ideation_node(state: ResearchState) -> dict:
    """LangGraph node: route to Candidate Factory (default) or legacy V0.

    Routing priority:
      1. Explicit legacy mode (--legacy-ideation or LEGACY_IDEATION=1) → IdeationAgentV0
      2. Everything else → Candidate Factory with template_id (defaults to
         built_environment_health when not specified in state).

    The V2 LLM ideation path is intentionally NOT reachable from this router.
    Candidate Factory is the production default; V2 is only accessible directly.
    """
    use_legacy = (
        state.get("legacy_ideation", False)
        or os.getenv("LEGACY_IDEATION", "0").strip() not in ("", "0", "false", "False")
    )

    if use_legacy:
        logger.info("Routing to IdeationAgentV0 (explicit legacy mode)")
        from agents._legacy.ideation_agent_v0 import IdeationAgentV0
        agent = IdeationAgentV0()
        return agent.run(state)

    template_id = state.get("template_id") or _DEFAULT_TEMPLATE_ID
    logger.info(
        "Routing to Candidate Factory ideation (template: %s%s)",
        template_id,
        " [default]" if not state.get("template_id") else "",
    )
    from agents.candidate_factory_ideation import run_candidate_factory_ideation
    return run_candidate_factory_ideation({**state, "template_id": template_id})
