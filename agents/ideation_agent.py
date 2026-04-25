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


def ideation_node(state: ResearchState) -> dict:
    """LangGraph node: route to legacy V0 or new V2 ideation agent."""
    use_legacy = (
        state.get("legacy_ideation", False)
        or os.getenv("LEGACY_IDEATION", "0").strip() not in ("", "0", "false", "False")
    )

    if use_legacy:
        logger.info("Routing to IdeationAgentV0 (legacy mode)")
        from agents._legacy.ideation_agent_v0 import IdeationAgentV0
        agent = IdeationAgentV0()
        return agent.run(state)

    logger.info("Routing to IdeationAgentV2")
    from agents.ideation_agent_v2 import IdeationAgentV2
    budget_override_usd = state.get("budget_override_usd")
    skip_reflection = state.get("skip_reflection", False)
    agent = IdeationAgentV2(
        budget_override_usd=budget_override_usd,
        skip_reflection=skip_reflection,
    )
    try:
        return agent.run(state)
    except HITLInterruption as e:
        return {
            "execution_status": "hitl_required",
            "hitl_interruption": {
                "kind": e.kind,
                "message": e.message,
                "failed_gates": e.failed_gates,
                "suggested_operations": e.suggested_operations,
                "diff_from_original": e.diff_from_original,
                "suggested_next_operations": e.suggested_next_operations,
            },
        }
    except Exception as e:
        from agents import settings as _settings
        cfg_path = _settings.reflection_config_path()
        try:
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            fallback_enabled = cfg.get("feature_flags", {}).get(
                "legacy_fallback_on_error", True
            )
        except Exception:
            fallback_enabled = True

        if fallback_enabled:
            logger.warning(
                "IdeationAgentV2 failed (%s); falling back to V0 (legacy_fallback_on_error=True)",
                e,
            )
            from agents._legacy.ideation_agent_v0 import IdeationAgentV0
            try:
                agent_v0 = IdeationAgentV0()
                return agent_v0.run(state)
            except Exception as legacy_exc:
                logger.warning(
                    "Legacy IdeationAgentV0 also failed (%s); using deterministic emergency fallback.",
                    legacy_exc,
                )
                return _emit_emergency_fallback_outputs(state)
        raise
