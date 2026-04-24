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
            agent_v0 = IdeationAgentV0()
            return agent_v0.run(state)
        raise
