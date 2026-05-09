import operator
from typing import Annotated, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
import sqlite3

from agents import settings
from agents.logging_config import get_logger

logger = get_logger(__name__)


class ResearchState(TypedDict):
    """
    Core state for the Auto-PI multi-agent system.
    Using pass-by-reference to prevent memory bloating.
    """
    domain_input: str
    field_scan_path: str
    candidate_topics_path: str
    current_plan_path: str
    validation_report_path: str
    literature_inventory_path: str
    draft_content_path: str
    raw_data_manifest_path: str
    research_context_path: str
    execution_status: str
    # Accumulates across nodes: each entry is "<node>:<reason>" (e.g. "ideation:plan_placeholder").
    # Uses operator.add so each node's list is appended, not replaced.
    degraded_nodes: Annotated[list[str], operator.add]


class _Module1State(ResearchState, total=False):
    """Extension fields added by the Module 1 upgrade.

    Defined with total=False so existing code that builds ResearchState without
    these keys continues to work — they are optional.
    """
    legacy_ideation: bool          # True → use IdeationAgentV0 (legacy path)
    user_topic_path: Optional[str] # Path to user-supplied topic YAML (Level 1 mode)
    ideation_mode: str             # "level_1" | "level_2"
    budget_override_usd: Optional[float]
    skip_reflection: bool
    hitl_interruption: dict
    template_id: Optional[str]
    runtime_tier: Optional[str]
    technology_options: dict
    automation_risk_tolerance: str
    cloud_constraints: dict
    enable_experimental: bool
    candidate_factory_enabled: bool
    # Candidate factory output paths — populated by ideation node
    candidate_cards_path: str
    feasibility_report_path: str
    repair_history_path: str
    development_pack_index_path: str
    gate_trace_path: str
    # Selection
    selected_candidate_id: str
    # Stage 2: novelty_check verdict drives the conditional second HITL
    novelty_verdict: str          # novel | partially_overlapping | already_published | unavailable
    novelty_search_terms: list


def _build_checkpointer():
    """Build a LangGraph checkpointer based on DATABASE_URL env var."""
    if settings.is_postgres():
        from langgraph.checkpoint.postgres import PostgresSaver
        return PostgresSaver.from_conn_string(settings.get_db_url())
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = settings.checkpoints_db_path()
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)


def _route_start(state: dict) -> str:
    """Conditional router: skip field_scanner when a user topic YAML is provided."""
    if state.get("user_topic_path"):
        logger.info("Level 1 mode: skipping field_scanner, routing direct to ideation")
        return "ideation"
    return "field_scanner"


def build_orchestrator():
    """Builds and compiles the two-stage candidate flow.

    Stage 1 (always-on, cheap): field_scanner → ideation → [HITL#1 pause]
        - ideation is the candidate factory by default; emits a diverse
          shortlist with research-question titles and `pending_literature_check`
          novelty placeholders.
        - HITL#1 fires before novelty_check.  The user picks one candidate,
          which `apply_idea_selection_*` promotes to rank-1.

    Stage 2 (after selection, on the picked candidate only):
        novelty_check → (conditional) → novelty_review [HITL#2 if not_novel]
                                  OR  → literature → development_pack
                                                  → drafter → data_fetcher → END
    """
    from agents.field_scanner_agent import field_scanner_node
    from agents.ideation_agent import ideation_node
    from agents.literature_agent import literature_node
    from agents.drafter_agent import drafter_node
    from agents.data_fetcher_agent import data_fetcher_node
    from agents.stage2_nodes import (
        novelty_check_node,
        novelty_review_node,
        development_pack_node,
        route_after_novelty,
    )

    builder = StateGraph(_Module1State)

    # Stage 1
    builder.add_node("field_scanner", field_scanner_node)
    builder.add_node("ideation", ideation_node)
    # Stage 2
    builder.add_node("novelty_check", novelty_check_node)
    builder.add_node("novelty_review", novelty_review_node)
    builder.add_node("literature", literature_node)
    builder.add_node("development_pack", development_pack_node)
    builder.add_node("drafter", drafter_node)
    builder.add_node("data_fetcher", data_fetcher_node)

    # Conditional fork at START: Level 1 skips field_scanner
    builder.add_conditional_edges(
        START,
        _route_start,
        {"field_scanner": "field_scanner", "ideation": "ideation"},
    )

    # Stage 1 linear edges
    builder.add_edge("field_scanner", "ideation")
    builder.add_edge("ideation", "novelty_check")

    # Conditional after novelty_check: pause at novelty_review only when
    # the verdict is `already_published` (HITL#2); otherwise continue.
    builder.add_conditional_edges(
        "novelty_check",
        route_after_novelty,
        {"novelty_review": "novelty_review", "literature": "literature"},
    )
    builder.add_edge("novelty_review", "literature")

    # Stage 2 tail
    builder.add_edge("literature", "development_pack")
    builder.add_edge("development_pack", "drafter")
    builder.add_edge("drafter", "data_fetcher")
    builder.add_edge("data_fetcher", END)

    memory = _build_checkpointer()

    # HITL#1 fires before novelty_check (Stage 1 done, awaiting user pick).
    # HITL#2 fires before novelty_review (only when already_published).
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["novelty_check", "novelty_review"],
    )
    return graph

if __name__ == "__main__":
    graph = build_orchestrator()
    logger.info("Orchestrator graph wired and compiled successfully.")
