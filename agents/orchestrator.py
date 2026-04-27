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
    """Builds and compiles the LangGraph workflow."""
    from agents.field_scanner_agent import field_scanner_node
    from agents.ideation_agent import ideation_node
    from agents.idea_validator_agent import idea_validator_node
    from agents.literature_agent import literature_node
    from agents.drafter_agent import drafter_node
    from agents.data_fetcher_agent import data_fetcher_node

    builder = StateGraph(_Module1State)

    # Add actual nodes
    builder.add_node("field_scanner", field_scanner_node)
    builder.add_node("ideation", ideation_node)
    builder.add_node("idea_validator", idea_validator_node)
    builder.add_node("literature", literature_node)
    builder.add_node("drafter", drafter_node)
    builder.add_node("data_fetcher", data_fetcher_node)

    # Conditional fork at START: Level 1 skips field_scanner
    builder.add_conditional_edges(
        START,
        _route_start,
        {"field_scanner": "field_scanner", "ideation": "ideation"},
    )

    # Remaining linear edges
    builder.add_edge("field_scanner", "ideation")
    builder.add_edge("ideation", "idea_validator")
    builder.add_edge("idea_validator", "literature")
    builder.add_edge("literature", "drafter")
    builder.add_edge("drafter", "data_fetcher")
    builder.add_edge("data_fetcher", END)

    # Establish Checkpointer
    memory = _build_checkpointer()

    # Compile graph with HITL after ideation
    graph = builder.compile(checkpointer=memory, interrupt_before=["literature"])
    return graph

if __name__ == "__main__":
    graph = build_orchestrator()
    logger.info("Orchestrator graph wired and compiled successfully.")
