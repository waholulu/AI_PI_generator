from typing import TypedDict
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
    literature_inventory_path: str
    draft_content_path: str
    raw_data_manifest_path: str
    research_context_path: str
    execution_status: str


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


def build_orchestrator():
    """Builds and compiles the LangGraph workflow."""
    from agents.field_scanner_agent import field_scanner_node
    from agents.ideation_agent import ideation_node
    from agents.literature_agent import literature_node
    from agents.drafter_agent import drafter_node
    from agents.data_fetcher_agent import data_fetcher_node

    builder = StateGraph(ResearchState)

    # Add actual nodes
    builder.add_node("field_scanner", field_scanner_node)
    builder.add_node("ideation", ideation_node)
    builder.add_node("literature", literature_node)
    builder.add_node("drafter", drafter_node)
    builder.add_node("data_fetcher", data_fetcher_node)

    # Define edges (linear default path)
    builder.add_edge(START, "field_scanner")
    builder.add_edge("field_scanner", "ideation")
    builder.add_edge("ideation", "literature")
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
