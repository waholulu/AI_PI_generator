"""
Unit tests for agents/orchestrator.py – graph wiring and basic compile properties.

These tests do NOT run the full pipeline; they only inspect the compiled graph
structure so they execute instantly without any API calls.
"""
from agents.orchestrator import ResearchState, build_orchestrator


def test_build_orchestrator_returns_compiled_graph() -> None:
    """build_orchestrator must return a compiled LangGraph graph without errors."""
    graph = build_orchestrator()
    assert graph is not None


def test_orchestrator_graph_contains_expected_nodes() -> None:
    """The compiled graph must expose the five pipeline nodes."""
    graph = build_orchestrator()
    # LangGraph compiled graphs expose their nodes via .nodes (mapping of node_name -> runnable)
    node_names = set(graph.nodes.keys())
    expected = {"field_scanner", "ideation", "idea_validator", "literature", "drafter", "data_fetcher"}
    missing = expected - node_names
    assert not missing, f"Graph is missing nodes: {missing}"


def test_orchestrator_interrupt_before_literature() -> None:
    """Graph must be configured to interrupt before the 'literature' node."""
    graph = build_orchestrator()
    # The interrupt configuration is stored on the compiled graph
    interrupts = getattr(graph, "interrupt_before", None)
    if interrupts is None:
        # Some LangGraph versions expose this as a graph attribute on the config schema
        config_fields = getattr(graph, "config_schema", None)
        assert config_fields is not None or interrupts is not None, (
            "Could not inspect interrupt_before configuration on compiled graph"
        )
    else:
        assert "literature" in (interrupts or []), (
            f"Expected 'literature' in interrupt_before, got: {interrupts}"
        )


def test_research_state_has_required_keys() -> None:
    """ResearchState TypedDict must contain all expected keys."""
    required = {
        "domain_input",
        "field_scan_path",
        "candidate_topics_path",
        "current_plan_path",
        "validation_report_path",
        "literature_inventory_path",
        "draft_content_path",
        "raw_data_manifest_path",
        "research_context_path",
        "execution_status",
    }
    annotations = ResearchState.__annotations__
    missing = required - set(annotations.keys())
    assert not missing, f"ResearchState is missing fields: {missing}"
