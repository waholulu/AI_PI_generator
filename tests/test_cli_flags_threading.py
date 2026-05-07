"""Tests that CLI flags (--budget-override-usd, --skip-reflection) are threaded
through the initial state correctly.

Note: these flags are kept in state for backward compatibility and future use
by the Candidate Factory pipeline, but ideation_node() no longer routes to
IdeationAgentV2 by default.  The test verifies the state-passing contract.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch


def test_cli_flags_threaded_into_initial_state():
    """--budget-override-usd and --skip-reflection must appear in initial_state passed to graph."""
    fake_graph_mod = types.ModuleType("langgraph.graph")
    fake_graph_mod.StateGraph = object
    fake_graph_mod.START = "START"
    fake_graph_mod.END = "END"
    sys.modules.setdefault("langgraph.graph", fake_graph_mod)

    sys.modules.setdefault(
        "dotenv",
        types.SimpleNamespace(load_dotenv=lambda: None),
    )
    import main

    argv = [
        "main.py",
        "--mode",
        "level_2",
        "--domain",
        "Urban Health",
        "--budget-override-usd",
        "0.50",
        "--skip-reflection",
    ]

    captured_state = {}

    class DummyGraph:
        def stream(self, input_state, config):
            captured_state.update(input_state or {})
            return [{"ideation": {"execution_status": "harvesting"}}]

        def get_state(self, config):
            class Snap:
                next = ()
                values = {"execution_status": "harvesting"}

            return Snap()

    with patch.object(sys, "argv", argv):
        with patch("agents.orchestrator.build_orchestrator", return_value=DummyGraph()):
            main.main()

    assert captured_state["budget_override_usd"] == 0.50
    assert captured_state["skip_reflection"] is True


def test_ideation_node_routes_to_candidate_factory_not_v2():
    """ideation_node() must call run_candidate_factory_ideation, NOT IdeationAgentV2."""
    from agents.ideation_agent import ideation_node

    factory_called = []

    with patch(
        "agents.candidate_factory_ideation.run_candidate_factory_ideation",
        side_effect=lambda s: factory_called.append(True) or {"execution_status": "ideation_complete"},
    ):
        with patch("agents.ideation_agent_v2.IdeationAgentV2") as mock_v2:
            ideation_node(
                {
                    "domain_input": "Urban Health",
                    "degraded_nodes": [],
                    "budget_override_usd": 0.50,
                    "skip_reflection": True,
                }
            )

    assert factory_called, "Candidate Factory must be called"
    mock_v2.assert_not_called()
