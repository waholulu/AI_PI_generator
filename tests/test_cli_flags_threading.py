from __future__ import annotations

import sys
import types
from unittest.mock import patch


def test_cli_flags_thread_to_ideation_agent_v2_constructor():
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

    from agents.ideation_agent import ideation_node

    with patch("agents.ideation_agent_v2.IdeationAgentV2") as mock_v2:
        mock_v2.return_value.run.return_value = {"execution_status": "harvesting"}
        ideation_node(
            {
                "domain_input": "Urban Health",
                "degraded_nodes": [],
                "budget_override_usd": 0.50,
                "skip_reflection": True,
            }
        )

    mock_v2.assert_called_once_with(
        budget_override_usd=0.50,
        skip_reflection=True,
    )
