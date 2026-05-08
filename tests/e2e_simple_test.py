import json
import os
import pytest
from agents.orchestrator import build_orchestrator, ResearchState


@pytest.mark.slow
def test_full_pipeline(tmp_path, monkeypatch):
    """Runs a full simulated run to ensure no crashes."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    (tmp_path / "output").mkdir()

    graph = build_orchestrator()
    config = {"configurable": {"thread_id": "test_e2e_1"}}

    initial_state = ResearchState(
        domain_input="Test Domain GeoAI",
        execution_status="starting"
    )

    # Stage 1: run until HITL#1 pause (before novelty_check)
    print("Running until first interrupt (after ideation)...")
    for event in graph.stream(initial_state, config, stream_mode="values"):
        print(f"Current State: {event.get('execution_status')}")

    # Simulate HITL#1: select the rank-1 candidate from the screening file
    from agents.hitl_helpers import apply_idea_selection
    apply_idea_selection(0)

    # Stage 2: resume and run through to completion
    print("Resuming graph from interrupt...")
    for event in graph.stream(None, config, stream_mode="values"):
        pass

    final_state = graph.get_state(config)
    assert final_state.values.get("execution_status") == "completed"

    print("E2E pipeline executed successfully.")


if __name__ == "__main__":
    test_full_pipeline()
