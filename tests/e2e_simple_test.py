import os
from agents.orchestrator import build_orchestrator, ResearchState

def test_full_pipeline():
    """Runs a full simulated run to ensure no crashes."""
    graph = build_orchestrator()
    config = {"configurable": {"thread_id": "test_e2e_1"}}
    
    # Initialize state
    initial_state = ResearchState(
        domain_input="Test Domain GeoAI",
        execution_status="starting"
    )
    
    # Stream events up to the interrupt
    print("Running until first interrupt (after ideation)...")
    for event in graph.stream(initial_state, config, stream_mode="values"):
        print(f"Current State: {event.get('execution_status')}")
        
    print("Resuming graph from interrupt...")
    for event in graph.stream(None, config, stream_mode="values"):
        pass

    final_state = graph.get_state(config)
    assert final_state.values.get("execution_status") == "completed"
    
    print("E2E pipeline executed successfully.")

if __name__ == "__main__":
    test_full_pipeline()
