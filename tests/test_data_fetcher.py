import json
import os

from agents.orchestrator import ResearchState
from agents.data_fetcher_agent import data_fetcher_node


def test_data_fetcher_node_offline() -> None:
    """Ensure data fetcher node runs and writes a well-formed manifest and run index."""
    os.makedirs("config", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    plan = {
        "data_sources": [
            {
                "name": "Test Source",
                "api_endpoint": "https://api.census.gov",
                "data_type": "table",
            }
        ]
    }
    with open("config/research_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f)

    # Minimal research context to populate outcomes/exposures
    os.makedirs("output", exist_ok=True)
    with open("output/research_context.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "plan_essentials": {
                    "outcomes": [{"variable": "Y"}],
                    "exposures": [{"variable": "X"}],
                }
            },
            f,
        )

    state = ResearchState(
        current_plan_path="config/research_plan.json",
        research_context_path="output/research_context.json",
        execution_status="fetching",
    )

    new_state = data_fetcher_node(state)

    assert new_state["execution_status"] == "completed"
    assert os.path.exists("data/raw/manifest.json")
    assert os.path.exists("data/raw/simulated_data.parquet")
    assert os.path.exists("output/run_index.json")

    with open("data/raw/manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    datasets = manifest.get("datasets", [])
    assert len(datasets) == 1
    entry = datasets[0]
    assert entry["source_name"] == "Test Source"
    assert entry["data_type"] == "table"
    assert entry["status"] == "success"
    assert entry["covers_outcomes"] == ["Y"]
    assert entry["covers_exposures"] == ["X"]

