import json
import os

from agents import settings
from agents.orchestrator import ResearchState
from agents.data_fetcher_agent import data_fetcher_node


def test_data_fetcher_node_offline(monkeypatch, tmp_path) -> None:
    """Data fetcher wrapper should write data_access_report.json, not mock parquet."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    os.makedirs(tmp_path / "config", exist_ok=True)

    plan = {
        "run_id": "r1",
        "project_title": "Test project",
        "research_question": "Does X affect Y?",
        "short_rationale": "Testing data access report generation.",
        "geography": "US",
        "time_window": "2010-2020",
        "exposure": {"name": "X", "measurement_proxy": "x"},
        "outcome": {"name": "Y", "measurement_proxy": "y"},
        "identification": {"primary_method": "fixed_effects", "key_threats": ["confounding"]},
        "data_sources": [
            {
                "name": "Test Source",
                "access_url": "https://example.org/data.csv",
                "expected_format": "csv",
                "access_notes": "x y us annual",
            }
        ],
        "literature_queries": ["x y", "x data", "y data"],
        "feasibility": {"overall_verdict": "warning"},
    }
    plan_path = settings.research_plan_path()
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    state = ResearchState(
        current_plan_path=plan_path,
        execution_status="fetching",
    )
    new_state = data_fetcher_node(state)

    assert new_state["execution_status"] == "completed"
    assert os.path.exists(settings.data_access_report_path())
    assert not os.path.exists("data/raw/simulated_data.parquet")
    assert os.path.exists(settings.run_index_path())

