import json
import os

from agents import settings
from agents.data_access_agent import data_access_node
from agents.orchestrator import ResearchState


def test_data_access_agent_writes_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    os.makedirs(tmp_path / "config", exist_ok=True)

    plan = {
        "run_id": "run-123",
        "project_title": "Air pollution and mortality",
        "research_question": "How does PM2.5 affect mortality?",
        "short_rationale": "Policy-relevant topic.",
        "geography": "US",
        "time_window": "2010-2020",
        "exposure": {"name": "PM2.5", "measurement_proxy": "pm2.5"},
        "outcome": {"name": "Mortality", "measurement_proxy": "mortality"},
        "identification": {"primary_method": "fixed_effects", "key_threats": ["confounding"]},
        "data_sources": [
            {
                "name": "EPA",
                "access_url": "https://example.org/pm25.csv",
                "expected_format": "csv",
                "access_notes": "pm2.5 US county annual",
            },
            {
                "name": "CDC",
                "access_url": "https://example.org/mortality.csv",
                "expected_format": "csv",
                "access_notes": "mortality US county annual",
            },
        ],
        "literature_queries": ["a", "b", "c"],
        "feasibility": {"overall_verdict": "warning"},
    }
    plan_path = settings.research_plan_path()
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    state = ResearchState(current_plan_path=plan_path, execution_status="fetching")
    out = data_access_node(state)
    assert out["execution_status"] == "completed"

    report_path = settings.data_access_report_path()
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["run_id"] == "run-123"
    assert "overall_verdict" in report
