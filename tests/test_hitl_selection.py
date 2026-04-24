import json
import os

from agents import settings
from agents.hitl_helpers import apply_idea_selection


def test_apply_idea_selection_updates_screening_and_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-1"
    token = settings.activate_run_scope(run_id)
    try:
        screening_path = settings.topic_screening_path()
        os.makedirs(os.path.dirname(screening_path), exist_ok=True)
        screening = {
            "run_id": run_id,
            "candidates": [
                {"title": "Topic A", "rank": 1, "topic_id": "a", "exposure_variable": "X1", "outcome_variable": "Y1", "method": "fixed_effects", "geography": "US", "declared_sources": ["S1"]},
                {"title": "Topic B", "rank": 2, "topic_id": "b", "exposure_variable": "X2", "outcome_variable": "Y2", "method": "fixed_effects", "geography": "US", "declared_sources": ["S2"]},
            ],
        }
        with open(screening_path, "w", encoding="utf-8") as f:
            json.dump(screening, f)

        plan_path = settings.research_plan_path()
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        context_path = settings.research_context_path()
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        selected = apply_idea_selection(1)
        assert selected == "Topic B"

        with open(screening_path, "r", encoding="utf-8") as f:
            updated = json.load(f)
        assert updated["candidates"][0]["title"] == "Topic B"
        assert updated["candidates"][0]["rank"] == 1

        with open(plan_path, "r", encoding="utf-8") as f:
            updated_plan = json.load(f)
        assert updated_plan["project_title"] == "Topic B"
    finally:
        settings.deactivate_run_scope(token)
