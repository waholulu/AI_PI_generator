import json
import os

from agents import settings
from agents.hitl_helpers import apply_idea_selection, load_validated_topics


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


def _write_validation(report: dict) -> None:
    path = settings.idea_validation_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)


def _write_screening_titles(titles: list) -> None:
    path = settings.topic_screening_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"run_id": "r", "candidates": [{"title": t, "rank": i + 1} for i, t in enumerate(titles)]},
            f,
        )


def test_load_validated_topics_aligns_with_screening_after_substitution(tmp_path, monkeypatch):
    """Pre-substitution failed originals must be dropped, substitutes preserved."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    token = settings.activate_run_scope("r1")
    try:
        # After substitution: candidates list now has Sub-A in slot 0
        _write_screening_titles(["Sub-A", "B"])
        _write_validation({
            "validated_ideas": [
                {"title": "A", "overall_verdict": "failed"},
                {"title": "Sub-A", "overall_verdict": "warning"},
                {"title": "B", "overall_verdict": "passed"},
            ],
        })
        ideas = load_validated_topics()
        assert [i["title"] for i in ideas] == ["Sub-A", "B"]
    finally:
        settings.deactivate_run_scope(token)


def test_load_validated_topics_returns_failed_when_all_failed(tmp_path, monkeypatch):
    """When every candidate failed validation (no substitution possible), the
    picker still returns all ideas so the user can choose or regenerate."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    token = settings.activate_run_scope("r2")
    try:
        _write_screening_titles(["A", "B", "C"])
        _write_validation({
            "validated_ideas": [
                {"title": "A", "overall_verdict": "failed"},
                {"title": "B", "overall_verdict": "failed"},
                {"title": "C", "overall_verdict": "failed"},
            ],
        })
        ideas = load_validated_topics()
        assert [i["title"] for i in ideas] == ["A", "B", "C"]
    finally:
        settings.deactivate_run_scope(token)


def test_load_validated_topics_aligns_with_failed_substitutes(tmp_path, monkeypatch):
    """When substitution happened but substitutes also failed, the picker
    should list the substitutes (matching topic_screening) — not the originals."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    token = settings.activate_run_scope("r3")
    try:
        _write_screening_titles(["Sub-A", "Sub-B", "C"])
        _write_validation({
            "validated_ideas": [
                {"title": "A", "overall_verdict": "failed"},
                {"title": "Sub-A", "overall_verdict": "failed"},
                {"title": "B", "overall_verdict": "failed"},
                {"title": "Sub-B", "overall_verdict": "failed"},
                {"title": "C", "overall_verdict": "failed"},
            ],
        })
        ideas = load_validated_topics()
        assert [i["title"] for i in ideas] == ["Sub-A", "Sub-B", "C"]
    finally:
        settings.deactivate_run_scope(token)


def test_load_validated_topics_falls_back_when_screening_missing(tmp_path, monkeypatch):
    """No topic_screening.json → fall back to non-failed-only filter."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    token = settings.activate_run_scope("r4")
    try:
        _write_validation({
            "validated_ideas": [
                {"title": "A", "overall_verdict": "failed"},
                {"title": "B", "overall_verdict": "passed"},
            ],
        })
        ideas = load_validated_topics()
        assert [i["title"] for i in ideas] == ["B"]
    finally:
        settings.deactivate_run_scope(token)
