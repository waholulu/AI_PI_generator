import json
import os
from typing import Any, Dict

import pytest

from agents import drafter_agent, settings
from agents.orchestrator import ResearchState


class _FakeLLM:
    """Minimal fake LLM to avoid real Gemini calls."""

    def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        pass

    def invoke(self, _: Dict[str, Any]) -> Any:
        class _Resp:
            content = "# Test Draft\n\nThis is a fake draft from the test LLM."

        return _Resp()


def test_drafter_node_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure drafter node runs using a mocked LLM and writes a draft."""
    plan = {
        "run_id": "r1",
        "project_title": "Test topic",
        "research_question": "How does X affect Y?",
        "short_rationale": "A concise rationale about policy relevance.",
        "geography": "US",
        "time_window": "2010-2020",
        "exposure": {"name": "X", "measurement_proxy": "x"},
        "outcome": {"name": "Y", "measurement_proxy": "y"},
        "identification": {"primary_method": "fixed_effects", "key_threats": ["confounding"]},
        "data_sources": [{"name": "Source A", "access_url": "https://example.org/a.csv", "expected_format": "csv"}],
        "literature_queries": ["x y", "x data", "y data"],
        "feasibility": {"overall_verdict": "warning"},
    }
    plan_path = settings.research_plan_path()
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    index_path = settings.literature_index_path()
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    # Patch DrafterAgent to use fake LLM instead of real Gemini
    monkeypatch.setattr(drafter_agent.DrafterAgent, "_init_llm", lambda self: _FakeLLM())  # type: ignore[arg-type]

    state = ResearchState(
        current_plan_path=plan_path,
        literature_inventory_path=index_path,
        execution_status="drafting",
    )
    new_state = drafter_agent.drafter_node(state)

    assert new_state["execution_status"] == "fetching"
    assert os.path.exists(settings.draft_path())
    with open(settings.draft_path(), "r", encoding="utf-8") as f:
        content = f.read()
    assert "# Research Memo" in content


def test_drafter_node_fallback_on_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure drafter writes fallback content when LLM raises an exception."""
    plan = {
        "run_id": "r1",
        "project_title": "Fallback topic",
        "research_question": "How does X affect Y?",
        "short_rationale": "A concise rationale about policy relevance.",
        "geography": "US",
        "time_window": "2010-2020",
        "exposure": {"name": "X", "measurement_proxy": "x"},
        "outcome": {"name": "Y", "measurement_proxy": "y"},
        "identification": {"primary_method": "fixed_effects", "key_threats": ["confounding"]},
        "data_sources": [{"name": "Source A", "access_url": "https://example.org/a.csv", "expected_format": "csv"}],
        "literature_queries": ["x y", "x data", "y data"],
        "feasibility": {"overall_verdict": "warning"},
    }
    plan_path = settings.research_plan_path()
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    index_path = settings.literature_index_path()
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    class _BrokenLLM:
        def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover
            pass

        def invoke(self, _: Dict[str, Any]) -> Any:
            raise RuntimeError("Simulated LLM failure")

    monkeypatch.setattr(drafter_agent.DrafterAgent, "_init_llm", lambda self: _BrokenLLM())  # type: ignore[arg-type]

    state = ResearchState(
        current_plan_path=plan_path,
        literature_inventory_path=index_path,
        execution_status="drafting",
    )
    new_state = drafter_agent.drafter_node(state)

    # Fallback must still transition state and write a file
    assert new_state["execution_status"] == "fetching"
    assert os.path.exists(settings.draft_path())
    with open(settings.draft_path(), "r", encoding="utf-8") as f:
        content = f.read()
    assert "literature evidence is limited" in content


def test_drafter_fallback_has_eight_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = {
        "run_id": "r2",
        "project_title": "Section test",
        "research_question": "How does X affect Y?",
        "short_rationale": "A concise rationale about policy relevance.",
        "geography": "US",
        "time_window": "2010-2020",
        "exposure": {"name": "X", "measurement_proxy": "x"},
        "outcome": {"name": "Y", "measurement_proxy": "y"},
        "identification": {"primary_method": "fixed_effects", "key_threats": ["confounding"]},
        "data_sources": [{"name": "Source A", "access_url": "https://example.org/a.csv", "expected_format": "csv"}],
        "literature_queries": ["x y", "x data", "y data"],
        "feasibility": {"overall_verdict": "warning"},
    }
    plan_path = settings.research_plan_path()
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f)

    class _BrokenLLM:
        def invoke(self, _: Dict[str, Any]) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(drafter_agent.DrafterAgent, "_init_llm", lambda self: _BrokenLLM())  # type: ignore[arg-type]
    out = drafter_agent.drafter_node(ResearchState(current_plan_path=plan_path, execution_status="drafting"))
    assert out["execution_status"] == "fetching"

    with open(settings.draft_path(), "r", encoding="utf-8") as f:
        memo = f.read()
    for heading in [
        "## 1. Proposed Title",
        "## 2. Research Question",
        "## 3. Why This Matters",
        "## 4. Data and Measurement",
        "## 5. Empirical Strategy",
        "## 6. Related Literature",
        "## 7. Main Risks",
        "## 8. Recommended Next Steps",
    ]:
        assert heading in memo

