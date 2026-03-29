import os
from typing import Any, Dict

import pytest

from agents.orchestrator import ResearchState
from agents import drafter_agent


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
    os.makedirs("config", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)

    with open("config/research_plan.json", "w", encoding="utf-8") as f:
        f.write('{"keywords": ["test"]}')

    # Minimal literature index file so loader sees something
    os.makedirs("data/literature", exist_ok=True)
    with open("data/literature/index.json", "w", encoding="utf-8") as f:
        f.write("[]")

    with open("prompts/academic_drafter.txt", "w", encoding="utf-8") as f:
        f.write("Test prompt")

    # Patch DrafterAgent to use fake LLM instead of real Gemini
    def _fake_init(self: drafter_agent.DrafterAgent) -> None:
        self.output_md = "output/Draft_v1.md"
        self.llm = _FakeLLM()

    monkeypatch.setattr(drafter_agent.DrafterAgent, "__init__", _fake_init)  # type: ignore[arg-type]

    state = ResearchState(
        current_plan_path="config/research_plan.json",
        literature_inventory_path="data/literature/index.json",
        execution_status="drafting",
    )
    new_state = drafter_agent.drafter_node(state)

    assert new_state["execution_status"] == "fetching"
    assert os.path.exists("output/Draft_v1.md")
    with open("output/Draft_v1.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test Draft" in content


def test_drafter_node_fallback_on_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure drafter writes fallback content when LLM raises an exception."""
    os.makedirs("config", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)

    with open("config/research_plan.json", "w", encoding="utf-8") as f:
        f.write('{"keywords": ["test"]}')

    os.makedirs("data/literature", exist_ok=True)
    with open("data/literature/index.json", "w", encoding="utf-8") as f:
        f.write("[]")

    with open("prompts/academic_drafter.txt", "w", encoding="utf-8") as f:
        f.write("Test prompt")

    class _BrokenLLM:
        def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover
            pass

        def invoke(self, _: Dict[str, Any]) -> Any:
            raise RuntimeError("Simulated LLM failure")

    def _failing_init(self: drafter_agent.DrafterAgent) -> None:
        self.output_md = "output/Draft_v1.md"
        self.llm = _BrokenLLM()

    monkeypatch.setattr(drafter_agent.DrafterAgent, "__init__", _failing_init)  # type: ignore[arg-type]

    state = ResearchState(
        current_plan_path="config/research_plan.json",
        literature_inventory_path="data/literature/index.json",
        execution_status="drafting",
    )
    new_state = drafter_agent.drafter_node(state)

    # Fallback must still transition state and write a file
    assert new_state["execution_status"] == "fetching"
    assert os.path.exists("output/Draft_v1.md")
    with open("output/Draft_v1.md", "r", encoding="utf-8") as f:
        content = f.read()
    # The known fallback sentinel from DrafterAgent.run
    assert "Fallback Draft" in content or "Failed to reach API" in content

