from __future__ import annotations

from pathlib import Path


def test_removed_runtime_dependencies_are_not_declared() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    forbidden = ["fastapi", "uvicorn", "langchain", "langgraph", "langgraph-checkpoint"]
    for item in forbidden:
        assert item not in text


def test_skill_entrypoint_exists() -> None:
    assert Path(".agents/skills/run-auto-pi-research/SKILL.md").exists()
    assert Path(".agents/skills/run-auto-pi-research/scripts/autopi.py").exists()
    assert Path(".agents/skills/run-auto-pi-research/references/workflow-contract.md").exists()
