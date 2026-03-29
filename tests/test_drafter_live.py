"""
Live Gemini integration test for DrafterAgent / drafter_node.

Requires:
- GEMINI_API_KEY or GOOGLE_API_KEY set in .env

Run only with:
    pytest -m live_llm tests/test_drafter_live.py
"""
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from agents.drafter_agent import drafter_node
from agents.orchestrator import ResearchState

# Known sentinel string that DrafterAgent writes when LLM is unavailable.
_FALLBACK_SENTINEL = "Failed to reach API"


def _require_gemini_env() -> None:
    """Fail clearly if the Gemini API key is not available."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise AssertionError(
            "GEMINI_API_KEY / GOOGLE_API_KEY must be set in .env for live LLM tests."
        )


@pytest.mark.live_llm
def test_drafter_live_produces_non_fallback_draft() -> None:
    """
    Live test: drafter_node calls the real Gemini model and produces a
    draft that is NOT the static fallback content.  Fails if the API key
    is missing or the LLM is unreachable.
    """
    _require_gemini_env()

    os.makedirs("config", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)
    os.makedirs("data/literature", exist_ok=True)

    # Minimal research plan
    plan = {
        "project_title": "GeoAI and Urban Planning Live Test",
        "keywords": ["GeoAI", "urban planning"],
        "data_sources": [],
        "methodology": {},
        "outcomes": [],
        "exposures": [],
        "research_questions": ["How does GeoAI affect urban planning?"],
        "hypotheses": [],
        "unit_of_analysis": "city",
        "study_type": "observational",
        "topic_screening": {},
    }
    with open("config/research_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f)

    # Minimal literature index (empty is fine)
    with open("data/literature/index.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    # Minimal drafter prompt
    with open("prompts/academic_drafter.txt", "w", encoding="utf-8") as f:
        f.write(
            "You are an academic writing assistant.  "
            "Draft a concise research proposal based on the plan and evidence provided."
        )

    state = ResearchState(
        current_plan_path="config/research_plan.json",
        literature_inventory_path="data/literature/index.json",
        execution_status="drafting",
    )

    new_state = drafter_node(state)

    # State transition
    assert new_state["execution_status"] == "fetching", (
        f"Expected 'fetching', got '{new_state.get('execution_status')}'"
    )

    draft_path = "output/Draft_v1.md"
    assert os.path.exists(draft_path), "Draft_v1.md was not created"

    with open(draft_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert content.strip(), "Draft_v1.md is empty"
    assert _FALLBACK_SENTINEL not in content, (
        "Draft contains fallback sentinel – the live LLM call did not succeed"
    )
