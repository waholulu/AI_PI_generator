"""
Live Gemini integration test for IdeationAgent / ideation_node.

Requires:
- GEMINI_API_KEY or GOOGLE_API_KEY set in .env
- GEMINI_FAST_MODEL and GEMINI_PRO_MODEL set (or use default model names)

Run only with:
    pytest -m live_llm tests/test_ideation_live.py
"""
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from agents.ideation_agent import ideation_node
from agents.orchestrator import ResearchState


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
def test_ideation_live_produces_research_plan() -> None:
    """
    Live test: ideation_node calls the real Gemini model and produces a
    non-empty research plan.  Fails if the API key is missing or the
    LLM call errors out.
    """
    _require_gemini_env()

    os.makedirs("config", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    state = ResearchState(
        domain_input="GeoAI and Urban Planning",
        execution_status="starting",
        field_scan_path="output/field_scan.json",
    )

    new_state = ideation_node(state)

    # State transition
    assert new_state["execution_status"] == "harvesting", (
        f"Expected 'harvesting', got '{new_state.get('execution_status')}'"
    )
    assert "current_plan_path" in new_state

    # Research plan must exist and contain required fields
    plan_path = new_state["current_plan_path"]
    assert os.path.exists(plan_path), f"research_plan.json not found at {plan_path}"

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    assert "project_title" in plan, "research_plan.json missing 'project_title'"
    assert "keywords" in plan, "research_plan.json missing 'keywords'"

    # Topic screening must exist and have at least one candidate
    screening_path = "output/topic_screening.json"
    assert os.path.exists(screening_path), "topic_screening.json not generated"

    with open(screening_path, "r", encoding="utf-8") as f:
        screening = json.load(f)

    candidates = screening.get("candidates", [])
    assert candidates, "No candidates produced by ideation – LLM may have returned empty output"

    top = candidates[0]
    # Core identity fields
    assert top.get("title"), "Top candidate has no title"

    # Two-round scoring fields (Steps 2 & 3)
    assert "initial_score" in top, "Top candidate missing initial_score from Step 2"
    assert "final_score" in top, "Top candidate missing final_score from Step 3"
    assert "rank" in top, "Top candidate missing rank from Step 3"

    # Enrichment fields (Step 4)
    assert "impact_evidence" in top, "Top candidate missing impact_evidence after enrichment"
    assert "quantitative_specs" in top, "Top candidate missing quantitative_specs after enrichment"
    assert top["quantitative_specs"].get("outcomes"), "quantitative_specs missing outcomes"
    assert top["quantitative_specs"].get("model_family"), "quantitative_specs missing model_family"
    assert "data_sources" in top, "Top candidate missing data_sources after enrichment"

    # Long-term JSONL archive should have been updated
    archive_path = "memory/enriched_top_candidates.jsonl"
    assert os.path.exists(archive_path), "Enriched JSONL archive not created"
    with open(archive_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert lines, "Enriched JSONL archive is empty"
    record = json.loads(lines[-1])
    assert record.get("title"), "Last archive record has no title"
    assert "quantitative_specs" in record, "Last archive record missing quantitative_specs"
