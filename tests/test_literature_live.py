import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from agents.literature_agent import literature_node
from agents.orchestrator import ResearchState


@pytest.mark.live_openalex
def test_literature_live_uses_openalex(tmp_path) -> None:
    """
    Live integration test for LiteratureHarvester.search_openalex and downstream indexing.

    Requires:
    - OPENALEX_API_KEY / OPENALEX_EMAIL set in .env
    - pyalex installed
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("OPENALEX_API_KEY")
    email = os.getenv("OPENALEX_EMAIL")
    if not api_key or not email:
        raise AssertionError("OPENALEX_API_KEY / OPENALEX_EMAIL must be configured for live OpenAlex tests.")

    try:
        import pyalex  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - missing dependency
        raise AssertionError("pyalex must be installed for live literature OpenAlex tests.") from exc

    # Prepare minimal research plan
    os.makedirs("config", exist_ok=True)
    with open("config/research_plan.json", "w", encoding="utf-8") as f:
        f.write('{"keywords": ["GeoAI", "urban planning"]}')

    state = ResearchState(
        current_plan_path="config/research_plan.json",
        execution_status="harvesting",
    )

    new_state = literature_node(state)

    assert new_state["execution_status"] == "drafting"
    index_path = "data/literature/index.json"
    bib_path = "output/references.bib"

    assert os.path.exists(index_path)
    assert os.path.exists(bib_path)

    with open(index_path, "r", encoding="utf-8") as f:
        inventory = json.load(f)

    assert isinstance(inventory, list) and inventory, "Expected non-empty literature inventory from live OpenAlex search"

