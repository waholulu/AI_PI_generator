import json
import os
from typing import Any, Dict, List

import pytest

from agents.orchestrator import ResearchState
from agents import ideation_agent


class _FakeStructuredLLM:
    """
    Fake structured LLM that handles every schema used by the pipeline:
      - LightCandidateTopicsList   (Step 1 – lightweight generation)
      - TopicScoresList            (Step 2 – combined screening + ranking)
      - RawCandidateTopic          (Step 4 – enrichment, but offline path bypasses this)
    """

    def __init__(self, schema: Any, *_: Any, **__: Any) -> None:  # pragma: no cover
        self._schema = schema

    def invoke(self, _: Any) -> Any:
        # Step 1: lightweight generation
        if self._schema is getattr(ideation_agent, "LightCandidateTopicsList", None):
            candidate = ideation_agent.LightCandidateTopic(
                title="Test Topic",
                brief_rationale="A brief rationale for testing.",
            )

            class _Resp:
                candidates: List[ideation_agent.LightCandidateTopic] = [candidate]

            return _Resp()

        # Step 2: combined screening + ranking
        if self._schema is getattr(ideation_agent, "TopicScoresList", None):
            score_obj = ideation_agent.TopicScore(
                title="Test Topic",
                score=85,
                passed_gates=True,
                rejection_reason="",
                rank=1,
            )

            class _Resp:
                scores: List[ideation_agent.TopicScore] = [score_obj]

            return _Resp()

        # Step 4: enrichment (RawCandidateTopic) — the offline branch in
        # _enrich_single() returns early via isinstance check so this path is
        # not reached in offline tests, but kept for completeness.
        if self._schema is getattr(ideation_agent, "RawCandidateTopic", None):
            return ideation_agent.RawCandidateTopic(
                title="Test Topic",
                impact_evidence="High impact justification",
                novelty_gap_type="Problem Gap",
                publishability="Journal A, Journal B",
                quantitative_specs=ideation_agent.QuantitativeSpecs(
                    unit_of_analysis="cities",
                    outcomes=["Outcome1"],
                    exposures=["Exposure1"],
                    estimand_and_strategy="ATE via DiD",
                    model_family="OLS",
                    robustness_checks=["check1", "check2", "check3", "check4", "check5", "check6"],
                    expected_tables_figures=["Table1"],
                ),
                data_sources=[
                    ideation_agent.DataSource(name="US Census", accessibility="Public API")
                ],
            )

        raise RuntimeError(f"Unsupported schema passed to _FakeStructuredLLM: {self._schema}")


class _FakeChat:
    """Adapter that mimics ChatGoogleGenerativeAI.with_structured_output."""

    def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover
        pass

    def with_structured_output(self, schema: Any, *_: Any, **__: Any) -> _FakeStructuredLLM:
        return _FakeStructuredLLM(schema)


@pytest.mark.skip(reason="IdeationAgent (V0) removed from ideation_agent module; see test_ideation_v2.py")
def test_ideation_node_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ideation node runs offline (end-to-end) by mocking LLM calls."""

    def _fake_init(self, use_strict_models: bool = False) -> None:
        self.fast_llm = _FakeChat()
        self.pro_llm = _FakeChat()

    monkeypatch.setattr(ideation_agent, "IdeationAgent", type("IdeationAgent", (), {"__init__": _fake_init}))

    state = ResearchState(
        domain_input="GeoAI and Urban Planning",
        execution_status="starting",
        field_scan_path="output/field_scan.json",
    )
    new_state = ideation_agent.ideation_node(state)

    assert new_state["execution_status"] == "harvesting"
    assert "current_plan_path" in new_state

    # research_plan.json must exist
    assert os.path.exists("config/research_plan.json")

    # topic_screening.json must exist and contain enriched top candidates
    assert os.path.exists("output/topic_screening.json")

    with open("output/topic_screening.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "candidates" in data
    assert data["candidates"], "Expected at least one enriched candidate"
    assert "run_id" in data, "topic_screening.json should carry run_id"
    assert "backup_candidates" in data, "topic_screening.json should carry backup_candidates"
    assert isinstance(data["backup_candidates"], list), "backup_candidates must be a list"

    top = data["candidates"][0]
    assert top["title"] == "Test Topic"

    # Enrichment fields must be present (populated by offline fallback in _enrich_single)
    assert "impact_evidence" in top, "Missing impact_evidence after enrichment"
    assert "quantitative_specs" in top, "Missing quantitative_specs after enrichment"
    assert top["quantitative_specs"].get("outcomes"), "quantitative_specs missing outcomes"
    assert top["quantitative_specs"].get("model_family"), "quantitative_specs missing model_family"
    assert "data_sources" in top, "Missing data_sources after enrichment"

    # Two-round scoring fields must be present
    assert "initial_score" in top, "Missing initial_score from Step 2"
    assert "final_score" in top, "Missing final_score from Step 3"
    assert "rank" in top, "Missing rank from Step 3"

    # Long-term JSONL archive should have been updated
    archive_path = "memory/enriched_top_candidates.jsonl"
    assert os.path.exists(archive_path), "Enriched JSONL archive not created"
    with open(archive_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert lines, "Archive is empty"
    record = json.loads(lines[-1])
    assert record["title"] == "Test Topic"
    assert "quantitative_specs" in record
    assert "run_id" in record
