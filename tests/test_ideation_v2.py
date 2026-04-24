"""Tests for agents/ideation_agent_v2.py and ideation_agent.py router — Day 5 TDD.

All tests mock LLM and file I/O; no live API calls.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents.budget_tracker import BudgetTracker
from agents.ideation_agent_v2 import (
    IdeationAgentV2,
    IdeationSeedGenerationError,
    _build_legacy_gates_map,
    _is_placeholder_candidate,
)
from agents.reflection_loop import ReflectionTrace, RoundRecord
from agents.rule_engine import GateResult
from models.topic_schema import (
    Contribution,
    ContributionPrimary,
    ExposureFamily,
    ExposureX,
    FinalStatus,
    Frequency,
    IdentificationPrimary,
    IdentificationStrategy,
    OutcomeFamily,
    OutcomeY,
    SamplingMode,
    SeedCandidate,
    SpatialScope,
    TemporalScope,
    Topic,
    TopicMeta,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_topic(topic_id: str = "t001") -> Topic:
    return Topic(
        meta=TopicMeta(topic_id=topic_id),
        exposure_X=ExposureX(
            family=ExposureFamily.AIR_QUALITY,
            specific_variable="PM2.5",
            spatial_unit="tract",
        ),
        outcome_Y=OutcomeY(
            family=OutcomeFamily.HEALTH,
            specific_variable="mortality",
            spatial_unit="tract",
        ),
        spatial_scope=SpatialScope(
            geography="US cities",
            spatial_unit="tract",
            sampling_mode=SamplingMode.PANEL,
        ),
        temporal_scope=TemporalScope(
            start_year=2010,
            end_year=2020,
            frequency=Frequency.ANNUAL,
        ),
        identification=IdentificationStrategy(
            primary=IdentificationPrimary.FE,
            key_threats=["confounding"],
            mitigations={"confounding": "confounding_control"},
        ),
        contribution=Contribution(
            primary=ContributionPrimary.CAUSAL_REFINEMENT,
            statement="Causal evidence on PM2.5 and mortality.",
            gap_addressed="Cross-sectional gap.",
        ),
        free_form_title="PM2.5 and mortality in US cities",
    )


def make_minimal_trace(
    topic_id: str,
    status: FinalStatus = FinalStatus.ACCEPTED,
    gate_results: list[GateResult] | None = None,
) -> ReflectionTrace:
    gr = gate_results or [
        GateResult("G1", "mechanism_plausibility", True, True, "ok", score=4, max_score=5),
        GateResult("G2", "scale_alignment", True, False, "ok"),
        GateResult("G3", "data_availability", True, False, "ok"),
        GateResult("G6", "automation_feasibility", True, False, "ok"),
    ]
    rnd = RoundRecord(
        round_num=1,
        pre_refine_topic_snapshot={},
        gate_results=gr,
        openalex_queries_log=[],
        llm_critique_raw={},
        llm_calls=[],
        decision=status.value,
        applied_operations=[],
        slot_diff={},
        four_tuple_sig="abc123",
        round_score=4.0,
        budget_snapshot={},
        wallclock_seconds=0.0,
    )
    return ReflectionTrace(
        topic_id=topic_id,
        seed_version={},
        final_status=status,
        rounds=[rnd],
        reject_reasons=[],
        convergence={},
        design_alternatives_considered=[],
        total_cost_usd=0.01,
        total_wallclock_seconds=0.5,
    )


# ── Test 1: _build_legacy_gates_map ──────────────────────────────────────────

def test_build_legacy_gates_map_has_six_fields():
    trace = make_minimal_trace("t001")
    result = _build_legacy_gates_map(trace)
    for key in ["impact", "quantitative", "novelty", "publishability", "automation", "data_availability"]:
        assert key in result


def test_build_legacy_gates_map_reflects_gate_pass():
    gr = [
        GateResult("G3", "data_availability", False, False, "year_gap", score=None),
    ]
    trace = make_minimal_trace("t001", gate_results=gr)
    result = _build_legacy_gates_map(trace)
    assert result["data_availability"] is False


def test_build_legacy_gates_map_includes_full_seven_gates():
    trace = make_minimal_trace("t001")
    result = _build_legacy_gates_map(trace)
    assert "full_seven_gates" in result


# ── Test 2: Level 1 — YAML topic loading and validation ──────────────────────

def test_level1_raises_user_input_error_on_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    from agents.ideation_agent_v2 import _UserInputError
    with pytest.raises(_UserInputError, match="not found"):
        agent.run_level1({"user_topic_path": "/nonexistent/topic.yaml"})


def test_level1_raises_user_input_error_when_path_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    from agents.ideation_agent_v2 import _UserInputError
    with pytest.raises(_UserInputError):
        agent.run_level1({})


def test_level1_raises_hitl_on_hard_blocker_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    topic = make_topic()
    # Inject mismatched spatial units to trigger G2 failure
    topic_dict = topic.model_dump(mode="json")
    topic_dict["exposure_X"]["spatial_unit"] = "point"
    topic_dict["outcome_Y"]["spatial_unit"] = "country"

    yaml_path = tmp_path / "bad_topic.yaml"
    yaml_path.write_text(yaml.dump(topic_dict))

    from models.topic_schema import HITLInterruption
    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    with pytest.raises(HITLInterruption) as exc_info:
        agent.run_level1({"user_topic_path": str(yaml_path)})
    assert exc_info.value.kind == "hard_blocker_failed"


def test_level1_accepted_writes_output_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    topic = make_topic()
    topic_dict = topic.model_dump(mode="json")
    topic_dict["declared_sources"] = ["NHGIS"]

    yaml_path = tmp_path / "good_topic.yaml"
    yaml_path.write_text(yaml.dump(topic_dict))

    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))

    # Mock reflection loop to return ACCEPTED without real LLM
    accepted_trace = make_minimal_trace("t001", FinalStatus.ACCEPTED)
    with patch("agents.ideation_agent_v2.run_reflection_loop", return_value=accepted_trace):
        result = agent.run_level1({"user_topic_path": str(yaml_path)})

    assert "candidate_topics_path" in result
    assert "current_plan_path" in result
    screening_file = Path(result["candidate_topics_path"])
    assert screening_file.exists()

    screening = json.loads(screening_file.read_text())
    assert screening["input_mode"] == "level_1"
    assert len(screening["candidates"]) == 1
    assert "legacy_six_gates" in screening["candidates"][0]


# ── Test 3: Level 1 — TENTATIVE → HITLInterruption ───────────────────────────

def test_level1_tentative_raises_hitl(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    topic = make_topic()
    topic_dict = topic.model_dump(mode="json")
    topic_dict["declared_sources"] = ["NHGIS"]

    yaml_path = tmp_path / "topic.yaml"
    yaml_path.write_text(yaml.dump(topic_dict))

    tentative_trace = make_minimal_trace("t001", FinalStatus.TENTATIVE)
    from models.topic_schema import HITLInterruption
    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))

    with patch("agents.ideation_agent_v2.run_reflection_loop", return_value=tentative_trace):
        with pytest.raises(HITLInterruption) as exc_info:
            agent.run_level1({"user_topic_path": str(yaml_path)})
    assert exc_info.value.kind == "refinable_still_failing_after_one_round"


# ── Test 4: Level 2 — LLM unavailable raises a clear error ──────────────────

def test_level2_raises_when_llm_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = None  # simulate missing/misconfigured API key

    with pytest.raises(IdeationSeedGenerationError, match="LLM"):
        agent._generate_seeds("Urban Planning", "", "")


def test_level2_raises_when_llm_returns_no_parseable_seeds(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = MagicMock()  # will be stubbed via method patch below

    with patch.object(agent, "_llm_generate_seeds", return_value=[]):
        with pytest.raises(IdeationSeedGenerationError, match="no parseable"):
            agent._generate_seeds("Urban Planning", "", "")


def test_placeholder_candidate_detected():
    assert _is_placeholder_candidate({"title": "Fallback topic 1"})
    assert _is_placeholder_candidate({"topic_id": "fallback_001"})
    assert not _is_placeholder_candidate({"title": "PM2.5 and mortality"})


# ── Test 5: Level 2 — accepted path writes all outputs ───────────────────────

def test_level2_writes_screening_and_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    topic = make_topic("seed_000")
    seed = SeedCandidate(topic=topic, declared_sources=["NHGIS"])

    accepted_trace = make_minimal_trace("seed_000", FinalStatus.ACCEPTED)

    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = None  # no LLM

    # Override _generate_seeds and reflection
    agent._generate_seeds = lambda *a, **kw: [seed]
    with patch("agents.ideation_agent_v2.run_reflection_loop", return_value=accepted_trace):
        result = agent.run_level2({"domain_input": "Urban Planning"})

    assert result["execution_status"] == "harvesting"
    screening = json.loads(Path(result["candidate_topics_path"]).read_text())
    assert screening["input_mode"] == "level_2"
    assert len(screening["candidates"]) >= 1

    summary_path = tmp_path / "output" / "ideation_run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "run_id" in summary
    assert "status_breakdown" in summary


# ── Test 6: Level 2 — tentative pool written ─────────────────────────────────

def test_level2_writes_tentative_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    topic = make_topic("seed_001")
    seed = SeedCandidate(topic=topic, declared_sources=["NHGIS"])
    tentative_trace = make_minimal_trace("seed_001", FinalStatus.TENTATIVE)

    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = None
    agent._generate_seeds = lambda *a, **kw: [seed]

    with patch("agents.ideation_agent_v2.run_reflection_loop", return_value=tentative_trace):
        agent.run_level2({"domain_input": "Urban Planning"})

    pool_path = tmp_path / "output" / "tentative_pool.json"
    assert pool_path.exists()
    pool_data = json.loads(pool_path.read_text())
    assert len(pool_data["tentative"]) == 1


# ── Test 7: Level 2 — no ACCEPTED reruns once then lists near-pass topics ──

def test_level2_rerun_then_lists_near_pass_topics(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    topic = make_topic("seed_002")
    topic.spatial_scope.geography = "United States"
    seed = SeedCandidate(topic=topic, declared_sources=["NHGIS"])
    tentative_trace = make_minimal_trace("seed_002", FinalStatus.TENTATIVE)

    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = None
    agent._generate_seeds = lambda *a, **kw: [seed]

    # First attempt has no ACCEPTED -> triggers auto-rerun.
    with patch.object(
        agent,
        "_run_reflection_batch",
        side_effect=[
            ([], [(seed, tentative_trace)], []),
            ([], [(seed, tentative_trace)], []),
        ],
    ) as mocked_batch:
        result = agent.run_level2({"domain_input": "Urban Planning"})

    screening = json.loads(Path(result["candidate_topics_path"]).read_text())
    assert len(screening["candidates"]) == 1
    assert mocked_batch.call_count == 2
    assert screening["candidates"][0]["final_status"] == "TENTATIVE"
    assert screening["candidates"][0]["title"].startswith("[US] ")
    assert screening["candidates"][0]["passed_gates_count"] >= 1


# ── Test 7: ideation_node router ─────────────────────────────────────────────

def test_ideation_node_routes_to_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("LEGACY_IDEATION", "1")

    with patch("agents._legacy.ideation_agent_v0.IdeationAgentV0") as MockV0:
        MockV0.return_value.run.return_value = {"execution_status": "done", "degraded_nodes": []}
        from importlib import reload
        import agents.ideation_agent as ia_mod
        reload(ia_mod)
        result = ia_mod.ideation_node({"domain_input": "test", "degraded_nodes": []})
    assert result["execution_status"] == "done"


def test_ideation_node_routes_to_v2_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("LEGACY_IDEATION", raising=False)

    with patch("agents.ideation_agent_v2.IdeationAgentV2") as MockV2:
        MockV2.return_value.run.return_value = {"execution_status": "harvesting", "degraded_nodes": []}
        from importlib import reload
        import agents.ideation_agent as ia_mod
        reload(ia_mod)
        result = ia_mod.ideation_node({"domain_input": "test", "degraded_nodes": []})
    assert result["execution_status"] == "harvesting"


# ── Test 8: schema re-exports are available from ideation_agent ───────────────

def test_schema_reexports_from_ideation_agent():
    from agents.ideation_agent import (
        LightCandidateTopic,
        TopicScore,
        RawCandidateTopic,
        ResearchPlanSchema,
        DataSource,
    )
    assert LightCandidateTopic is not None
    assert TopicScore is not None
    assert RawCandidateTopic is not None
    assert ResearchPlanSchema is not None
    assert DataSource is not None
