"""End-to-end integration tests for Module 1 V2 — Day 6.

All external dependencies (LLM, OpenAlex) are mocked.
Tests verify that:
  1. Level 2 path produces all required output files
  2. topic_screening.json contains legacy_six_gates field
  3. tentative_pool.json is written
  4. LEGACY_IDEATION=1 env var routes to V0 correctly
  5. ideation_run_summary.json is written with correct schema
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.budget_tracker import BudgetTracker
from agents.ideation_agent_v2 import IdeationAgentV2
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

def make_accepted_trace(topic_id: str) -> ReflectionTrace:
    gr = [
        GateResult("G2", "scale_alignment", True, False, "ok"),
        GateResult("G3", "data_availability", True, False, "ok"),
        GateResult("G6", "automation_feasibility", True, False, "ok"),
        GateResult("G1", "mechanism_plausibility", True, True, "ok", score=4, max_score=5),
        GateResult("G4", "identification_validity", True, True, "ok", score=4, max_score=5),
        GateResult("G5", "novelty", True, True, "ok", score=4, max_score=5),
        GateResult("G7", "contribution_clarity", True, True, "ok", score=4, max_score=5),
    ]
    rnd = RoundRecord(
        round_num=1,
        pre_refine_topic_snapshot={},
        gate_results=gr,
        openalex_queries_log=[],
        llm_critique_raw={},
        llm_calls=[],
        decision="ACCEPTED",
        applied_operations=[],
        slot_diff={},
        four_tuple_sig="sig1",
        round_score=4.0,
        budget_snapshot={},
        wallclock_seconds=0.0,
        g5_skipped=False,
    )
    return ReflectionTrace(
        topic_id=topic_id,
        seed_version={},
        final_status=FinalStatus.ACCEPTED,
        rounds=[rnd],
        reject_reasons=[],
        convergence={},
        design_alternatives_considered=[],
        total_cost_usd=0.01,
        total_wallclock_seconds=0.1,
    )


def make_tentative_trace(topic_id: str) -> ReflectionTrace:
    gr = [
        GateResult("G2", "scale_alignment", True, False, "ok"),
        GateResult("G3", "data_availability", True, False, "ok"),
        GateResult("G6", "automation_feasibility", True, False, "ok"),
        GateResult("G1", "mechanism_plausibility", False, True, "score=2<4", score=2, max_score=5),
        GateResult("G4", "identification_validity", False, True, "score=2<4", score=2, max_score=5),
        GateResult("G5", "novelty", False, True, "score=1<3", score=1, max_score=5),
        GateResult("G7", "contribution_clarity", False, True, "score=2<4", score=2, max_score=5),
    ]
    rnd = RoundRecord(
        round_num=1,
        pre_refine_topic_snapshot={},
        gate_results=gr,
        openalex_queries_log=[],
        llm_critique_raw={},
        llm_calls=[],
        decision="TENTATIVE",
        applied_operations=[],
        slot_diff={},
        four_tuple_sig="sig2",
        round_score=1.75,
        budget_snapshot={},
        wallclock_seconds=0.0,
        g5_skipped=False,
    )
    return ReflectionTrace(
        topic_id=topic_id,
        seed_version={},
        final_status=FinalStatus.TENTATIVE,
        rounds=[rnd],
        reject_reasons=[],
        convergence={},
        design_alternatives_considered=[],
        total_cost_usd=0.01,
        total_wallclock_seconds=0.1,
    )


def make_seed(topic_id: str = "s001") -> SeedCandidate:
    t = Topic(
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
            start_year=2010, end_year=2020, frequency=Frequency.ANNUAL
        ),
        identification=IdentificationStrategy(
            primary=IdentificationPrimary.FE,
            key_threats=["confounding"],
            mitigations={"confounding": "fixed effects controls"},
        ),
        contribution=Contribution(
            primary=ContributionPrimary.CAUSAL_REFINEMENT,
            statement="Causal evidence.",
            gap_addressed="Prior studies cross-sectional.",
        ),
        free_form_title=f"Integration test topic {topic_id}",
    )
    return SeedCandidate(topic=t, declared_sources=["NHGIS"])


# ── Integration Test 1: Level 2 full offline run ─────────────────────────────

def test_level2_produces_all_required_output_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    seed_accepted = make_seed("s_acc")
    seed_tentative = make_seed("s_tent")

    traces = {
        "s_acc": make_accepted_trace("s_acc"),
        "s_tent": make_tentative_trace("s_tent"),
    }

    def mock_reflection(seed, *a, **kw):
        return traces[seed.topic.meta.topic_id]

    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = None
    agent._generate_seeds = lambda *a, **kw: [seed_accepted, seed_tentative]

    with patch("agents.ideation_agent_v2.run_reflection_loop", side_effect=mock_reflection):
        result = agent.run_level2({"domain_input": "Urban Air Quality"})

    # All required files exist
    assert Path(result["candidate_topics_path"]).exists()
    assert Path(result["current_plan_path"]).exists()
    assert (tmp_path / "output" / "ideation_run_summary.json").exists()
    assert (tmp_path / "output" / "tentative_pool.json").exists()


# ── Integration Test 2: legacy_six_gates field in all candidates ──────────────

def test_screening_candidates_have_legacy_six_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    seed = make_seed("s001")
    trace = make_accepted_trace("s001")

    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = None
    agent._generate_seeds = lambda *a, **kw: [seed]

    with patch("agents.ideation_agent_v2.run_reflection_loop", return_value=trace):
        result = agent.run_level2({"domain_input": "Urban Planning"})

    screening = json.loads(Path(result["candidate_topics_path"]).read_text())
    for candidate in screening["candidates"]:
        assert "legacy_six_gates" in candidate, f"Missing legacy_six_gates in {candidate}"


# ── Integration Test 3: ideation_run_summary schema ──────────────────────────

def test_ideation_run_summary_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    seed = make_seed("s002")
    trace = make_accepted_trace("s002")

    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = None
    agent._generate_seeds = lambda *a, **kw: [seed]

    with patch("agents.ideation_agent_v2.run_reflection_loop", return_value=trace):
        agent.run_level2({"domain_input": "Health Geography"})

    summary_path = tmp_path / "output" / "ideation_run_summary.json"
    summary = json.loads(summary_path.read_text())

    for key in ["run_id", "started_at", "ended_at", "input_mode",
                "total_topics_attempted", "status_breakdown",
                "total_cost_usd", "total_wallclock_seconds"]:
        assert key in summary, f"Missing key '{key}' in ideation_run_summary.json"

    assert summary["input_mode"] == "level_2"
    assert summary["status_breakdown"]["ACCEPTED"] >= 1


# ── Integration Test 4: LEGACY_IDEATION=1 env routes to V0 ───────────────────

def test_legacy_ideation_env_routes_to_v0(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("LEGACY_IDEATION", "1")

    with patch("agents._legacy.ideation_agent_v0.IdeationAgentV0") as MockV0:
        MockV0.return_value.run.return_value = {
            "execution_status": "harvesting",
            "degraded_nodes": [],
        }
        from importlib import reload
        import agents.ideation_agent as ia
        reload(ia)
        result = ia.ideation_node({
            "domain_input": "test domain",
            "degraded_nodes": [],
        })

    assert result["execution_status"] == "harvesting"
    MockV0.return_value.run.assert_called_once()


# ── Integration Test 5: tentative pool contains topics with failed gates ──────

def test_tentative_pool_contains_failed_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    seed = make_seed("tent_001")
    trace = make_tentative_trace("tent_001")

    agent = IdeationAgentV2(budget=BudgetTracker(per_run_budget_usd=10.0))
    agent._llm = None
    agent._generate_seeds = lambda *a, **kw: [seed]

    with patch("agents.ideation_agent_v2.run_reflection_loop", return_value=trace):
        agent.run_level2({"domain_input": "Urban Planning"})

    pool_path = tmp_path / "output" / "tentative_pool.json"
    pool_data = json.loads(pool_path.read_text())
    assert len(pool_data["tentative"]) == 1
    assert "failed_gates" in pool_data["tentative"][0]
    # All four refinable gates failed → all should appear
    failed = pool_data["tentative"][0]["failed_gates"]
    assert len(failed) >= 3
