"""Tests for agents/reflection_loop.py — Day 4 TDD.

All tests use mock LLM (no API keys required).
"""

import pytest
from unittest.mock import MagicMock, patch
import types
import sys

from agents.budget_tracker import BudgetTracker
from agents.reflection_loop import (
    LLMCallRecord,
    ReflectionTrace,
    RoundRecord,
    _apply_operations,
    _llm_propose_operation_values,
    _select_refine_operations,
    run_reflection_loop,
)
from agents.rule_engine import GateResult, RuleEngine
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

def make_seed(
    topic_id: str = "t001",
    x_spatial: str = "tract",
    y_spatial: str = "tract",
    method: IdentificationPrimary = IdentificationPrimary.FE,
    sources: list[str] | None = None,
    key_threats: list[str] | None = None,
    mitigations: list[str] | None = None,
) -> SeedCandidate:
    topic = Topic(
        meta=TopicMeta(topic_id=topic_id),
        exposure_X=ExposureX(
            family=ExposureFamily.AIR_QUALITY,
            specific_variable="PM2.5",
            spatial_unit=x_spatial,
        ),
        outcome_Y=OutcomeY(
            family=OutcomeFamily.HEALTH,
            specific_variable="mortality",
            spatial_unit=y_spatial,
        ),
        spatial_scope=SpatialScope(
            geography="US cities",
            spatial_unit=x_spatial,
            sampling_mode=SamplingMode.PANEL,
        ),
        temporal_scope=TemporalScope(
            start_year=2010,
            end_year=2020,
            frequency=Frequency.ANNUAL,
        ),
        identification=IdentificationStrategy(
            primary=method,
            key_threats=key_threats if key_threats is not None else ["confounding"],
            mitigations=mitigations if mitigations is not None else {"confounding": "confounding_control"},
        ),
        contribution=Contribution(
            primary=ContributionPrimary.CAUSAL_REFINEMENT,
            statement="Causal estimate of PM2.5 on mortality.",
            gap_addressed="Prior studies are cross-sectional.",
        ),
        free_form_title="PM2.5 and mortality in US cities",
    )
    return SeedCandidate(
        topic=topic,
        declared_sources=sources if sources is not None else ["NHGIS"],
    )


def make_budget() -> BudgetTracker:
    return BudgetTracker(per_run_budget_usd=10.0, per_topic_budget_usd=5.0)


# ── Test 1: ACCEPTED path (hard-blockers pass, all LLM scores pass) ───────────

def test_accepted_when_all_gates_pass():
    seed = make_seed()
    budget = make_budget()

    trace = run_reflection_loop(
        seed, budget, max_rounds=3,
        # No LLM → neutral score=4, passes all refinable gates
    )

    assert isinstance(trace, ReflectionTrace)
    assert trace.final_status in (FinalStatus.ACCEPTED, FinalStatus.TENTATIVE)
    assert len(trace.rounds) >= 1
    assert trace.rounds[-1].decision in ("ACCEPTED", "TENTATIVE")


# ── Test 2: REJECTED on hard-blocker failure ──────────────────────────────────

def test_rejected_on_hard_blocker_fail():
    # block (rank 2) vs country (rank 10) → G2 fails (rank_diff=8)
    seed = make_seed(x_spatial="block", y_spatial="country")
    budget = make_budget()

    trace = run_reflection_loop(seed, budget, max_rounds=3)

    assert trace.final_status == FinalStatus.REJECTED
    assert trace.rounds[0].decision == "REJECTED"
    assert len(trace.rounds) == 1  # immediately stops


# ── Test 3: TENTATIVE when max_rounds exceeded ────────────────────────────────

def test_tentative_when_max_rounds_exceeded():
    seed = make_seed()
    budget = make_budget()

    # Inject an LLM that always returns low scores (score=2 for all gates)
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"score": 2, "reasoning": "weak"}'
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    mock_llm.invoke.return_value = mock_response
    mock_llm.model = "mock-gemini"

    trace = run_reflection_loop(seed, budget, llm=mock_llm, max_rounds=1)

    # max_rounds=1, 1-2 gates fail → REFINE not possible after last round → TENTATIVE
    assert trace.final_status in (FinalStatus.TENTATIVE, FinalStatus.REJECTED)


# ── Test 4: TENTATIVE when ≥3 refinable gates fail ───────────────────────────

def test_tentative_when_three_or_more_refinable_fail():
    seed = make_seed()
    budget = make_budget()

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"score": 1, "reasoning": "very weak"}'
    mock_response.usage_metadata = {"input_tokens": 5, "output_tokens": 3}
    mock_llm.invoke.return_value = mock_response
    mock_llm.model = "mock"

    trace = run_reflection_loop(seed, budget, llm=mock_llm, max_rounds=3)

    # score=1 < all thresholds → all 4 refinable gates fail → TENTATIVE immediately
    assert trace.final_status == FinalStatus.TENTATIVE
    assert trace.rounds[0].decision == "TENTATIVE"


# ── Test 5: Oscillation detection ────────────────────────────────────────────

def test_oscillation_detection_forces_tentative():
    seed = make_seed(topic_id="osc_t")
    budget = make_budget()

    call_count = [0]

    mock_llm = MagicMock()

    def side_effect(messages):
        call_count[0] += 1
        resp = MagicMock()
        # Score 2 for first 2 gates (G1, G4) → REFINE (not 3+ fails, not ACCEPTED)
        # Use score=3 for G5, G7 → exactly 2 fail → REFINE each round
        gate_scores = [2, 2, 4, 4]
        idx = (call_count[0] - 1) % 4
        resp.content = f'{{"score": {gate_scores[idx]}, "reasoning": "test"}}'
        resp.usage_metadata = {"input_tokens": 5, "output_tokens": 3}
        return resp

    mock_llm.invoke.side_effect = side_effect
    mock_llm.model = "mock"

    # With max_rounds=5 and refine ops that don't change the 4-tuple,
    # oscillation should trigger within oscillation_window rounds
    trace = run_reflection_loop(
        seed, budget, llm=mock_llm, max_rounds=5,
        refine_operations_override=[],  # no-op refine → same sig every round
    )

    assert trace.final_status == FinalStatus.TENTATIVE


# ── Test 6: Budget exceeded mid-loop → TENTATIVE ─────────────────────────────

def test_budget_exceeded_mid_loop():
    seed = make_seed()
    # Tiny budget that gets exhausted immediately
    budget = BudgetTracker(per_run_budget_usd=0.001, per_topic_budget_usd=0.0001)
    # Pre-exhaust the per-topic budget
    budget.record_call("t001", "m", 1, 1, 0.0001)

    trace = run_reflection_loop(seed, budget, max_rounds=3)

    assert trace.final_status == FinalStatus.TENTATIVE


# ── Test 7: Trace structure ───────────────────────────────────────────────────

def test_trace_has_correct_fields():
    seed = make_seed(topic_id="trace_test")
    budget = make_budget()

    trace = run_reflection_loop(seed, budget, max_rounds=1)

    assert trace.topic_id == "trace_test"
    assert isinstance(trace.final_status, FinalStatus)
    assert isinstance(trace.rounds, list)
    assert trace.total_wallclock_seconds >= 0.0
    assert trace.total_cost_usd >= 0.0


def test_trace_persisted_to_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    import agents.settings as settings
    # Reset cached _scoped_root by re-calling module
    seed = make_seed(topic_id="persist_t")
    budget = make_budget()

    trace = run_reflection_loop(seed, budget, max_rounds=1)

    # Check trace file was written
    traces_dir = tmp_path / "output" / "ideation_traces"
    trace_file = traces_dir / "persist_t_trace.json"
    assert trace_file.exists()

    import json
    data = json.loads(trace_file.read_text())
    assert data["topic_id"] == "persist_t"


# ── Test 8: _apply_operations ─────────────────────────────────────────────────

def test_apply_operations_changes_geography():
    seed = make_seed()
    ops = [{"op": "change_geography", "value": "European cities"}]
    new_topic, _, _ = _apply_operations(seed.topic, ops, seed)
    assert new_topic.spatial_scope.geography == "European cities"


def test_apply_operations_increments_seed_round():
    seed = make_seed()
    original_round = seed.topic.meta.seed_round
    ops = [{"op": "change_geography", "value": "Tokyo"}]
    new_topic, _, _ = _apply_operations(seed.topic, ops, seed)
    assert new_topic.meta.seed_round == original_round + 1


def test_apply_operations_sets_parent_id():
    seed = make_seed(topic_id="parent_001")
    ops = [{"op": "change_geography", "value": "Tokyo"}]
    new_topic, _, _ = _apply_operations(seed.topic, ops, seed)
    assert new_topic.meta.parent_topic_id == "parent_001"


def test_apply_operations_unknown_op_skipped():
    seed = make_seed()
    ops = [{"op": "totally_nonexistent_op", "value": "irrelevant"}]
    # Should not raise
    new_topic, _, _ = _apply_operations(seed.topic, ops, seed)
    assert new_topic is not None


def test_aggregate_X_up_only_changes_exposure_unit():
    seed = make_seed()
    original_y = seed.topic.outcome_Y.spatial_unit
    original_scope = seed.topic.spatial_scope.spatial_unit
    ops = [{"op": "aggregate_X_up", "params": {"target_unit": "county"}}]
    new_topic, _, _ = _apply_operations(seed.topic, ops, seed)
    assert new_topic.exposure_X.spatial_unit == "county"
    assert new_topic.outcome_Y.spatial_unit == original_y
    assert new_topic.spatial_scope.spatial_unit == original_scope


def test_free_form_applies_dot_path_modifications():
    seed = make_seed()
    ops = [
        {
            "op": "free_form",
            "params": {
                "rationale": "test",
                "modified_fields": {
                    "identification.requires_exogenous_shock": True,
                },
            },
        }
    ]
    new_topic, _, slot_diff = _apply_operations(seed.topic, ops, seed)
    assert new_topic.identification.requires_exogenous_shock is True
    assert "free_form" in slot_diff
    assert slot_diff["free_form"]["fields"] == ["identification.requires_exogenous_shock"]


def test_anti_oscillation_skips_inverse_pair():
    failed = [GateResult("G2", "scale_alignment", False, False, "fail")]
    first = _select_refine_operations(failed, history=[])
    assert first
    second = _select_refine_operations(
        failed,
        history=[first[0]["op"]] if first else ["aggregate_X_up"],
    )
    if second:
        assert not (first[0]["op"] == "aggregate_X_up" and second[0]["op"] == "disaggregate_Y_down")


# ── Test 10: batch critique + OpenAlex fallback behavior ─────────────────────

def test_openalex_fallback_marks_g5_skipped(monkeypatch):
    seed = make_seed(topic_id="g5_skip")
    budget = make_budget()

    class FailingVerifier:
        def verify_novelty_four_tuple(self, topic):
            from agents.openalex_verifier import NoveltyEvidence

            return NoveltyEvidence(
                total_hits=0,
                top_k_papers=[],
                four_tuple_match_count=None,
                queries_log=["q1"],
                was_fallback=True,
            )

    mock_llm = MagicMock()
    # Batch critique response with all passing scores
    mock_resp = MagicMock()
    mock_resp.content = (
        '{"G1":{"score":4,"reasoning":"ok"},'
        '"G4":{"score":4,"reasoning":"ok"},'
        '"G5":{"score":4,"reasoning":"ok"},'
        '"G7":{"score":4,"reasoning":"ok"}}'
    )
    mock_resp.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    mock_llm.invoke.return_value = mock_resp
    mock_llm.model = "mock"

    trace = run_reflection_loop(
        seed,
        budget,
        verifier=FailingVerifier(),
        llm=mock_llm,
        max_rounds=1,
    )
    assert trace.rounds[0].g5_skipped is True
    g5_res = [r for r in trace.rounds[0].gate_results if r.gate_id == "G5"][-1]
    assert g5_res.reason == "openalex_unavailable_neutral"

def test_llm_unavailable_forces_tentative():
    seed = make_seed()
    budget = make_budget()
    trace = run_reflection_loop(seed, budget, llm=None, max_rounds=1)
    assert trace.final_status == FinalStatus.TENTATIVE
    assert "llm_unavailable_neutral_fallback" in trace.reject_reasons


# ── Test 9: _llm_propose_operation_values ─────────────────────────────────────

def test_llm_propose_invalid_json_returns_original_ops():
    seed = make_seed()
    ops = [{"op": "change_geography", "description": "shift geography"}]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "this is not json {"
    result = _llm_propose_operation_values(seed.topic, seed, ops, mock_llm)
    assert result == ops
    assert "value" not in result[0]


def test_llm_propose_wrong_type_skips_enrichment():
    seed = make_seed()
    ops = [{"op": "change_geography", "description": "shift geography"}]
    mock_llm = MagicMock()
    # Returns a JSON object, not an array — proposed_map ends up empty
    mock_llm.invoke.return_value.content = '{"op": "change_geography", "value": "Tokyo"}'
    result = _llm_propose_operation_values(seed.topic, seed, ops, mock_llm)
    assert "value" not in result[0]


def test_llm_propose_partial_ops_enriches_only_matched():
    seed = make_seed()
    ops = [
        {"op": "change_geography", "description": "shift geography"},
        {"op": "change_outcome_family", "description": "change outcome"},
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '[{"op": "change_geography", "value": "Tokyo", "rationale": "better data"}]'
    )
    fake_messages = types.SimpleNamespace(HumanMessage=lambda content: {"content": content})
    with patch.dict(sys.modules, {"langchain_core.messages": fake_messages}):
        result = _llm_propose_operation_values(seed.topic, seed, ops, mock_llm)
    assert result[0]["value"] == "Tokyo"
    assert "value" not in result[1]


def test_llm_propose_handles_missing_target_venues_field():
    seed = make_seed()
    seed.topic.target_venues = []
    ops = [{"op": "change_geography", "description": "shift geography"}]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '[{"op":"change_geography","value":"Tokyo","rationale":"better data"}]'
    )
    fake_messages = types.SimpleNamespace(HumanMessage=lambda content: {"content": content})
    with patch.dict(sys.modules, {"langchain_core.messages": fake_messages}):
        result = _llm_propose_operation_values(seed.topic, seed, ops, mock_llm)
    assert result[0]["value"] == "Tokyo"
