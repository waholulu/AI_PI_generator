"""
Per-topic reflection loop for Module 1 — iterative topic refinement.

Implements the full decision table from the spec:
  - Hard-blocker fail (G2/G3/G6) → REJECTED immediately (no LLM cost)
  - All refinable pass → ACCEPTED
  - 1-2 refinable fail → REFINE (next round)
  - ≥3 refinable fail → TENTATIVE
  - max_rounds reached without ACCEPTED → TENTATIVE
  - Early-stop: rounds ≥ min_rounds and score delta < 0.5 → TENTATIVE
  - Oscillation: same 4-tuple signature for 3 consecutive rounds → TENTATIVE

LLM calls (G1/G4_llm/G5_llm/G7) use the fast Gemini model from env vars.
All LLM costs are recorded via BudgetTracker.
Traces are persisted to output/ideation_traces/{topic_id}_trace.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml

from agents.budget_tracker import BudgetExceededError, BudgetTracker
from agents.logging_config import get_logger
from agents.openalex_verifier import NoveltyEvidence, OpenAlexVerifier
from agents.rule_engine import GateResult, RuleEngine
from agents.settings import ideation_traces_dir, reflection_config_path
from models.topic_schema import FinalStatus, SeedCandidate, Topic

logger = get_logger(__name__)


# ── Trace data structures ─────────────────────────────────────────────────────

@dataclass
class LLMCallRecord:
    gate_id: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    score: Optional[int]
    reasoning: str


@dataclass
class RoundRecord:
    round_num: int
    gate_results: list[GateResult]
    llm_calls: list[LLMCallRecord]
    decision: str          # ACCEPTED / REFINE / TENTATIVE / REJECTED
    applied_operations: list[dict]
    four_tuple_sig: str
    round_score: float     # mean refinable score for this round


@dataclass
class ReflectionTrace:
    topic_id: str
    final_status: FinalStatus
    rounds: list[RoundRecord]
    total_cost_usd: float
    total_wallclock_seconds: float
    final_topic: Optional[dict] = None  # serialized Topic dict


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_reflection_config() -> dict:
    try:
        with open(reflection_config_path()) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("reflection_config.yaml unavailable: %s", e)
        return {}


# ── Refine operation applicator ───────────────────────────────────────────────

def _apply_operations(
    topic: Topic,
    operations: list[dict],
    seed_candidate: SeedCandidate,
) -> tuple[Topic, SeedCandidate]:
    """Apply suggested refine operations to produce a modified Topic copy.

    Operations reference field paths on the Topic model and carry a 'value'
    key with the proposed replacement.  Unknown ops are skipped with a warning.
    """
    import copy

    topic_dict = topic.model_dump()
    new_parent_id = topic.meta.topic_id
    new_seed_round = topic.meta.seed_round + 1

    for op in operations:
        op_name = op.get("op", "")
        value = op.get("value")
        if value is None:
            continue

        try:
            if op_name == "change_exposure_family":
                topic_dict["exposure_X"]["family"] = value
            elif op_name == "narrow_exposure_variable":
                topic_dict["exposure_X"]["specific_variable"] = value
            elif op_name == "change_outcome_family":
                topic_dict["outcome_Y"]["family"] = value
            elif op_name == "narrow_outcome_variable":
                topic_dict["outcome_Y"]["specific_variable"] = value
            elif op_name == "change_identification_strategy":
                topic_dict["identification"]["primary"] = value
            elif op_name == "add_key_threats":
                existing = topic_dict["identification"].get("key_threats", [])
                new_threats = value if isinstance(value, list) else [value]
                topic_dict["identification"]["key_threats"] = list(
                    dict.fromkeys(existing + new_threats)
                )
            elif op_name == "add_mitigations":
                existing = topic_dict["identification"].get("mitigations", [])
                new_mit = value if isinstance(value, list) else [value]
                topic_dict["identification"]["mitigations"] = list(
                    dict.fromkeys(existing + new_mit)
                )
            elif op_name == "change_geography":
                topic_dict["spatial_scope"]["geography"] = value
            elif op_name == "change_spatial_unit":
                topic_dict["spatial_scope"]["spatial_unit"] = value
                topic_dict["exposure_X"]["spatial_unit"] = value
                topic_dict["outcome_Y"]["spatial_unit"] = value
            elif op_name == "change_sampling_mode":
                topic_dict["spatial_scope"]["sampling_mode"] = value
            elif op_name == "adjust_temporal_scope":
                if isinstance(value, dict):
                    topic_dict["temporal_scope"].update(value)
            elif op_name == "change_frequency":
                topic_dict["temporal_scope"]["frequency"] = value
            elif op_name == "declare_additional_sources":
                new_sources = value if isinstance(value, list) else [value]
                seed_candidate = copy.copy(seed_candidate)
                seed_candidate.declared_sources = list(
                    dict.fromkeys(seed_candidate.declared_sources + new_sources)
                )
            elif op_name == "change_contribution_type":
                topic_dict["contribution"]["primary"] = value
            elif op_name == "strengthen_contribution_statement":
                topic_dict["contribution"]["statement"] = value
            elif op_name == "add_gap_addressed":
                topic_dict["contribution"]["gap_addressed"] = value
            elif op_name == "add_measurement_proxy":
                topic_dict["exposure_X"]["measurement_proxy"] = value
            else:
                logger.warning("Unknown refine operation: %s — skipped", op_name)
        except Exception as e:
            logger.warning("Failed to apply op %s: %s", op_name, e)

    topic_dict["meta"]["seed_round"] = new_seed_round
    topic_dict["meta"]["parent_topic_id"] = new_parent_id

    new_topic = Topic.model_validate(topic_dict)
    return new_topic, seed_candidate


# ── LLM gate scorer ───────────────────────────────────────────────────────────

def _llm_score_gate(
    gate_id: str,
    gate_name: str,
    topic: Topic,
    novelty_evidence: Optional[NoveltyEvidence],
    llm,
) -> LLMCallRecord:
    """Call LLM to score a single refinable gate; return LLMCallRecord."""
    prompts = {
        "G1": (
            f"Score 1-5 the mechanistic plausibility of:\n"
            f"Exposure: {topic.exposure_X.specific_variable} ({topic.exposure_X.family.value})\n"
            f"Outcome: {topic.outcome_Y.specific_variable} ({topic.outcome_Y.family.value})\n"
            f"Geography: {topic.spatial_scope.geography}\n"
            "Reply: {\"score\": <1-5>, \"reasoning\": \"...\"}  (JSON only)"
        ),
        "G4": (
            f"Score 1-5 the validity of this identification strategy:\n"
            f"Method: {topic.identification.primary.value}\n"
            f"Key threats: {topic.identification.key_threats}\n"
            f"Mitigations: {topic.identification.mitigations}\n"
            "Reply: {\"score\": <1-5>, \"reasoning\": \"...\"}  (JSON only)"
        ),
        "G5": (
            f"Score 1-5 the novelty of this research topic given "
            f"{getattr(novelty_evidence, 'four_tuple_match_count', 0)} exact four-tuple matches "
            f"out of {getattr(novelty_evidence, 'total_hits', 0)} papers found:\n"
            f"Topic: {topic.free_form_title or topic.exposure_X.specific_variable} → "
            f"{topic.outcome_Y.specific_variable}\n"
            "Reply: {\"score\": <1-5>, \"reasoning\": \"...\"}  (JSON only)"
        ),
        "G7": (
            f"Score 1-5 the clarity of this contribution statement:\n"
            f"\"{topic.contribution.statement}\"\n"
            f"Gap addressed: \"{topic.contribution.gap_addressed}\"\n"
            "Reply: {\"score\": <1-5>, \"reasoning\": \"...\"}  (JSON only)"
        ),
    }
    prompt = prompts.get(gate_id, f"Score 1-5: {gate_name}. Reply JSON only.")

    try:
        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        score = int(parsed.get("score", 3))
        reasoning = parsed.get("reasoning", "")
        usage = getattr(response, "usage_metadata", {}) or {}
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        cost = (tokens_in * 0.00000025 + tokens_out * 0.00000125)  # rough Gemini flash estimate
    except Exception as e:
        logger.warning("LLM gate %s scoring failed: %s — defaulting score=3", gate_id, e)
        score, reasoning, tokens_in, tokens_out, cost = 3, str(e), 0, 0, 0.0

    model_name = getattr(llm, "model", "unknown")
    return LLMCallRecord(
        gate_id=gate_id,
        model=model_name,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        score=score,
        reasoning=reasoning,
    )


def _gate_result_from_llm(record: LLMCallRecord, gate_name: str, threshold: int) -> GateResult:
    passed = (record.score or 0) >= threshold
    return GateResult(
        gate_id=record.gate_id,
        name=gate_name,
        passed=passed,
        refinable=True,
        reason="ok" if passed else f"score={record.score} < threshold={threshold}",
        score=record.score,
        max_score=5,
    )


# ── Main reflection loop ──────────────────────────────────────────────────────

def run_reflection_loop(
    seed_candidate: SeedCandidate,
    budget: BudgetTracker,
    rule_engine: Optional[RuleEngine] = None,
    verifier: Optional[OpenAlexVerifier] = None,
    llm=None,
    max_rounds: Optional[int] = None,
    refine_operations_override: Optional[list[dict]] = None,
) -> ReflectionTrace:
    """Run the per-topic reflection loop and return a complete ReflectionTrace.

    Parameters
    ----------
    seed_candidate:
        The topic + declared_sources to evaluate.
    budget:
        Shared BudgetTracker; check_can_proceed / record_call called each round.
    rule_engine:
        Pre-built RuleEngine; created fresh if None.
    verifier:
        Pre-built OpenAlexVerifier; created fresh if None.
    llm:
        LangChain LLM instance for G1/G4/G5/G7 scoring.  If None, LLM gates
        default to score=3 (neutral/passing) so the loop can run without API keys.
    max_rounds:
        Override reflection_config.yaml max_rounds.
    """
    cfg = _load_reflection_config()
    r_cfg = cfg.get("reflection", {})

    _max_rounds = max_rounds if max_rounds is not None else r_cfg.get("max_rounds", 3)
    _min_rounds = r_cfg.get("min_rounds_before_stop", 2)
    _early_stop_delta = r_cfg.get("early_stop_delta_threshold", 0.5)
    _oscillation_window = r_cfg.get("oscillation_window", 3)

    if rule_engine is None:
        rule_engine = RuleEngine()
    if verifier is None:
        verifier = OpenAlexVerifier()

    topic = seed_candidate.topic
    declared = seed_candidate.declared_sources

    topic_id = topic.meta.topic_id
    rounds: list[RoundRecord] = []
    total_cost = 0.0
    t_start = time.monotonic()

    recent_sigs: list[str] = []
    round_scores: list[float] = []
    final_status = FinalStatus.PENDING
    current_candidate = seed_candidate

    for round_num in range(1, _max_rounds + 1):
        # Budget guard
        try:
            budget.check_can_proceed(topic_id)
        except BudgetExceededError as e:
            logger.warning("Budget exceeded for %s: %s", topic_id, e)
            final_status = FinalStatus.TENTATIVE
            break

        current_topic = current_candidate.topic
        current_declared = current_candidate.declared_sources
        sig = current_topic.four_tuple_signature()
        recent_sigs.append(sig)

        # Oscillation check
        if (
            len(recent_sigs) >= _oscillation_window
            and len(set(recent_sigs[-_oscillation_window:])) == 1
        ):
            logger.info("Oscillation detected for %s — forcing TENTATIVE", topic_id)
            final_status = FinalStatus.TENTATIVE
            _append_round(rounds, round_num, [], [], "TENTATIVE", [], sig, 0.0)
            break

        # ── Zero-cost hard-blocker checks ─────────────────────────────────
        hard_results = rule_engine.run_hard_blockers(current_topic, current_declared)
        hard_failed = [r for r in hard_results if not r.passed]

        if hard_failed:
            logger.info("Hard-blocker failed for %s: %s", topic_id,
                        [r.gate_id for r in hard_failed])
            _append_round(rounds, round_num, hard_results, [], "REJECTED", [], sig, 0.0)
            final_status = FinalStatus.REJECTED
            break

        # G4 rule-engine part (zero cost)
        g4_rule = rule_engine.check_G4_threat_coverage(current_topic)

        # ── OpenAlex novelty evidence ─────────────────────────────────────
        novelty_ev: Optional[NoveltyEvidence] = None
        try:
            novelty_ev = verifier.verify_novelty_four_tuple(current_topic)
        except Exception as e:
            logger.warning("OpenAlex verifier failed: %s", e)

        # ── LLM refinable gates ───────────────────────────────────────────
        llm_calls: list[LLMCallRecord] = []
        refinable_results: list[GateResult] = []

        for gate_id, gate_name, threshold in [
            ("G1", "mechanism_plausibility", 4),
            ("G4", "identification_validity", 4),
            ("G5", "novelty", 3),
            ("G7", "contribution_clarity", 4),
        ]:
            if llm is not None:
                rec = _llm_score_gate(gate_id, gate_name, current_topic, novelty_ev, llm)
            else:
                # No LLM: use rule-engine G4 result for G4, neutral score=4 for others
                if gate_id == "G4":
                    rec = LLMCallRecord(
                        gate_id="G4", model="rule_engine",
                        tokens_in=0, tokens_out=0, cost_usd=0.0,
                        score=4 if g4_rule.passed else 2,
                        reasoning=g4_rule.reason,
                    )
                else:
                    rec = LLMCallRecord(
                        gate_id=gate_id, model="mock_neutral",
                        tokens_in=0, tokens_out=0, cost_usd=0.0,
                        score=4, reasoning="no_llm_neutral",
                    )
            llm_calls.append(rec)

            gate_res = _gate_result_from_llm(rec, gate_name, threshold)
            refinable_results.append(gate_res)

            if rec.cost_usd > 0:
                budget.record_call(topic_id, rec.model,
                                   rec.tokens_in, rec.tokens_out, rec.cost_usd)
                total_cost += rec.cost_usd

        all_results = hard_results + [g4_rule] + refinable_results
        refinable_failed = [r for r in refinable_results if not r.passed]
        n_failed = len(refinable_failed)

        round_score = (
            sum(r.score or 0 for r in refinable_results) / len(refinable_results)
            if refinable_results else 0.0
        )
        round_scores.append(round_score)

        # ── Decision ──────────────────────────────────────────────────────
        if n_failed == 0:
            decision = "ACCEPTED"
            _append_round(rounds, round_num, all_results, llm_calls,
                          decision, [], sig, round_score)
            final_status = FinalStatus.ACCEPTED
            break

        if n_failed >= 3:
            decision = "TENTATIVE"
            _append_round(rounds, round_num, all_results, llm_calls,
                          decision, [], sig, round_score)
            final_status = FinalStatus.TENTATIVE
            break

        # Early-stop check (after min_rounds)
        if round_num >= _min_rounds and len(round_scores) >= 2:
            delta = abs(round_scores[-1] - round_scores[-2])
            if delta < _early_stop_delta:
                logger.info("Early stop (delta=%.3f) for %s", delta, topic_id)
                decision = "TENTATIVE"
                _append_round(rounds, round_num, all_results, llm_calls,
                              decision, [], sig, round_score)
                final_status = FinalStatus.TENTATIVE
                break

        if round_num == _max_rounds:
            decision = "TENTATIVE"
            _append_round(rounds, round_num, all_results, llm_calls,
                          decision, [], sig, round_score)
            final_status = FinalStatus.TENTATIVE
            break

        # ── Refine: build operations and apply ────────────────────────────
        operations = _select_refine_operations(
            refinable_failed,
            refine_operations_override=refine_operations_override,
        )
        new_topic, new_candidate = _apply_operations(
            current_topic, operations, current_candidate
        )
        current_candidate = type(current_candidate)(
            topic=new_topic,
            declared_sources=new_candidate.declared_sources,
            declared_sources_rationale=new_candidate.declared_sources_rationale,
        )

        decision = "REFINE"
        _append_round(rounds, round_num, all_results, llm_calls,
                      decision, operations, sig, round_score)

    wallclock = time.monotonic() - t_start

    trace = ReflectionTrace(
        topic_id=topic_id,
        final_status=final_status,
        rounds=rounds,
        total_cost_usd=total_cost,
        total_wallclock_seconds=wallclock,
        final_topic=current_candidate.topic.model_dump() if current_candidate else None,
    )

    _persist_trace(trace)
    return trace


def _append_round(
    rounds: list[RoundRecord],
    num: int,
    gate_results: list[GateResult],
    llm_calls: list[LLMCallRecord],
    decision: str,
    ops: list[dict],
    sig: str,
    score: float,
) -> None:
    rounds.append(RoundRecord(
        round_num=num,
        gate_results=gate_results,
        llm_calls=llm_calls,
        decision=decision,
        applied_operations=ops,
        four_tuple_sig=sig,
        round_score=score,
    ))


def _select_refine_operations(
    failed_gates: list[GateResult],
    max_ops: int = 2,
    refine_operations_override: Optional[list[dict]] = None,
) -> list[dict]:
    """Select ≤ max_ops refine operations based on which gates failed."""
    if refine_operations_override is not None:
        return refine_operations_override[:max_ops]

    try:
        from agents.settings import refine_operations_path
        with open(refine_operations_path()) as f:
            import yaml as _yaml
            data = _yaml.safe_load(f) or {}
        catalog = data.get("operations", [])
    except Exception:
        catalog = []

    failed_ids = {r.gate_id + "_fail" for r in failed_gates}
    selected: list[dict] = []
    for entry in catalog:
        if any(tag in failed_ids for tag in entry.get("applicable_when", [])):
            selected.append({"op": entry["op"], "description": entry.get("description", "")})
            if len(selected) >= max_ops:
                break

    return selected


def _persist_trace(trace: ReflectionTrace) -> None:
    try:
        traces_dir = ideation_traces_dir()
        path = traces_dir / f"{trace.topic_id}_trace.json"

        def _serialize(obj):
            if isinstance(obj, (GateResult, LLMCallRecord, RoundRecord, ReflectionTrace)):
                return asdict(obj)
            if hasattr(obj, "value"):
                return obj.value
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return str(obj)

        import json as _json

        with open(path, "w") as f:
            _json.dump(asdict(trace), f, default=_serialize, indent=2)
    except Exception as e:
        logger.warning("Failed to persist trace for %s: %s", trace.topic_id, e)
