"""Per-topic reflection loop for Module 1."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

import yaml

from agents.budget_tracker import BudgetExceededError, BudgetTracker
from agents.logging_config import get_logger
from agents.openalex_verifier import NoveltyEvidence, OpenAlexVerifier
from agents.rule_engine import GateResult, RuleEngine
from agents.settings import ideation_traces_dir, prompts_dir, reflection_config_path
from models.topic_schema import FinalStatus, SeedCandidate, Topic

logger = get_logger(__name__)
_CRITIQUE_TEMPLATES_CACHE: Optional[dict] = None


def _load_critique_templates() -> dict:
    global _CRITIQUE_TEMPLATES_CACHE
    if _CRITIQUE_TEMPLATES_CACHE is not None:
        return _CRITIQUE_TEMPLATES_CACHE
    try:
        with open(prompts_dir() / "reflection_critique.txt") as f:
            _CRITIQUE_TEMPLATES_CACHE = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("reflection_critique.txt unavailable: %s", exc)
        _CRITIQUE_TEMPLATES_CACHE = {}
    return _CRITIQUE_TEMPLATES_CACHE


def clear_critique_templates_cache() -> None:
    global _CRITIQUE_TEMPLATES_CACHE
    _CRITIQUE_TEMPLATES_CACHE = None


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
    pre_refine_topic_snapshot: dict
    gate_results: list[GateResult]
    openalex_queries_log: list[str]
    llm_critique_raw: dict
    llm_calls: list[LLMCallRecord]
    decision: str
    applied_operations: list[dict]
    slot_diff: dict
    four_tuple_sig: str
    round_score: float
    budget_snapshot: dict
    wallclock_seconds: float
    g5_skipped: bool = False


@dataclass
class ReflectionTrace:
    topic_id: str
    seed_version: dict
    final_status: FinalStatus
    rounds: list[RoundRecord]
    reject_reasons: list[str]
    convergence: dict
    design_alternatives_considered: list[str]
    total_cost_usd: float
    total_wallclock_seconds: float
    final_topic: Optional[dict] = None


def _load_reflection_config() -> dict:
    try:
        with open(reflection_config_path()) as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("reflection_config.yaml unavailable: %s", exc)
        return {}


def _format_openalex_top_papers(novelty_evidence: Optional[NoveltyEvidence]) -> str:
    if not novelty_evidence or not novelty_evidence.top_k_papers:
        return "No OpenAlex papers returned."
    lines = []
    for idx, paper in enumerate(novelty_evidence.top_k_papers[:10], 1):
        title = paper.get("title", "Untitled")
        year = paper.get("publication_year", "N/A")
        cited = paper.get("cited_by_count", 0)
        venue = paper.get("venue", "unknown venue")
        lines.append(f'{idx}. "{title}" ({year}, cited={cited}) — {venue}')
    return "\n".join(lines)


def _set_nested(topic_dict: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = topic_dict
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _apply_operations(
    topic: Topic,
    operations: list[dict],
    seed_candidate: SeedCandidate,
) -> tuple[Topic, SeedCandidate, dict]:
    import copy

    topic_dict = topic.model_dump()
    original_topic = topic.model_dump()
    new_seed = copy.copy(seed_candidate)
    new_seed_round = topic.meta.seed_round + 1
    slot_diff: dict = {}

    for op in operations:
        op_name = op.get("op", "")
        params = op.get("params", {})
        value = op.get("value")
        if value is not None and isinstance(params, dict):
            params = {**params, "value": value}
        if not isinstance(params, dict):
            params = {}

        try:
            if op_name == "change_geography" and "value" in params:
                topic_dict["spatial_scope"]["geography"] = params["value"]
            elif op_name == "change_spatial_unit" and "value" in params:
                topic_dict["spatial_scope"]["spatial_unit"] = params["value"]
            elif op_name == "change_X_spatial_unit" and "value" in params:
                topic_dict["exposure_X"]["spatial_unit"] = params["value"]
            elif op_name == "change_Y_spatial_unit" and "value" in params:
                topic_dict["outcome_Y"]["spatial_unit"] = params["value"]
            elif op_name == "aggregate_X_up":
                unit = params.get("target_unit") or params.get("value")
                if unit:
                    topic_dict["exposure_X"]["spatial_unit"] = unit
            elif op_name == "disaggregate_Y_down":
                unit = params.get("target_unit") or params.get("value")
                if unit:
                    topic_dict["outcome_Y"]["spatial_unit"] = unit
            elif op_name == "harmonize_both_to":
                unit = params.get("common_unit") or params.get("value")
                if unit:
                    topic_dict["exposure_X"]["spatial_unit"] = unit
                    topic_dict["outcome_Y"]["spatial_unit"] = unit
            elif op_name == "add_fixed_effects":
                topic_dict["identification"]["primary"] = "fixed_effects"
            elif op_name == "switch_to_natural_experiment":
                topic_dict["identification"]["requires_exogenous_shock"] = True
                topic_dict["identification"]["primary"] = "diff_in_diff"
            elif op_name == "narrow_to_movers_sample":
                topic_dict["spatial_scope"]["sampling_mode"] = "longitudinal"
            elif op_name == "switch_to_RDD":
                topic_dict["identification"]["primary"] = "regression_discontinuity"
            elif op_name == "swap_geography":
                geo = params.get("new_geography") or params.get("value")
                if geo:
                    topic_dict["spatial_scope"]["geography"] = geo
            elif op_name == "swap_temporal":
                years = params.get("new_years")
                if isinstance(years, list) and len(years) >= 2:
                    topic_dict["temporal_scope"]["start_year"] = int(min(years))
                    topic_dict["temporal_scope"]["end_year"] = int(max(years))
            elif op_name == "swap_X_measurement":
                specific = params.get("new_specific")
                proxy = params.get("new_proxy")
                if specific:
                    topic_dict["exposure_X"]["specific_variable"] = specific
                if proxy:
                    topic_dict["exposure_X"]["measurement_proxy"] = proxy
            elif op_name == "add_heterogeneity_dim":
                dim = params.get("dimension")
                if dim:
                    statement = topic_dict["contribution"].get("statement", "")
                    topic_dict["contribution"]["statement"] = (
                        f"{statement} Includes heterogeneity by {dim}."
                    ).strip()
            elif op_name == "substitute_X_source":
                src = params.get("new_source") or params.get("value")
                if src:
                    new_seed.declared_sources = list(
                        dict.fromkeys([src, *new_seed.declared_sources])
                    )
            elif op_name == "narrow_scope":
                scope = params.get("new_scope") or params.get("value")
                if scope:
                    topic_dict["spatial_scope"]["geography"] = scope
            elif op_name == "shift_years":
                years = params.get("new_years")
                if isinstance(years, list) and len(years) >= 2:
                    topic_dict["temporal_scope"]["start_year"] = int(min(years))
                    topic_dict["temporal_scope"]["end_year"] = int(max(years))
            elif op_name == "sharpen_contribution":
                new_primary = params.get("new_primary")
                if new_primary:
                    topic_dict["contribution"]["primary"] = new_primary
                if params.get("new_statement"):
                    topic_dict["contribution"]["statement"] = params["new_statement"]
                if params.get("new_gap"):
                    topic_dict["contribution"]["gap_addressed"] = params["new_gap"]
            elif op_name == "free_form":
                modified = params.get("modified_fields", {})
                if isinstance(modified, dict):
                    for path, val in modified.items():
                        _set_nested(topic_dict, str(path), val)
                    slot_diff["free_form"] = {
                        "rationale": params.get("rationale", ""),
                        "fields": sorted(str(p) for p in modified.keys()),
                    }
            elif op_name == "change_exposure_family" and "value" in params:
                topic_dict["exposure_X"]["family"] = params["value"]
            elif op_name == "change_outcome_family" and "value" in params:
                topic_dict["outcome_Y"]["family"] = params["value"]
            elif op_name == "change_identification_strategy" and "value" in params:
                topic_dict["identification"]["primary"] = params["value"]
            elif op_name == "declare_additional_sources" and "value" in params:
                new_sources = params["value"] if isinstance(params["value"], list) else [params["value"]]
                new_seed.declared_sources = list(dict.fromkeys(new_seed.declared_sources + new_sources))
            elif op_name == "strengthen_contribution_statement" and "value" in params:
                topic_dict["contribution"]["statement"] = params["value"]
            elif op_name == "add_gap_addressed" and "value" in params:
                topic_dict["contribution"]["gap_addressed"] = params["value"]
        except Exception as exc:
            logger.warning("Failed to apply op %s: %s", op_name, exc)

    topic_dict["meta"]["seed_round"] = new_seed_round
    topic_dict["meta"]["parent_topic_id"] = topic.meta.topic_id
    new_topic = Topic.model_validate(topic_dict)

    for path in (
        "exposure_X.family",
        "exposure_X.specific_variable",
        "exposure_X.spatial_unit",
        "outcome_Y.family",
        "outcome_Y.specific_variable",
        "outcome_Y.spatial_unit",
        "spatial_scope.geography",
        "spatial_scope.spatial_unit",
        "temporal_scope.start_year",
        "temporal_scope.end_year",
        "identification.primary",
        "identification.requires_exogenous_shock",
        "contribution.primary",
        "contribution.statement",
        "contribution.gap_addressed",
    ):
        before = original_topic
        after = topic_dict
        for part in path.split("."):
            before = before.get(part, None) if isinstance(before, dict) else None
            after = after.get(part, None) if isinstance(after, dict) else None
        if before != after:
            slot_diff[path] = {"before": before, "after": after}

    return new_topic, new_seed, slot_diff


def _llm_score_gate(
    gate_id: str,
    gate_name: str,
    topic: Topic,
    novelty_evidence: Optional[NoveltyEvidence],
    llm,
) -> tuple[LLMCallRecord, dict]:
    templates = _load_critique_templates()
    template = templates.get(gate_id, {}).get("template", "")
    if not template:
        raise ValueError(f"Missing critique template for {gate_id}")

    prompt = template.format(
        exposure_variable=topic.exposure_X.specific_variable,
        exposure_family=topic.exposure_X.family.value,
        outcome_variable=topic.outcome_Y.specific_variable,
        outcome_family=topic.outcome_Y.family.value,
        geography=topic.spatial_scope.geography,
        spatial_unit=topic.spatial_scope.spatial_unit,
        method=topic.identification.primary.value,
        key_threats=topic.identification.key_threats,
        mitigations=topic.identification.mitigations,
        contribution_statement=topic.contribution.statement,
        gap_addressed=topic.contribution.gap_addressed or "",
        total_hits=getattr(novelty_evidence, "total_hits", 0),
        four_tuple_match_count=getattr(novelty_evidence, "four_tuple_match_count", 0),
        oa_top_papers_formatted=_format_openalex_top_papers(novelty_evidence),
        target_venues=topic.target_venues,
    )

    from langchain_core.messages import HumanMessage

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)
    usage = getattr(response, "usage_metadata", {}) or {}
    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)
    cost = tokens_in * 0.00000025 + tokens_out * 0.00000125
    score = parsed.get("score")
    score = int(score) if score is not None else None
    record = LLMCallRecord(
        gate_id=gate_id,
        model=getattr(llm, "model", "unknown"),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        score=score,
        reasoning=parsed.get("reasoning", ""),
    )
    return record, parsed


def _llm_score_all_refinable_gates(
    topic: Topic,
    novelty_evidence: Optional[NoveltyEvidence],
    llm,
) -> tuple[dict[str, LLMCallRecord], dict]:
    templates = _load_critique_templates()
    template = templates.get("BATCH", {}).get("template", "")
    if not template:
        raise ValueError("BATCH critique template missing")

    prompt = template.format(
        exposure_variable=topic.exposure_X.specific_variable,
        exposure_family=topic.exposure_X.family.value,
        outcome_variable=topic.outcome_Y.specific_variable,
        outcome_family=topic.outcome_Y.family.value,
        geography=topic.spatial_scope.geography,
        spatial_unit=topic.spatial_scope.spatial_unit,
        method=topic.identification.primary.value,
        key_threats=topic.identification.key_threats,
        mitigations=topic.identification.mitigations,
        contribution_statement=topic.contribution.statement,
        gap_addressed=topic.contribution.gap_addressed or "",
        total_hits=getattr(novelty_evidence, "total_hits", 0),
        four_tuple_match_count=getattr(novelty_evidence, "four_tuple_match_count", 0),
        oa_top_papers_formatted=_format_openalex_top_papers(novelty_evidence),
        target_venues=topic.target_venues,
    )
    from langchain_core.messages import HumanMessage

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)
    usage = getattr(response, "usage_metadata", {}) or {}
    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)
    total_cost = tokens_in * 0.00000025 + tokens_out * 0.00000125

    records: dict[str, LLMCallRecord] = {}
    for idx, gate_id in enumerate(("G1", "G4", "G5", "G7")):
        payload = parsed.get(gate_id, {})
        score = payload.get("score")
        score = int(score) if score is not None else None
        records[gate_id] = LLMCallRecord(
            gate_id=gate_id,
            model=getattr(llm, "model", "unknown"),
            tokens_in=tokens_in if idx == 0 else 0,
            tokens_out=tokens_out if idx == 0 else 0,
            cost_usd=total_cost if idx == 0 else 0.0,
            score=score,
            reasoning=payload.get("reasoning", ""),
        )
    return records, parsed


def _gate_result_from_llm(record: LLMCallRecord, gate_name: str, threshold: int) -> GateResult:
    if record.score is None:
        return GateResult(
            gate_id=record.gate_id,
            name=gate_name,
            passed=False,
            refinable=True,
            reason="llm_call_failed",
            score=None,
            max_score=5,
        )
    return GateResult(
        gate_id=record.gate_id,
        name=gate_name,
        passed=record.score >= threshold,
        refinable=True,
        reason="ok" if record.score >= threshold else f"score={record.score} < threshold={threshold}",
        score=record.score,
        max_score=5,
    )


def _llm_propose_operation_values(
    topic: Topic,
    seed_candidate: SeedCandidate,
    operations: list[dict],
    llm,
) -> list[dict]:
    if not llm or not operations:
        return operations

    try:
        template = (prompts_dir() / "reflection_refine.txt").read_text()
        prompt = template.format(
            topic_id=topic.meta.topic_id,
            exposure=f"{topic.exposure_X.specific_variable} ({topic.exposure_X.family.value})",
            outcome=f"{topic.outcome_Y.specific_variable} ({topic.outcome_Y.family.value})",
            geography=topic.spatial_scope.geography,
            method=topic.identification.primary.value,
            spatial_unit=topic.spatial_scope.spatial_unit,
            declared_sources=", ".join(seed_candidate.declared_sources),
            operations_list="\n".join(
                f"- {op.get('op')}: {op.get('description', '')}" for op in operations
            ),
            enum_block="",
            target_venues=topic.target_venues,
        )
        try:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
        except Exception:
            response = llm.invoke(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        proposed = json.loads(raw)
        if not isinstance(proposed, list):
            return operations

        mapped = {}
        for item in proposed:
            if isinstance(item, dict) and "op" in item:
                mapped[item["op"]] = item

        enriched = []
        for op in operations:
            payload = mapped.get(op.get("op", ""))
            if not payload:
                enriched.append(op)
                continue
            new_op = dict(op)
            if "value" in payload:
                new_op["value"] = payload["value"]
            if "params" in payload and isinstance(payload["params"], dict):
                new_op["params"] = payload["params"]
            if "rationale" in payload:
                new_op["rationale"] = payload["rationale"]
            enriched.append(new_op)
        return enriched
    except Exception as exc:
        logger.warning("LLM refine value proposal failed: %s", exc)
        return operations


def _select_refine_operations(
    failed_gates: list[GateResult],
    max_ops: int = 2,
    refine_operations_override: Optional[list[dict]] = None,
    history: Optional[list[str]] = None,
) -> list[dict]:
    if refine_operations_override is not None:
        return refine_operations_override[:max_ops]

    try:
        from agents.settings import refine_operations_path

        with open(refine_operations_path()) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}

    catalog = data.get("operations", [])
    anti_pairs = {
        tuple(pair)
        for pair in data.get("usage_rules", {}).get("anti_oscillation_pairs", [])
        if isinstance(pair, list) and len(pair) == 2
    }
    failed_ids = {r.gate_id for r in failed_gates}
    selected: list[dict] = []

    last_op = history[-1] if history else None
    for entry in catalog:
        targets = set(entry.get("targets_gate", []))
        if "any" not in targets and not (targets & failed_ids):
            continue
        op_name = entry.get("op")
        if not op_name:
            continue
        if last_op and (last_op, op_name) in anti_pairs:
            continue
        selected.append(
            {
                "op": op_name,
                "description": entry.get("description", ""),
                "params_schema": entry.get("params_schema", {}),
            }
        )
        if len(selected) >= max_ops:
            break

    if not selected:
        selected.append({"op": "free_form", "description": "fallback"})
    return selected


def _append_round(
    rounds: list[RoundRecord],
    round_num: int,
    pre_refine_topic_snapshot: dict,
    gate_results: list[GateResult],
    openalex_queries_log: list[str],
    llm_critique_raw: dict,
    llm_calls: list[LLMCallRecord],
    decision: str,
    applied_operations: list[dict],
    slot_diff: dict,
    four_tuple_sig: str,
    round_score: float,
    budget_snapshot: dict,
    wallclock_seconds: float,
    g5_skipped: bool = False,
) -> None:
    rounds.append(
        RoundRecord(
            round_num=round_num,
            pre_refine_topic_snapshot=pre_refine_topic_snapshot,
            gate_results=gate_results,
            openalex_queries_log=openalex_queries_log,
            llm_critique_raw={**llm_critique_raw, "g5_skipped": g5_skipped},
            llm_calls=llm_calls,
            decision=decision,
            applied_operations=applied_operations,
            slot_diff=slot_diff,
            four_tuple_sig=four_tuple_sig,
            round_score=round_score,
            budget_snapshot=budget_snapshot,
            wallclock_seconds=wallclock_seconds,
            g5_skipped=g5_skipped,
        )
    )


def run_reflection_loop(
    seed_candidate: SeedCandidate,
    budget: BudgetTracker,
    rule_engine: Optional[RuleEngine] = None,
    verifier: Optional[OpenAlexVerifier] = None,
    llm=None,
    max_rounds: Optional[int] = None,
    refine_operations_override: Optional[list[dict]] = None,
) -> ReflectionTrace:
    cfg = _load_reflection_config()
    r_cfg = cfg.get("reflection", {})
    _max_rounds = max_rounds if max_rounds is not None else r_cfg.get("max_rounds", 3)
    _min_rounds = r_cfg.get("min_rounds_before_stop", 2)
    _early_stop_delta = r_cfg.get("early_stop_delta_threshold", 0.5)
    _oscillation_window = r_cfg.get("oscillation_window", 3)

    rule_engine = rule_engine or RuleEngine()
    verifier = verifier or OpenAlexVerifier()

    topic_id = seed_candidate.topic.meta.topic_id
    rounds: list[RoundRecord] = []
    round_scores: list[float] = []
    signature_history: list[str] = []
    op_history: list[str] = []
    reject_reasons: list[str] = []
    final_status = FinalStatus.PENDING
    total_cost = 0.0
    start_wallclock = time.monotonic()
    current_candidate = seed_candidate
    force_tentative_due_to_no_llm = llm is None

    for round_num in range(1, _max_rounds + 1):
        round_start = time.monotonic()
        try:
            budget.check_can_proceed(topic_id)
        except BudgetExceededError as exc:
            reject_reasons.append(str(exc))
            final_status = FinalStatus.TENTATIVE
            break

        current_topic = current_candidate.topic
        four_tuple_sig = current_topic.four_tuple_signature()
        signature_history.append(four_tuple_sig)

        if len(signature_history) >= _oscillation_window and len(set(signature_history[-_oscillation_window:])) == 1:
            final_status = FinalStatus.TENTATIVE
            reject_reasons.append("oscillation_detected")
            _append_round(
                rounds,
                round_num,
                current_topic.model_dump(),
                [],
                [],
                {},
                [],
                "TENTATIVE",
                [],
                {},
                four_tuple_sig,
                0.0,
                budget.snapshot(),
                time.monotonic() - round_start,
            )
            break

        hard_results = rule_engine.run_hard_blockers(current_topic, current_candidate.declared_sources)
        hard_failed = [r for r in hard_results if not r.passed]
        if hard_failed:
            final_status = FinalStatus.REJECTED
            reject_reasons.extend([r.reason for r in hard_failed])
            _append_round(
                rounds,
                round_num,
                current_topic.model_dump(),
                hard_results,
                [],
                {},
                [],
                "REJECTED",
                [],
                {},
                four_tuple_sig,
                0.0,
                budget.snapshot(),
                time.monotonic() - round_start,
            )
            break

        g4_rule = rule_engine.check_G4_threat_coverage(current_topic)
        novelty_ev = None
        try:
            novelty_ev = verifier.verify_novelty_four_tuple(current_topic)
        except Exception as exc:
            logger.warning("OpenAlex verifier failed: %s", exc)

        llm_calls: list[LLMCallRecord] = []
        llm_critique_raw: dict = {}
        g5_skipped = novelty_ev is not None and novelty_ev.four_tuple_match_count is None
        refinable_results: list[GateResult] = []

        if llm is None:
            llm_critique_raw = {"reason": "llm_unavailable_neutral_fallback"}
            reject_reasons.append("llm_unavailable_neutral_fallback")
            for gate_id, gate_name, threshold in (
                ("G1", "mechanism_plausibility", 4),
                ("G4", "identification_validity", 4),
                ("G5", "novelty", 3),
                ("G7", "contribution_clarity", 4),
            ):
                if gate_id == "G4":
                    score = 4 if g4_rule.passed else 2
                    reason = g4_rule.reason
                else:
                    score = 3
                    reason = "mock_neutral"
                rec = LLMCallRecord(
                    gate_id=gate_id,
                    model="mock_neutral",
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    score=score,
                    reasoning=reason,
                )
                llm_calls.append(rec)
                refinable_results.append(_gate_result_from_llm(rec, gate_name, threshold))
        else:
            gate_records: dict[str, LLMCallRecord] = {}
            try:
                gate_records, llm_critique_raw = _llm_score_all_refinable_gates(current_topic, novelty_ev, llm)
            except Exception as exc:
                logger.warning("Batch critique failed: %s; fallback to per-gate", exc)
                for gate_id, gate_name in (
                    ("G1", "mechanism_plausibility"),
                    ("G4", "identification_validity"),
                    ("G5", "novelty"),
                    ("G7", "contribution_clarity"),
                ):
                    try:
                        rec, parsed = _llm_score_gate(gate_id, gate_name, current_topic, novelty_ev, llm)
                        gate_records[gate_id] = rec
                        llm_critique_raw[gate_id] = parsed
                    except Exception as gate_exc:
                        gate_records[gate_id] = LLMCallRecord(
                            gate_id=gate_id,
                            model=getattr(llm, "model", "unknown"),
                            tokens_in=0,
                            tokens_out=0,
                            cost_usd=0.0,
                            score=None,
                            reasoning=str(gate_exc),
                        )

            for gate_id, gate_name, threshold in (
                ("G1", "mechanism_plausibility", 4),
                ("G4", "identification_validity", 4),
                ("G5", "novelty", 3),
                ("G7", "contribution_clarity", 4),
            ):
                if gate_id == "G5" and g5_skipped:
                    refinable_results.append(
                        GateResult(
                            gate_id="G5",
                            name="novelty",
                            passed=True,
                            refinable=True,
                            reason="openalex_unavailable_neutral",
                            score=None,
                            max_score=5,
                        )
                    )
                    llm_calls.append(
                        LLMCallRecord(
                            gate_id="G5",
                            model="openalex_neutral",
                            tokens_in=0,
                            tokens_out=0,
                            cost_usd=0.0,
                            score=None,
                            reasoning="openalex_unavailable_neutral",
                        )
                    )
                    continue

                rec = gate_records.get(gate_id)
                if rec is None:
                    rec = LLMCallRecord(
                        gate_id=gate_id,
                        model=getattr(llm, "model", "unknown"),
                        tokens_in=0,
                        tokens_out=0,
                        cost_usd=0.0,
                        score=None,
                        reasoning="missing_gate_record",
                    )
                llm_calls.append(rec)
                if rec.cost_usd > 0:
                    budget.record_call(topic_id, rec.model, rec.tokens_in, rec.tokens_out, rec.cost_usd)
                    total_cost += rec.cost_usd
                refinable_results.append(_gate_result_from_llm(rec, gate_name, threshold))

        all_results = hard_results + [g4_rule] + refinable_results
        refinable_failed = [r for r in refinable_results if not r.passed]
        round_score = (
            sum(r.score or 0 for r in refinable_results if r.score is not None) / max(1, len(refinable_results))
        )
        round_scores.append(round_score)

        if not refinable_failed and not force_tentative_due_to_no_llm:
            final_status = FinalStatus.ACCEPTED
            _append_round(
                rounds,
                round_num,
                current_topic.model_dump(),
                all_results,
                novelty_ev.queries_log if novelty_ev else [],
                llm_critique_raw,
                llm_calls,
                "ACCEPTED",
                [],
                {},
                four_tuple_sig,
                round_score,
                budget.snapshot(),
                time.monotonic() - round_start,
                g5_skipped=g5_skipped,
            )
            break

        if len(refinable_failed) >= 3 or round_num == _max_rounds or (
            round_num >= _min_rounds and len(round_scores) >= 2 and abs(round_scores[-1] - round_scores[-2]) < _early_stop_delta
        ):
            final_status = FinalStatus.TENTATIVE
            _append_round(
                rounds,
                round_num,
                current_topic.model_dump(),
                all_results,
                novelty_ev.queries_log if novelty_ev else [],
                llm_critique_raw,
                llm_calls,
                "TENTATIVE",
                [],
                {},
                four_tuple_sig,
                round_score,
                budget.snapshot(),
                time.monotonic() - round_start,
                g5_skipped=g5_skipped,
            )
            break

        ops = _select_refine_operations(
            refinable_failed,
            refine_operations_override=refine_operations_override,
            history=op_history,
        )
        if refine_operations_override is None:
            ops = _llm_propose_operation_values(current_topic, current_candidate, ops, llm)
        new_topic, new_candidate, slot_diff = _apply_operations(current_topic, ops, current_candidate)
        op_history.extend([op.get("op", "") for op in ops if op.get("op")])
        current_candidate = SeedCandidate(
            topic=new_topic,
            declared_sources=new_candidate.declared_sources,
            declared_sources_rationale=new_candidate.declared_sources_rationale,
        )
        _append_round(
            rounds,
            round_num,
            current_topic.model_dump(),
            all_results,
            novelty_ev.queries_log if novelty_ev else [],
            llm_critique_raw,
            llm_calls,
            "REFINE",
            ops,
            slot_diff,
            four_tuple_sig,
            round_score,
            budget.snapshot(),
            time.monotonic() - round_start,
            g5_skipped=g5_skipped,
        )

    total_wallclock = time.monotonic() - start_wallclock
    trace = ReflectionTrace(
        topic_id=topic_id,
        seed_version=seed_candidate.topic.model_dump(),
        final_status=final_status,
        rounds=rounds,
        reject_reasons=reject_reasons,
        convergence={
            "score_trajectory": round_scores,
            "signature_history": signature_history,
            "early_stop_reason": reject_reasons[-1] if reject_reasons else "",
        },
        design_alternatives_considered=[op for op in op_history if op],
        total_cost_usd=total_cost,
        total_wallclock_seconds=total_wallclock,
        final_topic=current_candidate.topic.model_dump(),
    )
    _persist_trace(trace)
    return trace


def _persist_trace(trace: ReflectionTrace) -> None:
    try:
        path = ideation_traces_dir() / f"{trace.topic_id}_trace.json"
        with open(path, "w") as f:
            json.dump(asdict(trace), f, indent=2)
    except Exception as exc:
        logger.warning("Failed to persist trace for %s: %s", trace.topic_id, exc)
