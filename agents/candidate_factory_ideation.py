"""Deterministic candidate factory ideation path (Stage 1 of two-stage flow).

Called by ideation_node() as the production default. Replaces LLM seed
generation with deterministic Cartesian-product expansion via
compose_candidates(), then runs precheck → repair → format display cards.

Stage 1 (this file) is intentionally cheap:
  - No OpenAlex novelty calls (cards carry novelty_status="pending_literature_check")
  - No development pack generation (deferred to Stage 2 after user selection)
  - Diversity selection on the shortlist so users don't see one-exposure × N-outcome variants

Stage 2 (orchestrator: novelty_check → literature → development_pack → drafter →
data_fetcher) runs only for the candidate the user picks at the HITL checkpoint.
"""
from __future__ import annotations

import json

from agents import settings
from agents.candidate_composer import compose_candidates
from agents.candidate_export_validator import validate_candidate_export_contract
from agents.candidate_feasibility import precheck_candidate
from agents.candidate_flag_classifier import compute_candidate_readiness
from agents.candidate_output_writer import (
    write_development_pack_index,
    write_feasibility_report,
    write_gate_trace,
)
from agents.candidate_repair import repair_candidate
from agents.candidate_reranker import rerank_candidates
from agents.identification_template_filler import ensure_identification_metadata
from agents.development_pack_status import evaluate_development_pack_readiness
from agents.final_ranker import score_candidate
from agents.logging_config import get_logger
from agents.research_template_loader import load_research_template
from agents.training_feasibility import evaluate_training_candidate
from models.candidate_composer_schema import ComposeRequest

logger = get_logger(__name__)


# ── Display title templates ─────────────────────────────────────────────────
# Maps (exposure_family, outcome_family) → research-question-style template.
# Templates use {exposure_phrase} / {outcome_phrase} placeholders.  Fallback
# generic template kicks in for combinations not listed here.  Polish is
# scoped to built_environment_health_us_tract for now per the two-stage plan.
DISPLAY_TITLE_TEMPLATES: dict[tuple[str, str], str] = {
    ("street_network", "respiratory"):
        "Can {exposure_phrase} Help Explain Neighborhood {outcome_phrase} Burden?",
    ("street_network", "cardiometabolic"):
        "Does {exposure_phrase} Shape Cardiometabolic Risk Across US Census Tracts?",
    ("street_network", "physical_activity"):
        "How Does {exposure_phrase} Relate to {outcome_phrase} at the Census-Tract Level?",
    ("greenspace", "respiratory"):
        "Is Neighborhood {exposure_phrase} Protective Against {outcome_phrase}?",
    ("greenspace", "mental_health"):
        "Does Neighborhood {exposure_phrase} Influence {outcome_phrase} Outcomes?",
    ("greenspace", "cardiometabolic"):
        "Can {exposure_phrase} Reduce Cardiometabolic Risk in Urban Tracts?",
    ("walkability", "physical_activity"):
        "How Strongly Does {exposure_phrase} Predict Tract-Level {outcome_phrase}?",
    ("walkability", "obesity"):
        "Is {exposure_phrase} Associated with Lower Tract-Level {outcome_phrase} Prevalence?",
    ("air_pollution", "respiratory"):
        "How Much of Neighborhood {outcome_phrase} Burden Tracks {exposure_phrase} Exposure?",
    ("food_environment", "obesity"):
        "Does the {exposure_phrase} Predict {outcome_phrase} Across Census Tracts?",
}

_GENERIC_TITLE_TEMPLATE = (
    "How Does {exposure_phrase} Relate to {outcome_phrase} at the {unit} Level?"
)


# Training_research strategy → human-readable phrase used in titles, research
# questions, and dev-pack copy. Shared between _make_title / _make_research_question
# (legacy contract fields) and _format_training_card (display layer).
_STRATEGY_PHRASES: dict[str, str] = {
    "sft_full_finetune": "supervised fine-tuning",
    "lora_adapter": "LoRA adapter fine-tuning",
    "qlora_4bit": "4-bit QLoRA fine-tuning",
}


def _humanize(token: str) -> str:
    return token.replace("_", " ").strip()


def _humanize_unit(unit: str) -> str:
    mapping = {
        "us_census_tract": "Census-Tract",
        "us_county": "County",
        "us_state": "State",
        "us_zcta": "ZCTA",
    }
    return mapping.get(unit, _humanize(unit).title())


def _make_title(c) -> str:
    """Legacy contract-style title (e.g. 'Street Network and Respiratory').

    Retained on the card for backward compatibility; the user-visible label
    is `display.display_title` produced by `format_display_card()`.
    """
    if getattr(c, "unit_of_analysis", "") == "training_run":
        strategy = getattr(c, "exposure_family", "") or ""
        task_label = getattr(c, "outcome_task_label", None)
        if task_label:
            return f"{_humanize(strategy).title()} → {task_label}"
        out = getattr(c, "outcome_family", "").replace("_", " ").title()
        return f"{_humanize(strategy).title()} → {out}"
    exp = c.exposure_family.replace("_", " ").title()
    out = c.outcome_family.replace("_", " ").title()
    return f"{exp} and {out}"


def _make_research_question(c) -> str:
    if getattr(c, "unit_of_analysis", "") == "training_run":
        strategy_phrase = _STRATEGY_PHRASES.get(
            getattr(c, "exposure_family", ""), _humanize(getattr(c, "exposure_family", ""))
        )
        task_label = getattr(c, "outcome_task_label", None) or _humanize(
            getattr(c, "outcome_family", "")
        ).title()
        metric_phrase = _humanize(getattr(c, "outcome_family", "")) or "task performance"
        return (
            f"Does {strategy_phrase} improve {metric_phrase} on \"{task_label}\" "
            f"relative to a zero-shot baseline?"
        )
    exp = c.exposure_family.replace("_", " ")
    out = c.outcome_family.replace("_", " ")
    return f"How does {exp} affect {out} at the {c.unit_of_analysis} level?"


def _format_training_card(candidate, gate_status: dict | None = None) -> dict:
    """Display formatter for training_research candidates.

    Generates LLM-flavoured text grounded in the domain-derived task
    (`outcome_task_*` fields) rather than the spatial-research boilerplate.
    """
    strategy = getattr(candidate, "exposure_family", "") or ""
    strategy_phrase = _STRATEGY_PHRASES.get(strategy, _humanize(strategy))
    metric_family = getattr(candidate, "outcome_family", "") or ""
    metric_phrase = _humanize(metric_family) or "task performance"
    task_label = getattr(candidate, "outcome_task_label", "") or _humanize(metric_family).title()
    task_description = (getattr(candidate, "outcome_task_description", "") or "").strip()
    dataset_hint = getattr(candidate, "outcome_task_dataset_hint", "") or ""
    domain_input = (getattr(candidate, "outcome_task_domain_input", "") or "").strip()
    method_template = getattr(candidate, "method_template", "") or "paired baseline-vs-treatment"

    # The fallback path injects "(placeholder — set an LLM API key for ...)"
    # into task_description. Detect this so the rationale shown to the user
    # stays clean instead of embedding internal warnings as if they were
    # research justification.
    is_placeholder_task = (
        "placeholder" in task_description.lower()
        or "gemini was unavailable" in task_description.lower()
    )

    display_title = f"Can {strategy_phrase.title()} Improve {task_label}?"

    research_question = (
        f"Does {strategy_phrase} of an open base language model improve "
        f"{metric_phrase} on the task \"{task_label}\" relative to a "
        f"zero-shot baseline, under a {_humanize(method_template).lower()} "
        f"protocol?"
    )

    domain_clause = f' Domain: "{domain_input}".' if domain_input else ""
    if task_description and not is_placeholder_task:
        rationale = (
            f"{task_description.rstrip('.')}.{domain_clause} "
            f"Testing this with {strategy_phrase} on Colab-tier hardware "
            f"probes whether parameter-efficient adaptation delivers "
            f"measurable gains on a domain-specific task."
        )
    else:
        rationale = (
            f"Apply {strategy_phrase} to a {task_label.lower()} derived from "
            f"the user's domain{(' (' + domain_input + ')') if domain_input else ''}, "
            f"comparing against a zero-shot baseline on Colab-tier hardware. "
            f"Set an LLM API key to replace this placeholder task with a "
            f"domain-grounded set."
        )

    contribution_angle = (
        f"Pairs a domain-grounded task ({task_label}) with {strategy_phrase} "
        f"and a paired baseline-vs-treatment protocol, runnable end-to-end on "
        f"Colab T4."
        + (f" Suggested data: {dataset_hint}." if dataset_hint else "")
    )

    execution_summary = (
        f"Strategy: {strategy_phrase}. "
        f"Task: {task_label} ({metric_phrase}). "
        f"Method: {_humanize(method_template).lower()}. "
        f"Unit: training_run."
    )

    return {
        "display_title": display_title,
        "research_question": research_question,
        "rationale": rationale,
        "contribution_angle": contribution_angle,
        "execution_summary": execution_summary,
        "novelty_status": "pending_literature_check",
    }


def format_display_card(candidate, gate_status: dict | None = None) -> dict:
    """Convert a composed candidate contract into a research-question-style card.

    Returns a dict with keys:
      display_title       — research-question phrasing the user sees first
      research_question   — full sentence with geography + unit_of_analysis
      rationale           — why this matters (1–2 sentences)
      contribution_angle  — what's new about this combination
      execution_summary   — folded layer-2 string (sources + method + grain)

    Pure function — no I/O, no LLM calls.  Lives here (not a separate agent)
    so display formatting stays close to the contract that produced it.

    Branches on unit_of_analysis: training_research candidates (unit ==
    "training_run") get LLM-flavoured copy via `_format_training_card`;
    spatial-research candidates keep the built-environment-flavoured copy
    below.
    """
    if getattr(candidate, "unit_of_analysis", "") == "training_run":
        return _format_training_card(candidate, gate_status)

    gs = gate_status or {}
    exposure_family = getattr(candidate, "exposure_family", "")
    outcome_family = getattr(candidate, "outcome_family", "")
    exposure_source = getattr(candidate, "exposure_source", "") or ""
    outcome_source = getattr(candidate, "outcome_source", "") or ""
    method_template = getattr(candidate, "method_template", "") or ""
    unit_of_analysis = getattr(candidate, "unit_of_analysis", "") or ""
    claim_strength = getattr(candidate, "claim_strength", "") or "associational"

    exposure_phrase = _humanize(exposure_family).title() + " Connectivity" \
        if exposure_family == "street_network" \
        else _humanize(exposure_family).title()
    outcome_phrase = _humanize(outcome_family).title()
    unit_phrase = _humanize_unit(unit_of_analysis)

    template = DISPLAY_TITLE_TEMPLATES.get(
        (exposure_family, outcome_family), _GENERIC_TITLE_TEMPLATE
    )
    display_title = template.format(
        exposure_phrase=exposure_phrase,
        outcome_phrase=outcome_phrase,
        unit=unit_phrase,
    )

    is_quasi_causal = claim_strength == "quasi_causal"
    if is_quasi_causal:
        research_question = (
            f"At the US {unit_phrase.lower()} level, what is the estimated "
            f"quasi-causal effect of {exposure_phrase.lower()} on "
            f"{outcome_phrase.lower()} outcomes under a "
            f"{_humanize(method_template).lower()} design?"
        )
    else:
        research_question = (
            f"At the US {unit_phrase.lower()} level, is {exposure_phrase.lower()} "
            f"associated with {outcome_phrase.lower()} outcomes after controlling "
            f"for sociodemographic confounders?"
        )

    if is_quasi_causal:
        rationale = (
            f"Treating {exposure_phrase.lower()} as a target-trial exposure at the "
            f"{unit_phrase.lower()} grain makes the X -> Y claim explicit: define "
            f"time zero, baseline covariates, exposure contrast, follow-up, and "
            f"diagnostics before estimating effects with public data."
        )
    else:
        rationale = (
            f"Linking {exposure_phrase.lower()} to {outcome_phrase.lower()} at the "
            f"{unit_phrase.lower()} grain offers a policy-actionable lens: small-area "
            f"variation in built-environment features is hypothesized to drive measurable "
            f"differences in population health, and both inputs are publicly available."
        )

    contribution_angle = (
        f"Combines {exposure_source or 'a public exposure dataset'} with "
        f"{outcome_source or 'a public outcome dataset'} at a finer grain than most "
        f"prior work, using a {_humanize(method_template).lower() or 'cross-sectional'} "
        f"design with a {claim_strength} claim."
    )
    method_screening = getattr(candidate, "method_screening", {}) or {}
    primary_reason = method_screening.get("primary_reason", "")
    if is_quasi_causal and primary_reason:
        contribution_angle = (
            f"Recommended {_humanize(method_template).lower()} because "
            f"{primary_reason}; other quasi-causal designs are retained in "
            f"method_screening for audit."
        )

    execution_summary = (
        f"Sources: {exposure_source or '(exposure source TBD)'} + "
        f"{outcome_source or '(outcome source TBD)'}. "
        f"Method: {_humanize(method_template).lower() or 'cross-sectional regression'}. "
        f"Grain: {unit_phrase.lower()}."
    )

    return {
        "display_title": display_title,
        "research_question": research_question,
        "rationale": rationale,
        "contribution_angle": contribution_angle,
        "execution_summary": execution_summary,
        # Surface the generic placeholder novelty status so UI can label it
        # consistently while Stage 2 has not yet run.
        "novelty_status": "pending_literature_check",
    }


def _to_card(
    c,
    title: str,
    rq: str,
    scores: dict | None = None,
    gate_status: dict | None = None,
    repair_history: list[dict] | None = None,
    pack_readiness: dict | None = None,
    readiness_summary: dict | None = None,
    display: dict | None = None,
) -> dict:
    gs = gate_status or {}
    pr = pack_readiness or {}
    rs = readiness_summary or {}
    return {
        "candidate_id": c.candidate_id,
        "title": title,
        "research_question": rq,
        # Layer 1 / Layer 2 display block — UI consumes this; raw contract
        # fields (exposure_source, method, …) are still present for the
        # folded "Layer 2" view and for backward compatibility.
        "display": display or {},
        "exposure_label": c.exposure_family,
        "exposure_source": c.exposure_source,
        "outcome_label": c.outcome_family,
        "outcome_source": c.outcome_source,
        "unit_of_analysis": c.unit_of_analysis,
        "method": c.method_template,
        "claim_strength": c.claim_strength,
        "method_screening": c.method_screening,
        "technology_tags": c.technology_tags,
        "required_secrets": gs.get("required_secrets", c.required_secrets),
        "automation_risk": c.automation_risk,
        "cloud_safe": c.cloud_safe,
        "scores": scores or {},
        "gate_status": gs,
        "repair_history": repair_history or [],
        "shortlist_status": gs.get("shortlist_status", "review"),
        "readiness_summary": rs,
        "readiness": rs.get("readiness", gs.get("shortlist_status", "review")),
        "user_visible_reasons": rs.get("user_visible_reasons", []),
        "debug_flags": rs.get("debug_flags", []),
        # Stage 1 never runs novelty checks — UI shows a neutral placeholder.
        "novelty_status": "pending_literature_check",
        # Stage 1 does NOT generate development packs.  Stage 2's
        # development_pack node populates these for the selected candidate.
        "development_pack_status": pr.get("development_pack_status", "not_generated"),
        "claude_code_ready": pr.get("claude_code_ready", False),
        "development_pack_files": pr.get("development_pack_files", []),
        "_raw": c.model_dump(),
    }


def _card_to_screening_entry(card: dict) -> dict:
    """Convert a ranked candidate card to a topic_screening.json entry.

    Called after rank_candidates() so rank is final.  Reads all data from the
    card dict (which embeds _raw from ComposedCandidate.model_dump()) so there
    is no need to keep a parallel screening list during the main loop.

    evaluation.user_visible_reasons contains only actionable, filtered reasons.
    Raw gate flags live in debug.gate_reasons only (never surfaced to UI directly).
    """
    raw = card.get("_raw") or {}
    gs = card.get("gate_status") or {}
    scores = card.get("scores") or {}
    rs = card.get("readiness_summary") or {}
    display = card.get("display") or {}
    exp = card.get("exposure_label", "").replace("_", " ")
    out = card.get("outcome_label", "").replace("_", " ")

    # Training-research candidates aren't geographic — geography is the model
    # release window, not a country. Detect via unit_of_analysis so the
    # screening entry doesn't claim US-only when it's really LLM training.
    is_training_research = card.get("unit_of_analysis") == "training_run"
    geography = "model_release" if is_training_research else "United States"

    if is_training_research:
        task_label = raw.get("outcome_task_label") or out.title()
        strategy_phrase = _STRATEGY_PHRASES.get(
            card.get("exposure_label", ""), exp
        )
        brief_rationale = (
            f"Test whether {strategy_phrase} of an open base LM improves "
            f"{out} on the task \"{task_label}\" against a zero-shot baseline."
        )
    elif card.get("claim_strength") == "quasi_causal":
        brief_rationale = (
            f"Estimate the quasi-causal effect of {exp} on {out} using "
            f"{card.get('method', '')}, grounded in "
            f"{card.get('exposure_source', '')} and {card.get('outcome_source', '')}."
        )
    else:
        brief_rationale = (
            f"Assess whether {exp} is empirically associated with {out} "
            f"using {card.get('exposure_source', '')} and {card.get('outcome_source', '')}."
        )

    return {
        "candidate_id": card["candidate_id"],
        "topic_id": card["candidate_id"],
        "title": card["title"],
        "display": display,
        "polished_title": card.get("polished_title"),
        "rerank": card.get("rerank", {}),
        "tech_lens_type": card.get("tech_lens_type"),
        "empirical_deepening_claim": card.get("empirical_deepening_claim"),
        "empirical_value_score": card.get("empirical_value_score"),
        "rank": card.get("rank", 0),
        "research_question": card["research_question"],
        "novelty_status": card.get("novelty_status", "pending_literature_check"),
        "exposure_variable": card.get("exposure_label", ""),
        "outcome_variable": card.get("outcome_label", ""),
        "geography": geography,
        "unit_of_analysis": card.get("unit_of_analysis", ""),
        "method": card.get("method", ""),
        "claim_strength": card.get("claim_strength", "associational"),
        "method_screening": raw.get("method_screening", card.get("method_screening", {})),
        "key_threats": raw.get("key_threats", []),
        "mitigations": raw.get("mitigations", {}),
        "declared_sources": [card.get("exposure_source", ""), card.get("outcome_source", "")],
        "exposure_source": card.get("exposure_source", ""),
        "outcome_source": card.get("outcome_source", ""),
        "exposure_family": card.get("exposure_label", ""),
        "outcome_family": card.get("outcome_label", ""),
        "outcome_task_id": raw.get("outcome_task_id"),
        "outcome_task_label": raw.get("outcome_task_label"),
        "outcome_task_description": raw.get("outcome_task_description"),
        "outcome_task_modality": raw.get("outcome_task_modality"),
        "outcome_task_dataset_hint": raw.get("outcome_task_dataset_hint"),
        "outcome_task_domain_input": raw.get("outcome_task_domain_input"),
        "join_plan": raw.get("join_plan", {}),
        "brief_rationale": brief_rationale,
        "technology_tags": card.get("technology_tags", []),
        "automation_risk": card.get("automation_risk", "medium"),
        "required_secrets": card.get("required_secrets", []),
        "gate_status": gs,
        "repair_history": card.get("repair_history", []),
        "shortlist_status": card.get("shortlist_status", "review"),
        "readiness": card.get("readiness", "needs_review"),
        "evaluation": {
            "overall_verdict": gs.get("overall", "pending"),
            "readiness": card.get("readiness", "needs_review"),
            "user_visible_reasons": card.get("user_visible_reasons", []),
            "score": round(float(scores.get("overall", 0.0)), 3),
            "rerank": card.get("rerank", {}),
            "empirical_deepening_claim": card.get("empirical_deepening_claim"),
        },
        "debug": {
            "gate_reasons": gs.get("reasons", []),
            "repair_history": card.get("repair_history", []),
        },
    }


_ELIGIBLE_READINESS = {"ready", "ready_after_auto_fix", "needs_review"}


def _select_shortlist(ranked_cards: list[dict], shortlist_size: int = 5) -> list[dict]:
    """Return top-k non-blocked candidates (legacy non-diverse helper)."""
    return [
        c for c in ranked_cards
        if c.get("readiness") in _ELIGIBLE_READINESS
    ][:shortlist_size]


def select_diverse_shortlist(
    ranked_cards: list[dict],
    shortlist_size: int = 5,
    max_per_exposure: int = 1,
    max_per_outcome: int = 2,
) -> list[dict]:
    """Greedy diverse shortlist selection.

    Walks ranked_cards in score-desc order and admits each candidate iff its
    exposure_family and outcome_family quotas are not yet full.  Blocked
    candidates are never eligible.  When the quota-respecting pool is smaller
    than ``shortlist_size``, falls back to fill the remaining slots from the
    same eligible pool ignoring quotas — better to surface near-duplicates
    than under-fill the picker.
    """
    eligible = [c for c in ranked_cards if c.get("readiness") in _ELIGIBLE_READINESS]
    if not eligible:
        return []

    selected: list[dict] = []
    selected_ids: set[str] = set()
    exposure_count: dict[str, int] = {}
    outcome_count: dict[str, int] = {}

    for card in eligible:
        if len(selected) >= shortlist_size:
            break
        exp = str(card.get("exposure_label") or "")
        # For training_research, outcome diversity must track the domain task
        # (outcome_task_id), not the metric_family (which only has 3 values).
        raw = card.get("_raw") or {}
        task_id = raw.get("outcome_task_id")
        out = str(task_id or card.get("outcome_label") or "")
        if exposure_count.get(exp, 0) >= max_per_exposure:
            continue
        if outcome_count.get(out, 0) >= max_per_outcome:
            continue
        selected.append(card)
        selected_ids.add(str(card.get("candidate_id")))
        exposure_count[exp] = exposure_count.get(exp, 0) + 1
        outcome_count[out] = outcome_count.get(out, 0) + 1

    if len(selected) < shortlist_size:
        # Fill remaining slots ignoring quotas (still excluding blocked).
        for card in eligible:
            if len(selected) >= shortlist_size:
                break
            cid = str(card.get("candidate_id"))
            if cid in selected_ids:
                continue
            selected.append(card)
            selected_ids.add(cid)

    return selected


def _enabled_technologies(state: dict, extra: list[str] | None = None) -> list[str]:
    enabled = [
        key for key, value in (state.get("technology_options") or {}).items()
        if bool(value)
    ]
    for item in extra or []:
        if item not in enabled:
            enabled.append(item)
    return enabled


def _compose_request(
    state: dict,
    *,
    template_id: str,
    max_candidates: int,
    enable_experimental: bool | None = None,
    preferred_technology: list[str] | None = None,
    automation_risk_tolerance: str | None = None,
) -> ComposeRequest:
    return ComposeRequest(
        template_id=template_id,
        domain_input=state["domain_input"],
        max_candidates=max_candidates,
        enable_tier2=(state.get("technology_options") or {}).get("remote_sensing", True),
        enable_experimental=(
            state.get("enable_experimental", False)
            if enable_experimental is None
            else enable_experimental
        ),
        no_paid_api=(state.get("cloud_constraints") or {}).get("no_paid_api", True),
        no_manual_download=(state.get("cloud_constraints") or {}).get("no_manual_download", True),
        preferred_technology=preferred_technology
        if preferred_technology is not None
        else _enabled_technologies(state),
        automation_risk_tolerance=automation_risk_tolerance
        or state.get("automation_risk_tolerance", "low_medium"),
    )


def _load_template_dict(template_id: str) -> dict:
    try:
        return load_research_template(template_id)
    except FileNotFoundError:
        return {}


def _process_composed_candidates(
    candidates: list,
    *,
    req: ComposeRequest,
    template_dict: dict,
    state: dict,
) -> tuple[list[dict], list[dict]]:
    """Run the shared precheck/repair/scoring pipeline for composed candidates."""
    is_training_research = template_dict.get("kind") == "training_research"
    runtime_tier = state.get("runtime_tier", "colab_t4")

    cards: list[dict] = []
    all_repair_histories: list[dict] = []

    for c in candidates:
        # Step 4: normalize / enrich metadata (unconditional pre-processing).
        c = ensure_identification_metadata(c)
        # Step 5: deterministic precheck.
        gate_status = precheck_candidate(c)
        # Step 6: deterministic repair; all auto-fix events go to repair_history only.
        repaired_c, gate_status, repair_history = repair_candidate(c, gate_status)
        all_repair_histories.extend(repair_history)
        # Step 7: strict export validation.
        gate_status = validate_candidate_export_contract(
            repaired_c,
            gate_status,
            no_paid_api=req.no_paid_api,
        )

        # Step 7b: training-research feasibility (license / GPU / leakage plan).
        # Subchecks are merged into gate_status. License or leakage failures
        # downgrade overall to fail; GPU warnings downgrade pass→warning.
        # Per the approved plan, GPU shortfall never hard-blocks.
        if is_training_research:
            train_fb = evaluate_training_candidate(
                repaired_c, template_dict, runtime_tier=runtime_tier
            )
            existing_subchecks = dict(gate_status.get("subchecks") or {})
            existing_subchecks.update(train_fb["subchecks"])
            gate_status["subchecks"] = existing_subchecks
            gate_status.setdefault("reasons", []).extend(train_fb["reasons"])

            cur_overall = gate_status.get("overall", "pass")
            if train_fb["overall"] == "fail":
                gate_status["overall"] = "fail"
                gate_status["passed"] = False
                gate_status["shortlist_status"] = "blocked"
            elif train_fb["overall"] == "warning" and cur_overall == "pass":
                gate_status["overall"] = "warning"
                gate_status["shortlist_status"] = "review"
        # ── Contract-first: title and research question are derived ONLY from
        # the structured fields after the contract has been validated, so
        # candidates that fail the contract are not given a polished title
        # that masks data-availability problems.
        title = _make_title(repaired_c)
        rq = _make_research_question(repaired_c)
        # Step 8: single translation layer — raw gate flags → user-facing readiness.
        readiness_summary = compute_candidate_readiness(
            repaired_c.model_dump(), gate_status, repair_history
        )
        # Step 9: weighted score.
        scores = score_candidate(repaired_c.model_dump(), gate_status, repair_history)

        # Stage 1 placeholder for pack readiness — every card starts as
        # not_generated.  Stage 2 generates a real pack only for the
        # candidate the user picks.
        pack_readiness = evaluate_development_pack_readiness(
            repaired_c.model_dump(), gate_status, pack_dir=None
        )

        # Layer 1 / Layer 2 display formatting (research-question style title
        # + folded execution_summary).  Pure function, no I/O.
        display = format_display_card(repaired_c, gate_status)

        cards.append(
            _to_card(
                repaired_c, title, rq, scores, gate_status,
                repair_history, pack_readiness, readiness_summary,
                display=display,
            )
        )

    return cards, all_repair_histories


def _load_field_scan_summary(state: dict) -> dict:
    field_scan_summary = {}
    field_scan_path = state.get("field_scan_path") or settings.field_scan_path()
    try:
        with open(field_scan_path, "r", encoding="utf-8") as fh:
            field_scan_summary = json.load(fh)
    except Exception:
        field_scan_summary = {}
    return field_scan_summary


def _should_probe_speculative_template(state: dict, template_id: str) -> bool:
    if "speculative_candidates_enabled" in state:
        return bool(state.get("speculative_candidates_enabled"))
    # Default: stable built-environment runs get a parallel high-risk idea lane.
    return template_id == "built_environment_health"


def _select_speculative_cards(ranked_cards: list[dict], limit: int) -> list[dict]:
    experimental_cards = [
        c for c in ranked_cards
        if (
            "experimental" in set(c.get("technology_tags") or [])
            or "streetview_cv" in set(c.get("technology_tags") or [])
            or c.get("automation_risk") == "high"
            or c.get("readiness") == "blocked"
        )
    ]
    return experimental_cards[:limit]


def _speculative_unlocks(card: dict) -> list[str]:
    unlocks: list[str] = []
    if "experimental" in set(card.get("technology_tags") or []):
        unlocks.append("Enable explicit experimental-source review.")
    if card.get("required_secrets"):
        unlocks.append(
            "Provide required API secrets: "
            + ", ".join(str(s) for s in card.get("required_secrets") or [])
        )
    if card.get("automation_risk") == "high":
        unlocks.append("Accept a manual-review execution path for high automation risk.")
    if not bool(card.get("cloud_safe", True)):
        unlocks.append("Confirm provider policy and raw-image caching constraints.")
    return unlocks


def _card_to_speculative_entry(card: dict, rank: int) -> dict:
    entry = _card_to_screening_entry(card)
    entry["rank"] = rank
    entry["speculative_status"] = "review_only"
    entry["selectable"] = False
    entry["why_speculative"] = (
        "High-novelty technology candidate kept outside the safe shortlist "
        "because it requires explicit human review before execution."
    )
    entry["unlock_requirements"] = _speculative_unlocks(card)
    return entry


def _prefix_speculative_candidate_ids(cards: list[dict], namespace: str) -> list[dict]:
    prefixed: list[dict] = []
    safe_namespace = namespace.replace("_experimental", "").replace("built_environment_health", "streetview")
    for card in cards:
        new_card = dict(card)
        old_id = str(new_card.get("candidate_id", "candidate"))
        new_id = f"spec_{safe_namespace}_{old_id}"
        new_card["candidate_id"] = new_id
        raw = dict(new_card.get("_raw") or {})
        if raw:
            raw["candidate_id"] = new_id
            new_card["_raw"] = raw
        prefixed.append(new_card)
    return prefixed


def _generate_speculative_cards(
    state: dict,
    *,
    field_scan_summary: dict,
    limit: int,
) -> list[dict]:
    template_id = "built_environment_health_experimental"
    req = _compose_request(
        state,
        template_id=template_id,
        max_candidates=state.get("speculative_max_candidates", 20),
        enable_experimental=True,
        preferred_technology=_enabled_technologies(
            state, ["streetview_cv", "vision", "deep_learning"]
        ),
        automation_risk_tolerance="experimental",
    )
    candidates = compose_candidates(req)
    if not candidates:
        return []
    template_dict = _load_template_dict(template_id)
    cards, _ = _process_composed_candidates(
        candidates, req=req, template_dict=template_dict, state=state
    )
    ranked = rerank_candidates(cards, state["domain_input"], field_scan_summary)
    selected = _select_speculative_cards(ranked, limit)
    return _prefix_speculative_candidate_ids(selected, template_id)


def _write_speculative_candidates(
    *,
    run_id: str,
    domain_input: str,
    template_id: str,
    entries: list[dict],
) -> str:
    payload = {
        "run_id": run_id,
        "status": "review_only",
        "domain_input": domain_input,
        "source_template_id": template_id,
        "candidate_count": len(entries),
        "candidates": entries,
    }
    path = settings.speculative_candidates_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote speculative_candidates.json — %d candidates", len(entries))
    return path


def run_candidate_factory_ideation(state: dict) -> dict:
    """Stage 1 of the two-stage candidate flow.

    Pipeline:
      compose_candidates → normalize/enrich → precheck → repair → export_validate
      → compute_readiness → score → rank full pool → diverse-select shortlist
      → format display cards → write outputs

    Stage 1 deliberately does NOT run novelty checks (cards carry
    novelty_status="pending_literature_check") and does NOT generate
    development packs.  Both are deferred to Stage 2 nodes that run after
    the user picks one candidate at the HITL checkpoint.

    candidate_cards.json         — safe/full production pool, ranked
    topic_screening.json         — executable shortlist + speculative side lane
    speculative_candidates.json  — review-only high-risk/new-technology ideas

    Returns file-path references for downstream pipeline nodes.
    """
    template_id = state["template_id"]
    domain_input = state["domain_input"]
    # max_candidates controls the internal generation pool size, NOT the UI display count.
    max_candidates = state.get("max_candidates", 40)
    # shortlist_size controls how many candidates appear in topic_screening.json / UI.
    shortlist_size = state.get("shortlist_size", 5)
    speculative_size = state.get("speculative_size", 5)

    logger.info(
        "Candidate factory ideation — template: %s  pool: %d  shortlist: %d",
        template_id, max_candidates, shortlist_size,
    )

    req = _compose_request(state, template_id=template_id, max_candidates=max_candidates)
    candidates = compose_candidates(req)

    if not candidates:
        logger.error(
            "compose_candidates() returned no candidates for template %s", template_id
        )
        return {
            "execution_status": "failed",
            "degraded_nodes": ["ideation:no_candidates_from_factory"],
        }

    template_dict = _load_template_dict(template_id)
    cards, all_repair_histories = _process_composed_candidates(
        candidates, req=req, template_dict=template_dict, state=state
    )

    field_scan_summary = _load_field_scan_summary(state)
    run_id = settings.current_run_scope() or state.get("run_id") or "unknown"

    # Rerank full pool with domain-fit + research-value signals. This keeps
    # deterministic feasibility checks intact while improving the user-visible
    # shortlist for the requested domain.
    ranked_cards = rerank_candidates(cards, domain_input, field_scan_summary)

    # Diverse shortlist selection: max 2 per exposure_family, max 2 per
    # outcome_family.  Blocked candidates are never eligible — they live in
    # candidate_cards.json as diagnostics only.  No development pack
    # generation in Stage 1.
    shortlist_cards = select_diverse_shortlist(ranked_cards, shortlist_size)
    speculative_cards = _select_speculative_cards(ranked_cards, speculative_size)
    if _should_probe_speculative_template(state, template_id):
        speculative_cards.extend(
            _generate_speculative_cards(
                state,
                field_scan_summary=field_scan_summary,
                limit=speculative_size,
            )
        )
    speculative_cards = speculative_cards[:speculative_size]

    # Step 12a: write candidate_cards.json — full ranked pool (debug/QA use).
    cards_path = settings.output_dir() / "candidate_cards.json"
    cards_path.write_text(
        json.dumps(ranked_cards, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %d candidate cards → %s", len(ranked_cards), cards_path)

    write_feasibility_report(run_id, ranked_cards)
    write_development_pack_index(run_id, ranked_cards)
    write_gate_trace(run_id, ranked_cards)

    # Step 12b: write topic_screening.json — shortlist only (user-visible).
    # Convert each shortlist card to the screening entry format after ranking so
    # rank numbers are final.  Raw gate flags are kept in debug block only.
    shortlist_entries = [_card_to_screening_entry(c) for c in shortlist_cards]
    speculative_entries = [
        _card_to_speculative_entry(c, i)
        for i, c in enumerate(speculative_cards, start=1)
    ]
    speculative_path = _write_speculative_candidates(
        run_id=run_id,
        domain_input=domain_input,
        template_id="built_environment_health_experimental",
        entries=speculative_entries,
    )
    screening = {
        "run_id": run_id,
        "status": "pending_review",
        "ideation_mode": "candidate_factory",
        "template_id": template_id,
        "shortlist_size": len(shortlist_entries),
        "pool_size": len(ranked_cards),
        "speculative_size": len(speculative_entries),
        "candidates": shortlist_entries,
        "speculative_candidates": speculative_entries,
    }
    screening_path = settings.topic_screening_path()
    with open(screening_path, "w", encoding="utf-8") as fh:
        json.dump(screening, fh, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote topic_screening.json — shortlist %d of %d candidates; speculative %d",
        len(shortlist_entries), len(ranked_cards), len(speculative_entries),
    )

    repair_history_path = settings.repair_history_path()
    with open(repair_history_path, "w", encoding="utf-8") as fh:
        json.dump(all_repair_histories, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote repair_history.json — %d entries", len(all_repair_histories))

    first = shortlist_entries[0] if shortlist_entries else None
    minimal_plan = {
        "run_id": run_id,
        "project_title": first["title"] if first else f"{domain_input}: no executable candidate",
        "research_question": first["research_question"] if first else "",
        "template_id": template_id,
        "candidates": shortlist_entries,
        "speculative_candidates": speculative_entries,
    }
    plan_path = settings.research_plan_path()
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(minimal_plan, fh, indent=2, ensure_ascii=False)

    return {
        "candidate_topics_path": str(screening_path),
        "current_plan_path": str(plan_path),
        "candidate_cards_path": str(cards_path),
        "speculative_candidates_path": speculative_path,
        "repair_history_path": str(repair_history_path),
        "feasibility_report_path": str(settings.output_dir() / "feasibility_report.json"),
        "development_pack_index_path": str(settings.output_dir() / "development_pack_index.json"),
        "gate_trace_path": str(settings.output_dir() / "gate_trace.json"),
        "execution_status": "ideation_complete",
    }
