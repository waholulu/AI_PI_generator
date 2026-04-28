"""Deterministic candidate factory ideation path.

Called by ideation_node() when candidate_factory_enabled=True (i.e. a
template_id was provided in the run request). Replaces LLM seed generation
with a deterministic Cartesian-product expansion via compose_candidates().

Step 3 upgrade: after precheck, each candidate is passed through
repair_candidate() to apply deterministic fixes. Repair history is embedded
in both candidate_cards.json and topic_screening.json, and the full
per-candidate history is written to output/repair_history.json.
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
from agents.identification_template_filler import ensure_identification_metadata
from agents.development_pack_status import evaluate_development_pack_readiness
from agents.development_pack_writer import write_development_pack
from agents.final_ranker import rank_candidates, score_candidate
from agents.logging_config import get_logger
from models.candidate_composer_schema import ComposeRequest

logger = get_logger(__name__)


def _make_title(c) -> str:
    exp = c.exposure_family.replace("_", " ").title()
    out = c.outcome_family.replace("_", " ").title()
    return f"{exp} and {out}"


def _make_research_question(c) -> str:
    exp = c.exposure_family.replace("_", " ")
    out = c.outcome_family.replace("_", " ")
    return f"How does {exp} affect {out} at the {c.unit_of_analysis} level?"


def _to_card(
    c,
    title: str,
    rq: str,
    scores: dict | None = None,
    gate_status: dict | None = None,
    repair_history: list[dict] | None = None,
    pack_readiness: dict | None = None,
    readiness_summary: dict | None = None,
) -> dict:
    gs = gate_status or {}
    pr = pack_readiness or {}
    rs = readiness_summary or {}
    return {
        "candidate_id": c.candidate_id,
        "title": title,
        "research_question": rq,
        "exposure_label": c.exposure_family,
        "exposure_source": c.exposure_source,
        "outcome_label": c.outcome_family,
        "outcome_source": c.outcome_source,
        "unit_of_analysis": c.unit_of_analysis,
        "method": c.method_template,
        "claim_strength": c.claim_strength,
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
    exp = card.get("exposure_label", "").replace("_", " ")
    out = card.get("outcome_label", "").replace("_", " ")
    return {
        "candidate_id": card["candidate_id"],
        "topic_id": card["candidate_id"],
        "title": card["title"],
        "rank": card.get("rank", 0),
        "research_question": card["research_question"],
        "exposure_variable": card.get("exposure_label", ""),
        "outcome_variable": card.get("outcome_label", ""),
        "geography": "United States",
        "unit_of_analysis": card.get("unit_of_analysis", ""),
        "method": card.get("method", ""),
        "key_threats": raw.get("key_threats", []),
        "mitigations": raw.get("mitigations", {}),
        "declared_sources": [card.get("exposure_source", ""), card.get("outcome_source", "")],
        "exposure_source": card.get("exposure_source", ""),
        "outcome_source": card.get("outcome_source", ""),
        "exposure_family": card.get("exposure_label", ""),
        "outcome_family": card.get("outcome_label", ""),
        "join_plan": raw.get("join_plan", {}),
        "brief_rationale": (
            f"Assess whether {exp} is empirically associated with {out} "
            f"using {card.get('exposure_source', '')} and {card.get('outcome_source', '')}."
        ),
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
        },
        "debug": {
            "gate_reasons": gs.get("reasons", []),
            "repair_history": card.get("repair_history", []),
        },
    }


def _select_shortlist(ranked_cards: list[dict], shortlist_size: int = 5) -> list[dict]:
    """Return top-k non-blocked candidates; falls back to top-k overall if needed."""
    shortlist = [
        c for c in ranked_cards
        if c.get("readiness") in {"ready", "ready_after_auto_fix", "needs_review"}
    ][:shortlist_size]
    if len(shortlist) < shortlist_size:
        shortlist = ranked_cards[:shortlist_size]
    return shortlist


def run_candidate_factory_ideation(state: dict) -> dict:
    """Write candidate_cards.json (full ranked pool) and topic_screening.json (shortlist).

    Pipeline (13 steps per plan §3):
      compose_candidates → normalize/enrich → precheck → repair → export_validate
      → compute_readiness → score → rank full pool → select shortlist → write outputs
      → generate development packs for ready candidates

    candidate_cards.json  — full pool (max_candidates entries), ranked
    topic_screening.json  — shortlist only (shortlist_size entries, default 5)

    Returns file-path references for downstream pipeline nodes.
    """
    template_id = state["template_id"]
    domain_input = state["domain_input"]
    # max_candidates controls the internal generation pool size, NOT the UI display count.
    max_candidates = state.get("max_candidates", 40)
    # shortlist_size controls how many candidates appear in topic_screening.json / UI.
    shortlist_size = state.get("shortlist_size", 5)

    logger.info(
        "Candidate factory ideation — template: %s  pool: %d  shortlist: %d",
        template_id, max_candidates, shortlist_size,
    )

    req = ComposeRequest(
        template_id=template_id,
        domain_input=domain_input,
        max_candidates=max_candidates,
        enable_tier2=(state.get("technology_options") or {}).get("remote_sensing", True),
        enable_experimental=state.get("enable_experimental", False),
        no_paid_api=(state.get("cloud_constraints") or {}).get("no_paid_api", True),
        no_manual_download=(state.get("cloud_constraints") or {}).get("no_manual_download", True),
        preferred_technology=[
            key for key, enabled in (state.get("technology_options") or {}).items() if bool(enabled)
        ],
        automation_risk_tolerance=state.get("automation_risk_tolerance", "low_medium"),
    )
    candidates = compose_candidates(req)

    if not candidates:
        logger.error(
            "compose_candidates() returned no candidates for template %s", template_id
        )
        return {
            "execution_status": "failed",
            "degraded_nodes": ["ideation:no_candidates_from_factory"],
        }

    # Steps 4–9: per-candidate processing (no rank assigned yet — ranking happens after).
    cards: list[dict] = []
    all_repair_histories: list[dict] = []
    run_id = settings.current_run_scope() or "unknown"

    for c in candidates:
        title = _make_title(c)
        rq = _make_research_question(c)

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
        # Step 8: single translation layer — raw gate flags → user-facing readiness.
        readiness_summary = compute_candidate_readiness(
            repaired_c.model_dump(), gate_status, repair_history
        )
        # Step 9: weighted score.
        scores = score_candidate(repaired_c.model_dump(), gate_status, repair_history)

        # Development packs are generated later (Step 13) after ranking, but we need
        # pack_readiness to embed in each card.  Pass pack_dir=None here; packs are
        # created in the post-ranking loop only for shortlisted ready candidates.
        pack_readiness = evaluate_development_pack_readiness(
            repaired_c.model_dump(), gate_status, pack_dir=None
        )

        cards.append(
            _to_card(
                repaired_c, title, rq, scores, gate_status,
                repair_history, pack_readiness, readiness_summary,
            )
        )

    # Step 10: rank full pool (readiness → risk → score).
    ranked_cards = rank_candidates(cards)

    # Step 13: generate development packs for ready shortlist candidates.
    shortlist_cards = _select_shortlist(ranked_cards, shortlist_size)
    shortlist_ids = {c["candidate_id"] for c in shortlist_cards}
    for card in ranked_cards:
        if card["candidate_id"] not in shortlist_ids:
            continue
        if card.get("readiness") not in {"ready", "ready_after_auto_fix"}:
            continue
        raw = card.get("_raw") or {}
        try:
            pack_dir = write_development_pack(run_id, raw)
            new_pack_readiness = evaluate_development_pack_readiness(raw, card.get("gate_status") or {}, pack_dir)
            card["development_pack_status"] = new_pack_readiness.get("development_pack_status", card["development_pack_status"])
            card["claude_code_ready"] = new_pack_readiness.get("claude_code_ready", card["claude_code_ready"])
            card["development_pack_files"] = new_pack_readiness.get("development_pack_files", card["development_pack_files"])
        except Exception as exc:
            logger.warning("development pack generation failed for %s: %s", card["candidate_id"], exc)

    # Step 12a: write candidate_cards.json — full ranked pool (debug/QA use).
    cards_path = settings.output_dir() / "candidate_cards.json"
    cards_path.write_text(json.dumps(ranked_cards, indent=2, ensure_ascii=False))
    logger.info("Wrote %d candidate cards → %s", len(ranked_cards), cards_path)

    write_feasibility_report(run_id, ranked_cards)
    write_development_pack_index(run_id, ranked_cards)
    write_gate_trace(run_id, ranked_cards)

    # Step 12b: write topic_screening.json — shortlist only (user-visible).
    # Convert each shortlist card to the screening entry format after ranking so
    # rank numbers are final.  Raw gate flags are kept in debug block only.
    shortlist_entries = [_card_to_screening_entry(c) for c in shortlist_cards]
    screening = {
        "run_id": run_id,
        "status": "pending_review",
        "ideation_mode": "candidate_factory",
        "template_id": template_id,
        "shortlist_size": len(shortlist_entries),
        "pool_size": len(ranked_cards),
        "candidates": shortlist_entries,
    }
    screening_path = settings.topic_screening_path()
    with open(screening_path, "w", encoding="utf-8") as fh:
        json.dump(screening, fh, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote topic_screening.json — shortlist %d of %d candidates",
        len(shortlist_entries), len(ranked_cards),
    )

    repair_history_path = settings.repair_history_path()
    with open(repair_history_path, "w", encoding="utf-8") as fh:
        json.dump(all_repair_histories, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote repair_history.json — %d entries", len(all_repair_histories))

    first = shortlist_entries[0]
    minimal_plan = {
        "run_id": run_id,
        "project_title": first["title"],
        "research_question": first["research_question"],
        "template_id": template_id,
        "candidates": shortlist_entries,
    }
    plan_path = settings.research_plan_path()
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump(minimal_plan, fh, indent=2, ensure_ascii=False)

    return {
        "candidate_topics_path": str(screening_path),
        "current_plan_path": str(plan_path),
        "candidate_cards_path": str(cards_path),
        "repair_history_path": str(repair_history_path),
        "feasibility_report_path": str(settings.output_dir() / "feasibility_report.json"),
        "development_pack_index_path": str(settings.output_dir() / "development_pack_index.json"),
        "gate_trace_path": str(settings.output_dir() / "gate_trace.json"),
        "execution_status": "ideation_complete",
    }
