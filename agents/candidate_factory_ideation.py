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
) -> dict:
    gs = gate_status or {}
    pr = pack_readiness or {}
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
        "development_pack_status": pr.get("development_pack_status", "not_generated"),
        "claude_code_ready": pr.get("claude_code_ready", False),
        "development_pack_files": pr.get("development_pack_files", []),
        "_raw": c.model_dump(),
    }


def _to_screening_candidate(
    c,
    title: str,
    rq: str,
    rank: int,
    gate_status: dict | None = None,
    repair_history: list[dict] | None = None,
) -> dict:
    """Format a candidate for topic_screening.json.

    Includes all fields that apply_idea_selection_by_candidate_id() and
    build_research_plan_from_candidate() expect, plus gate_status / repair_history
    from the precheck + repair pipeline.
    """
    exp = c.exposure_family.replace("_", " ")
    out = c.outcome_family.replace("_", " ")
    gs = gate_status or {}
    return {
        "candidate_id": c.candidate_id,
        "topic_id": c.candidate_id,
        "title": title,
        "rank": rank,
        "research_question": rq,
        "exposure_variable": c.exposure_family,
        "outcome_variable": c.outcome_family,
        "geography": "United States",
        "unit_of_analysis": c.unit_of_analysis,
        "method": c.method_template,
        "key_threats": c.key_threats,
        "mitigations": c.mitigations,
        "declared_sources": [c.exposure_source, c.outcome_source],
        "exposure_source": c.exposure_source,
        "outcome_source": c.outcome_source,
        "exposure_family": c.exposure_family,
        "outcome_family": c.outcome_family,
        "join_plan": c.join_plan,
        "brief_rationale": (
            f"Assess whether {exp} is empirically associated with {out} "
            f"using {c.exposure_source} and {c.outcome_source}."
        ),
        "technology_tags": c.technology_tags,
        "automation_risk": c.automation_risk,
        "required_secrets": gs.get("required_secrets", []),
        "gate_status": gs,
        "repair_history": repair_history or [],
        "shortlist_status": gs.get("shortlist_status", "review"),
        "evaluation": {
            "overall_verdict": gs.get("overall", "pending"),
            "reasons": gs.get("reasons", []),
        },
    }


def run_candidate_factory_ideation(state: dict) -> dict:
    """Write candidate_cards.json, topic_screening.json, and repair_history.json.

    Pipeline:
      compose_candidates → precheck_candidate → repair_candidate → write outputs

    Returns file-path references for downstream pipeline nodes.
    """
    template_id = state["template_id"]
    domain_input = state["domain_input"]

    logger.info("Candidate factory ideation — template: %s", template_id)

    req = ComposeRequest(
        template_id=template_id,
        domain_input=domain_input,
        max_candidates=state.get("max_candidates", 20),
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

    cards: list[dict] = []
    screening_candidates: list[dict] = []
    all_repair_histories: list[dict] = []

    run_id = settings.current_run_scope() or "unknown"

    for rank, c in enumerate(candidates, start=1):
        title = _make_title(c)
        rq = _make_research_question(c)

        c = ensure_identification_metadata(c)
        gate_status = precheck_candidate(c)
        repaired_c, gate_status, repair_history = repair_candidate(c, gate_status)
        all_repair_histories.extend(repair_history)
        gate_status = validate_candidate_export_contract(
            repaired_c,
            gate_status,
            no_paid_api=req.no_paid_api,
        )

        scores = score_candidate(repaired_c.model_dump(), gate_status, repair_history)

        # Auto-generate development pack for ready candidates only.
        shortlist = gate_status.get("shortlist_status", "blocked")
        pack_dir = None
        if shortlist == "ready":
            try:
                pack_dir = write_development_pack(run_id, repaired_c.model_dump())
            except Exception as exc:
                logger.warning(
                    "development pack generation failed for %s: %s",
                    repaired_c.candidate_id, exc
                )

        pack_readiness = evaluate_development_pack_readiness(
            repaired_c.model_dump(), gate_status, pack_dir
        )

        cards.append(
            _to_card(repaired_c, title, rq, scores, gate_status, repair_history, pack_readiness)
        )
        screening_candidates.append(
            _to_screening_candidate(repaired_c, title, rq, rank, gate_status, repair_history)
        )

    cards = rank_candidates(cards)

    cards_path = settings.output_dir() / "candidate_cards.json"
    cards_path.write_text(json.dumps(cards, indent=2, ensure_ascii=False))
    logger.info("Wrote %d candidate cards → %s", len(cards), cards_path)

    write_feasibility_report(run_id, cards)
    write_development_pack_index(run_id, cards)
    write_gate_trace(run_id, cards)

    screening = {
        "run_id": run_id,
        "status": "pending_review",
        "ideation_mode": "candidate_factory",
        "template_id": template_id,
        "candidates": screening_candidates,
    }
    screening_path = settings.topic_screening_path()
    with open(screening_path, "w", encoding="utf-8") as fh:
        json.dump(screening, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote topic_screening.json — %d candidates", len(screening_candidates))

    repair_history_path = settings.repair_history_path()
    with open(repair_history_path, "w", encoding="utf-8") as fh:
        json.dump(all_repair_histories, fh, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote repair_history.json — %d entries", len(all_repair_histories)
    )

    first = screening_candidates[0]
    minimal_plan = {
        "run_id": run_id,
        "project_title": first["title"],
        "research_question": first["research_question"],
        "template_id": template_id,
        "candidates": screening_candidates,
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
