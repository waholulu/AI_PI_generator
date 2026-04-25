"""Deterministic candidate factory ideation path.

Called by ideation_node() when candidate_factory_enabled=True (i.e. a
template_id was provided in the run request). Replaces LLM seed generation
with a deterministic Cartesian-product expansion via compose_candidates().
"""
from __future__ import annotations

import json

from agents import settings
from agents.candidate_composer import compose_candidates
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


def _to_card(c, title: str, rq: str) -> dict:
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
        "claim_strength": "associational",
        "technology_tags": c.technology_tags,
        "required_secrets": [],
        "automation_risk": c.automation_risk,
        "scores": {},
        "gate_status": {},
        "repair_history": [],
        "development_pack_status": "not_generated",
        "_raw": c.model_dump(),
    }


def _to_screening_candidate(c, title: str, rq: str, rank: int) -> dict:
    """Format a candidate for topic_screening.json.

    Includes all fields that apply_idea_selection_by_candidate_id() and
    build_research_plan_from_candidate() expect.
    """
    exp = c.exposure_family.replace("_", " ")
    out = c.outcome_family.replace("_", " ")
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
        "brief_rationale": (
            f"Assess whether {exp} is empirically associated with {out} "
            f"using {c.exposure_source} and {c.outcome_source}."
        ),
        "technology_tags": c.technology_tags,
        "automation_risk": c.automation_risk,
    }


def run_candidate_factory_ideation(state: dict) -> dict:
    """Write candidate_cards.json and topic_screening.json from compose_candidates().

    Returns file-path references for downstream pipeline nodes.
    """
    template_id = state["template_id"]
    domain_input = state["domain_input"]

    logger.info("Candidate factory ideation — template: %s", template_id)

    req = ComposeRequest(
        template_id=template_id,
        domain_input=domain_input,
        max_candidates=state.get("max_candidates", 20),
        enable_experimental=state.get("enable_experimental", False),
    )
    candidates = compose_candidates(req)

    if not candidates:
        logger.error("compose_candidates() returned no candidates for template %s", template_id)
        return {
            "execution_status": "failed",
            "degraded_nodes": ["ideation:no_candidates_from_factory"],
        }

    cards: list[dict] = []
    screening_candidates: list[dict] = []
    for rank, c in enumerate(candidates, start=1):
        title = _make_title(c)
        rq = _make_research_question(c)
        cards.append(_to_card(c, title, rq))
        screening_candidates.append(_to_screening_candidate(c, title, rq, rank))

    cards_path = settings.output_dir() / "candidate_cards.json"
    cards_path.write_text(json.dumps(cards, indent=2, ensure_ascii=False))
    logger.info("Wrote %d candidate cards → %s", len(cards), cards_path)

    run_id = settings.current_run_scope() or "unknown"
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
        "execution_status": "ideation_complete",
    }
