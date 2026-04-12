"""
Shared helpers for the HITL (human-in-the-loop) checkpoint.

Used by CLI (main.py), API (api/server.py), and Streamlit (ui/app.py) to:
- Load and display validated topics
- Record user-rejected topics in memory + graveyard
- Re-run ideation + idea_validator outside the LangGraph for topic regeneration
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents import settings
from agents.logging_config import get_logger
from agents.memory_retriever import MemoryRetriever

logger = get_logger(__name__)

MAX_REGENERATION_ROUNDS = int(os.getenv("HITL_MAX_REGENERATION_ROUNDS", "3"))


def load_validated_topics(validation_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load validated ideas from idea_validation.json.

    Returns list of dicts with title, brief_rationale, rank, overall_verdict, etc.
    Returns empty list if file is missing or unreadable.
    """
    path = validation_path or settings.idea_validation_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        return report.get("validated_ideas", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load validation report from %s: %s", path, exc)
        return []


def record_rejected_topics(
    topics: List[Dict[str, Any]],
    domain: str,
    round_num: int,
) -> None:
    """Store all current topics as 'rejected_by_user' in CSV memory and graveyard.

    Called when the user chooses to regenerate topics at the HITL checkpoint.
    This ensures the ideation agent's memory context includes these rejected
    topics, preventing regeneration of the same ideas.
    """
    memory = MemoryRetriever()
    screening_path = settings.topic_screening_path()

    # 1. Record each topic in the CSV memory
    for topic in topics:
        title = topic.get("title", "")
        if not title:
            continue
        try:
            memory.store_idea(
                topic=title,
                domain=domain,
                status="rejected_by_user",
                rejection_reason=f"User rejected at HITL checkpoint (round {round_num})",
                metadata={
                    "score": topic.get("final_score", topic.get("initial_score")),
                    "round": round_num,
                },
                source_file=screening_path,
            )
        except Exception as exc:
            logger.warning("Failed to store rejected topic '%s' in memory: %s", title[:40], exc)

    # 2. Append to domain-scoped graveyard JSON
    graveyard_path = settings.ideas_graveyard_path(domain=domain)
    existing: list = []
    if os.path.exists(graveyard_path):
        try:
            with open(graveyard_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []

    for topic in topics:
        title = topic.get("title", "")
        if not title:
            continue
        existing.append({
            "title": title,
            "rejection_reason": f"rejected_by_user (round {round_num})",
            "brief_rationale": topic.get("brief_rationale", ""),
            "score": topic.get("final_score", topic.get("initial_score")),
            "rejected_at": datetime.now(timezone.utc).isoformat(),
        })

    os.makedirs(os.path.dirname(graveyard_path), exist_ok=True)
    with open(graveyard_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info(
        "Recorded %d rejected topics in memory and graveyard (round %d)",
        len(topics), round_num,
    )


def regenerate_topics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run ideation + idea_validator outside the LangGraph.

    Expects *state* to contain at minimum ``domain_input`` and
    ``field_scan_path``.  The agents write updated files to disk
    (topic_screening.json, idea_validation.json, research_plan.json,
    research_context.json) so that subsequent graph nodes (literature,
    drafter, etc.) pick up the new content.

    Returns the merged state dict with updated file paths.
    """
    from agents.ideation_agent import IdeationAgent
    from agents.idea_validator_agent import IdeaValidatorAgent

    logger.info("Regenerating topics for domain: %s", state.get("domain_input", "?"))

    # Phase 1: Re-run ideation
    ideation = IdeationAgent()
    ideation_result = ideation.run(state)

    # Merge ideation result into state for the validator
    merged = {**state, **ideation_result}

    # Phase 2: Re-run idea validator
    validator = IdeaValidatorAgent()
    validator_result = validator.run(merged)

    # Merge validator result
    merged.update(validator_result)

    logger.info("Topic regeneration complete.")
    return merged
