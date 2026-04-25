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
from agents.research_plan_builder import build_research_plan_from_candidate

logger = get_logger(__name__)

MAX_REGENERATION_ROUNDS = int(os.getenv("HITL_MAX_REGENERATION_ROUNDS", "3"))


def apply_idea_selection(idea_index: int) -> str | None:
    """Promote candidate at *idea_index* to rank-1 and sync plan/context."""
    screening_path = settings.topic_screening_path()
    context_path = settings.research_context_path()
    plan_path = settings.research_plan_path()

    if not os.path.exists(screening_path):
        return None

    try:
        with open(screening_path, "r", encoding="utf-8") as f:
            screening = json.load(f)
    except Exception:
        return None

    candidates: list = screening.get("candidates", [])
    if idea_index < 0 or idea_index >= len(candidates):
        return None

    selected = candidates.pop(idea_index)
    candidates.insert(0, selected)
    for i, candidate in enumerate(candidates):
        candidate["rank"] = i + 1
    screening["candidates"] = candidates

    with open(screening_path, "w", encoding="utf-8") as f:
        json.dump(screening, f, indent=2, ensure_ascii=False)

    run_id = screening.get("run_id", "unknown")
    selected_title = str(selected.get("title") or "")

    if os.path.exists(plan_path):
        try:
            plan = build_research_plan_from_candidate(
                selected,
                evaluation=selected.get("evaluation"),
                run_id=run_id,
            )
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Could not sync research_plan.json after selection: %s", exc)

    if os.path.exists(context_path):
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                ctx = json.load(f)
        except Exception:
            ctx = {}
        if not isinstance(ctx, dict):
            ctx = {}
        ctx["selected_topic"] = selected
        ctx["selection_overridden"] = True
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(ctx, f, indent=2, ensure_ascii=False)

    return selected_title


def apply_idea_selection_by_candidate_id(candidate_id: str) -> str | None:
    """Promote candidate with *candidate_id* to rank-1 and sync plan/context."""
    screening_path = settings.topic_screening_path()
    if not os.path.exists(screening_path):
        return None

    try:
        with open(screening_path, "r", encoding="utf-8") as f:
            screening = json.load(f)
    except Exception:
        return None

    candidates: list = screening.get("candidates", [])
    selected_index = next(
        (
            idx
            for idx, candidate in enumerate(candidates)
            if str(candidate.get("candidate_id", "")).strip() == candidate_id
            or str(candidate.get("topic_id", "")).strip() == candidate_id
        ),
        None,
    )
    if selected_index is None and candidate_id.startswith("legacy_"):
        # API list endpoint may synthesize legacy IDs when screening candidates
        # do not carry a canonical candidate_id.
        suffix = candidate_id.removeprefix("legacy_")
        if suffix.isdigit():
            legacy_index = int(suffix) - 1
            if 0 <= legacy_index < len(candidates):
                selected_index = legacy_index

    if selected_index is None:
        return None

    return apply_idea_selection(selected_index)


def load_validated_topics(validation_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load validated ideas aligned with topic_screening.json candidate order.

    Returns list of dicts with title, brief_rationale, rank, overall_verdict, etc.
    Indices match topic_screening.json candidates so apply_idea_selection(i)
    operates on the entry shown at position i.

    The validator appends both pre-substitution originals (verdict="failed") and
    their substitutes to validated_ideas. We dedupe by title against the current
    topic_screening.candidates list, preferring the substitute when both exist.

    Falls back to non-failed-only filtering when screening data is unavailable,
    and finally to all ideas — never returns empty when validation_ideas exist,
    so the HITL picker can always offer a choice (even if all are flagged).
    """
    path = validation_path or settings.idea_validation_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load validation report from %s: %s", path, exc)
        return []

    ideas = report.get("validated_ideas", []) or []
    if not ideas:
        return []

    screening_titles = _load_screening_titles()
    if screening_titles:
        by_title: Dict[str, Dict[str, Any]] = {}
        for idea in ideas:
            title = str(idea.get("title", ""))
            if not title:
                continue
            prev = by_title.get(title)
            if prev is None:
                by_title[title] = idea
            elif (
                prev.get("overall_verdict") == "failed"
                and idea.get("overall_verdict") != "failed"
            ):
                by_title[title] = idea
        aligned = [by_title[t] for t in screening_titles if t in by_title]
        if aligned:
            return aligned

    non_failed = [i for i in ideas if i.get("overall_verdict") != "failed"]
    if non_failed:
        return non_failed
    return ideas


def _load_screening_titles() -> List[str]:
    """Return the ordered list of titles from topic_screening.json (or [])."""
    screening_path = settings.topic_screening_path()
    if not os.path.exists(screening_path):
        return []
    try:
        with open(screening_path, "r", encoding="utf-8") as f:
            screening = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return [str(c.get("title", "")) for c in screening.get("candidates", [])]


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
    from agents.ideation_agent import ideation_node
    from agents.idea_validator_agent import IdeaValidatorAgent

    logger.info("Regenerating topics for domain: %s", state.get("domain_input", "?"))

    # Phase 1: Re-run ideation (via thin router so V2 / V0 routing applies)
    ideation_result = ideation_node(state)

    # Merge ideation result into state for the validator
    merged = {**state, **ideation_result}

    # Phase 2: Re-run idea validator
    validator = IdeaValidatorAgent()
    validator_result = validator.run(merged)

    # Merge validator result
    merged.update(validator_result)

    logger.info("Topic regeneration complete.")
    return merged


# ── TENTATIVE pool helpers (Module 1 upgrade) ─────────────────────────────────

def load_tentative_topics(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load TENTATIVE topics from tentative_pool.json.

    Returns an empty list if the file is missing or unreadable.
    """
    pool_path = path or settings.tentative_pool_path()
    if not os.path.exists(pool_path):
        return []
    try:
        with open(pool_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tentative", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load tentative pool from %s: %s", pool_path, exc)
        return []


def _save_tentative_pool(pool: List[Dict[str, Any]], path: Optional[str] = None) -> None:
    pool_path = path or settings.tentative_pool_path()
    existing_meta: dict = {}
    if os.path.exists(pool_path):
        try:
            with open(pool_path, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
        except Exception:
            pass
    existing_meta["tentative"] = pool
    os.makedirs(os.path.dirname(pool_path), exist_ok=True)
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(existing_meta, f, indent=2, ensure_ascii=False)


def promote_tentative(idx: int, path: Optional[str] = None) -> bool:
    """Move a TENTATIVE topic at index *idx* into topic_screening.json as rank-1.

    Returns True on success, False if index is out of range or file missing.
    """
    pool = load_tentative_topics(path)
    if idx < 0 or idx >= len(pool):
        logger.warning("promote_tentative: index %d out of range (pool size=%d)", idx, len(pool))
        return False

    entry = pool.pop(idx)
    _save_tentative_pool(pool, path)

    # Insert as rank-1 in topic_screening.json
    screening_path = settings.topic_screening_path()
    screening: dict = {"candidates": []}
    if os.path.exists(screening_path):
        try:
            with open(screening_path, "r", encoding="utf-8") as f:
                screening = json.load(f)
        except Exception:
            pass

    candidates: list = screening.get("candidates", [])
    promoted = {
        **entry,
        "rank": 1,
        "promoted_from_tentative": True,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    candidates.insert(0, promoted)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
    screening["candidates"] = candidates

    with open(screening_path, "w", encoding="utf-8") as f:
        json.dump(screening, f, indent=2, ensure_ascii=False)

    logger.info("Promoted TENTATIVE topic '%s' to rank-1 in topic_screening.json",
                entry.get("title", entry.get("topic_id", "?")))
    return True


def kill_tentative(idx: int, domain: str, path: Optional[str] = None) -> bool:
    """Move a TENTATIVE topic at index *idx* into the domain graveyard.

    Returns True on success, False if index is out of range.
    """
    pool = load_tentative_topics(path)
    if idx < 0 or idx >= len(pool):
        logger.warning("kill_tentative: index %d out of range (pool size=%d)", idx, len(pool))
        return False

    entry = pool.pop(idx)
    _save_tentative_pool(pool, path)

    graveyard_path = settings.ideas_graveyard_path(domain=domain)
    existing: list = []
    if os.path.exists(graveyard_path):
        try:
            with open(graveyard_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.append({
        "title": entry.get("title", entry.get("topic_id", "")),
        "topic_id": entry.get("topic_id", ""),
        "rejection_reason": "killed_from_tentative_pool",
        "failed_gates": entry.get("failed_gates", []),
        "rejected_at": datetime.now(timezone.utc).isoformat(),
    })

    os.makedirs(os.path.dirname(graveyard_path), exist_ok=True)
    with open(graveyard_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info("Killed TENTATIVE topic '%s' → graveyard", entry.get("title", "?"))
    return True


def rerun_tentative_reflection(
    idx: int,
    path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Re-run one reflection round for the TENTATIVE topic at index *idx*.

    If the topic becomes ACCEPTED it is promoted automatically.
    Returns the updated pool entry (or None if idx is invalid).
    """
    from agents.budget_tracker import BudgetTracker
    from agents.reflection_loop import run_reflection_loop
    from agents.rule_engine import RuleEngine
    from models.topic_schema import FinalStatus, SeedCandidate, Topic

    pool = load_tentative_topics(path)
    if idx < 0 or idx >= len(pool):
        logger.warning("rerun_tentative_reflection: index %d out of range", idx)
        return None

    entry = pool[idx]
    try:
        topic = Topic.model_validate(entry.get("topic_dict", {}))
    except Exception as e:
        logger.warning("Cannot reconstruct Topic for rerun: %s", e)
        return None

    seed = SeedCandidate(
        topic=topic,
        declared_sources=entry.get("declared_sources", []),
    )
    budget = BudgetTracker()
    trace = run_reflection_loop(seed, budget, rule_engine=RuleEngine(), max_rounds=1)

    entry["last_rerun_status"] = trace.final_status.value
    entry["last_rerun_at"] = datetime.now(timezone.utc).isoformat()
    entry["trace_rounds"] = entry.get("trace_rounds", 0) + len(trace.rounds)

    if trace.final_status == FinalStatus.ACCEPTED:
        logger.info("Tentative topic '%s' → ACCEPTED after rerun; promoting.", entry.get("title"))
        pool.pop(idx)
        _save_tentative_pool(pool, path)
        promote_tentative.__doc__  # satisfy linter
        # Rebuild pool without the promoted entry and promote inline
        screening_path = settings.topic_screening_path()
        screening: dict = {"candidates": []}
        if os.path.exists(screening_path):
            try:
                with open(screening_path, "r", encoding="utf-8") as f:
                    screening = json.load(f)
            except Exception:
                pass
        candidates = screening.get("candidates", [])
        promoted = {**entry, "rank": 1, "promoted_from_tentative": True,
                    "promoted_at": datetime.now(timezone.utc).isoformat()}
        candidates.insert(0, promoted)
        for i, c in enumerate(candidates):
            c["rank"] = i + 1
        screening["candidates"] = candidates
        with open(screening_path, "w", encoding="utf-8") as f:
            json.dump(screening, f, indent=2, ensure_ascii=False)
        return promoted

    pool[idx] = entry
    _save_tentative_pool(pool, path)
    return entry
