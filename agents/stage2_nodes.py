"""Stage 2 LangGraph nodes for the two-stage candidate flow.

After the user picks one candidate at the HITL checkpoint, the graph runs:

    novelty_check  →  (cond: HITL#2 if not_novel) → literature
                                               →   development_pack
                                               →   drafter
                                               →   data_fetcher

These nodes operate ONLY on the rank-1 candidate in topic_screening.json
(promoted there by `apply_idea_selection_by_candidate_id`).  They are
intentionally cheap: novelty_check makes a single OpenAlex pass for the
selected candidate; development_pack writes one Claude Code pack.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import settings
from agents.development_pack_status import evaluate_development_pack_readiness
from agents.development_pack_writer import write_development_pack
from agents.logging_config import get_logger

logger = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_selected_candidate() -> dict[str, Any] | None:
    """Return the rank-1 candidate from topic_screening.json (or None)."""
    path = settings.topic_screening_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    # Selection promotes the chosen candidate to index 0.
    return candidates[0]


def _candidate_to_raw_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    """Recover a ComposedCandidate-shaped payload from a screening entry.

    `apply_idea_selection_by_candidate_id` keeps the screening entry intact;
    the underlying ComposedCandidate fields used by `write_development_pack`
    are nested under `_raw` on full cards but flat on screening entries.
    Build the union here so either shape works.
    """
    raw = candidate.get("_raw") or {}
    if raw:
        return raw
    # Synthesize from the screening entry's flat fields.
    return {
        "candidate_id": candidate.get("candidate_id") or candidate.get("topic_id"),
        "exposure_family": candidate.get("exposure_family") or candidate.get("exposure_label"),
        "outcome_family": candidate.get("outcome_family") or candidate.get("outcome_label"),
        "exposure_source": candidate.get("exposure_source"),
        "outcome_source": candidate.get("outcome_source"),
        "unit_of_analysis": candidate.get("unit_of_analysis"),
        "method_template": candidate.get("method") or candidate.get("method_template"),
        "claim_strength": candidate.get("claim_strength", "associational"),
        "technology_tags": candidate.get("technology_tags", []),
        "required_secrets": candidate.get("required_secrets", []),
        "automation_risk": candidate.get("automation_risk", "medium"),
        "cloud_safe": candidate.get("cloud_safe", True),
        "key_threats": candidate.get("key_threats", []),
        "mitigations": candidate.get("mitigations", {}),
        "join_plan": candidate.get("join_plan", {}),
    }


def _write_research_context_update(updates: dict[str, Any]) -> None:
    """Merge ``updates`` into research_context.json (creating it if absent)."""
    path = settings.research_context_path()
    ctx: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    ctx = loaded
        except (OSError, json.JSONDecodeError):
            ctx = {}
    ctx.update(updates)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)


# ── Node 1: novelty_check ──────────────────────────────────────────────────

def novelty_check_node(state: dict) -> dict:
    """Slim novelty check on the user-selected candidate only.

    Reuses ``check_novelty`` from idea_validator_agent (single OpenAlex pass +
    LLM assessment).  No substitution, no backup pool — the user chose this
    candidate.  Writes the verdict to research_context.json so the second
    HITL checkpoint can render similar papers, and returns
    ``novelty_verdict`` in the state for the conditional edge.

    Verdict values:
      - ``novel``                 → continue straight to literature
      - ``partially_overlapping`` → continue (warning attached)
      - ``already_published``     → triggers HITL#2 confirmation
      - ``unavailable``           → degraded; continue with warning
    """
    logger.info("--- Stage 2: novelty_check on selected candidate ---")

    candidate = _load_selected_candidate()
    if candidate is None:
        logger.warning("novelty_check: no selected candidate found; skipping.")
        return {
            "novelty_verdict": "unavailable",
            "execution_status": "novelty_skipped",
        }

    title = str(candidate.get("title") or "")
    rationale = str(candidate.get("brief_rationale") or "")

    try:
        # Late import to avoid circular dependency at module load time.
        from agents.idea_validator_agent import IdeaValidatorAgent, check_novelty

        agent = IdeaValidatorAgent()
        from_year = datetime.now(timezone.utc).year - int(
            os.getenv("NOVELTY_CHECK_YEARS", "2")
        )
        n_queries = int(os.getenv("NOVELTY_QUERIES_PER_IDEA", "3"))
        results_per_query = int(os.getenv("NOVELTY_RESULTS_PER_QUERY", "10"))

        result = check_novelty(
            agent.llm, title, rationale, from_year, n_queries, results_per_query
        )
        verdict = result.verdict
        similar_papers = list(result.similar_papers or [])
        search_queries = list(result.search_queries_used or [])
    except Exception as exc:
        logger.warning("novelty_check failed (%s); marking unavailable.", exc)
        verdict = "unavailable"
        similar_papers = []
        search_queries = []

    logger.info(
        "novelty_check verdict for selected candidate: %s (similar=%d)",
        verdict, len(similar_papers),
    )

    _write_research_context_update({
        "novelty_verdict": verdict,
        "novelty_similar_papers": similar_papers[:5],
        "novelty_search_queries": search_queries,
        "novelty_checked_at": datetime.now(timezone.utc).isoformat(),
    })

    degraded = ["novelty_check:unavailable"] if verdict == "unavailable" else []

    return {
        "novelty_verdict": verdict,
        "novelty_search_terms": search_queries,
        "execution_status": "novelty_checked",
        "degraded_nodes": degraded,
    }


def route_after_novelty(state: dict) -> str:
    """Conditional edge: pause for HITL#2 only when novelty is exhausted.

    Returns ``"literature"`` to proceed straight through; the LangGraph
    ``interrupt_before=["literature"]`` configured at compile time fires the
    HITL pause when ``novelty_verdict == "already_published"``.

    LangGraph cannot conditionally interrupt at runtime, so we emulate the
    behavior by routing to a dedicated ``novelty_review`` no-op node when a
    second checkpoint is needed.
    """
    if state.get("novelty_verdict") == "already_published":
        return "novelty_review"
    return "literature"


def novelty_review_node(state: dict) -> dict:
    """No-op pass-through used as the second HITL checkpoint anchor.

    The graph is compiled with ``interrupt_before=["novelty_review"]`` so
    execution pauses here when ``novelty_verdict == "already_published"``.
    The user reviews similar papers and either confirms (resume → literature)
    or rejects (re-run picker via the existing /approve regenerate path).
    """
    logger.info("--- Stage 2: HITL#2 anchor (novelty_review) — confirmation pending ---")
    return {"execution_status": "awaiting_novelty_confirmation"}


# ── Node 2: development_pack ───────────────────────────────────────────────

def development_pack_node(state: dict) -> dict:
    """Generate a Claude Code development pack for the selected candidate.

    Runs after literature so the pack's analysis plan + claude_task_prompt
    can reference real evidence harvested from the literature node.  Writes
    a single entry into development_pack_index.json (replacing the empty
    Stage 1 placeholder).
    """
    logger.info("--- Stage 2: development_pack for selected candidate ---")

    candidate = _load_selected_candidate()
    if candidate is None:
        logger.warning("development_pack: no selected candidate; skipping.")
        return {
            "development_pack_index_path": str(
                settings.output_dir() / "development_pack_index.json"
            ),
            "execution_status": "development_pack_skipped",
        }

    run_id = settings.current_run_scope() or "unknown"
    payload = _candidate_to_raw_payload(candidate)
    candidate_id = str(payload.get("candidate_id") or candidate.get("candidate_id") or "selected")

    pack_dir: Path | None = None
    try:
        pack_dir = write_development_pack(run_id, payload)
        logger.info("development_pack written → %s", pack_dir)
    except Exception as exc:
        logger.warning("write_development_pack failed for %s: %s", candidate_id, exc)

    gate_status = candidate.get("gate_status") or {}
    readiness = evaluate_development_pack_readiness(payload, gate_status, pack_dir)

    # Write a one-entry development_pack_index.json.
    index = {
        "run_id": run_id,
        "packs": [
            {
                "candidate_id": candidate_id,
                "status": readiness.get("development_pack_status", "review_required"),
                "claude_code_ready": readiness.get("claude_code_ready", False),
                "files": (
                    {
                        fname: str((pack_dir / fname))
                        for fname in readiness.get("development_pack_files", [])
                    }
                    if pack_dir is not None
                    else {}
                ),
            }
        ],
    }
    out_path = settings.output_dir() / "development_pack_index.json"
    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Wrote development_pack_index.json (Stage 2): %s, claude_code_ready=%s",
        candidate_id, readiness.get("claude_code_ready"),
    )

    degraded = []
    if pack_dir is None:
        degraded.append("development_pack:write_failed")
    elif not readiness.get("claude_code_ready"):
        degraded.append("development_pack:not_claude_ready")

    return {
        "development_pack_index_path": str(out_path),
        "execution_status": "development_pack_written",
        "degraded_nodes": degraded,
    }
