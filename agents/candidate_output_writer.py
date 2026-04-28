"""Writes standardized run-scoped output files for the candidate factory.

Responsibilities:
- feasibility_report.json  — per-candidate gate subchecks + summary counts
- development_pack_index.json — mapping of candidate_id → pack status + file paths
- gate_trace.json — per-candidate gate trace (pass-through from cards)

All output files go through settings path helpers; never hardcode "output/".
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from agents import settings
from agents.logging_config import get_logger

logger = get_logger(__name__)


def _gate_overall(card: dict) -> str:
    return card.get("gate_status", {}).get("overall", "fail")


def _shortlist(card: dict) -> str:
    return card.get("shortlist_status", "blocked")


def write_feasibility_report(run_id: str, cards: list[dict[str, Any]]) -> Path:
    """Write output/feasibility_report.json from candidate cards."""
    shortlist_counts: Counter = Counter(_shortlist(c) for c in cards)
    readiness_counts: Counter = Counter(c.get("readiness", "unknown") for c in cards)
    risk_counts: Counter = Counter(c.get("automation_risk", "high") for c in cards)
    claude_ready_count = sum(1 for c in cards if c.get("claude_code_ready", False))

    # Raw gate flags — for QA / gate pass-rate analysis.
    debug_reason_counts: Counter = Counter()
    for card in cards:
        gs = card.get("gate_status", {})
        for reason in gs.get("reasons", []):
            debug_reason_counts[reason] += 1

    # User-visible reasons — what users actually see (blocking + review tier only).
    user_visible_reason_counts: Counter = Counter()
    for card in cards:
        for reason in card.get("user_visible_reasons", []):
            user_visible_reason_counts[reason] += 1

    candidate_summaries = []
    for card in cards:
        gs = card.get("gate_status", {})
        candidate_summaries.append(
            {
                "candidate_id": card["candidate_id"],
                "overall": _gate_overall(card),
                "shortlist_status": _shortlist(card),
                "readiness": card.get("readiness", "unknown"),
                "claude_code_ready": card.get("claude_code_ready", False),
                "automation_risk": card.get("automation_risk", "high"),
                "subchecks": gs.get("subchecks", {}),
                # Raw gate reasons for debugging — not shown in UI.
                "reasons": gs.get("reasons", []),
                "user_visible_reasons": card.get("user_visible_reasons", []),
            }
        )

    report = {
        "run_id": run_id,
        "candidate_count": len(cards),
        "summary": {
            "ready": shortlist_counts.get("ready", 0),
            "review": shortlist_counts.get("review", 0),
            "blocked": shortlist_counts.get("blocked", 0),
            "low_risk": risk_counts.get("low", 0),
            "medium_risk": risk_counts.get("medium", 0),
            "high_risk": risk_counts.get("high", 0),
            "claude_code_ready": claude_ready_count,
        },
        "readiness_counts": dict(readiness_counts),
        "user_visible_reason_counts": dict(user_visible_reason_counts),
        "debug_reason_counts": dict(debug_reason_counts),
        "candidates": candidate_summaries,
    }

    out_path = settings.output_dir() / "feasibility_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote feasibility_report.json — %d candidates", len(cards))
    return out_path


def write_development_pack_index(run_id: str, cards: list[dict[str, Any]]) -> Path:
    """Write output/development_pack_index.json from candidate cards."""
    packs_dir = settings.development_packs_dir()
    packs = []
    for card in cards:
        cid = card["candidate_id"]
        pack_dir = packs_dir / cid
        status = card.get("development_pack_status", "not_generated")
        claude_ready = card.get("claude_code_ready", False)
        file_names = card.get("development_pack_files", [])
        files = (
            {fname: str(pack_dir / fname) for fname in file_names}
            if pack_dir.exists()
            else {}
        )
        packs.append(
            {
                "candidate_id": cid,
                "status": status,
                "claude_code_ready": claude_ready,
                "files": files,
            }
        )

    index = {"run_id": run_id, "packs": packs}
    out_path = settings.output_dir() / "development_pack_index.json"
    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Wrote development_pack_index.json — %d packs (%d claude_code_ready)",
        len(packs),
        sum(1 for p in packs if p["claude_code_ready"]),
    )
    return out_path


def write_gate_trace(run_id: str, cards: list[dict[str, Any]]) -> Path:
    """Write output/gate_trace.json — per-candidate gate trace from cards."""
    trace = {
        "run_id": run_id,
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "gate_status": c.get("gate_status", {}),
                "repair_history": c.get("repair_history", []),
                "scores": c.get("scores", {}),
            }
            for c in cards
        ],
    }
    out_path = settings.output_dir() / "gate_trace.json"
    out_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote gate_trace.json — %d candidates", len(cards))
    return out_path
