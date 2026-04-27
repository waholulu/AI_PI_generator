"""Evaluates whether a candidate's development pack is Claude Code Ready.

Pack status values (in order of readiness):
  not_generated       — pack directory was never created
  generated           — directory exists but required files are missing
  complete            — all required files present but a blocker prevents ready
  blocked_by_high_risk — complete but automation_risk == "high"
  blocked_by_secret   — complete but required_secrets is non-empty
  review_required     — complete, low/medium risk, no secrets, but gate warning or
                         experimental technology tags
  claude_code_ready   — all files present, no blockers, gate pass or warning (not fail)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


_REQUIRED_FILES = [
    "implementation_spec.json",
    "claude_task_prompt.md",
    "data_contract.yaml",
    "feature_plan.yaml",
    "analysis_plan.yaml",
    "acceptance_tests.md",
]

_HIGH_RISK_TAGS = {"streetview_cv", "deep_learning", "satellite_cv", "experimental"}


def evaluate_development_pack_readiness(
    candidate: dict[str, Any],
    gate_status: dict[str, Any],
    pack_dir: Path | None,
) -> dict[str, Any]:
    """Return granular readiness status for a candidate's development pack.

    Status values (most to least ready):
      claude_code_ready   — ready for Claude Code autonomous implementation
      review_required     — ready after human review (experimental tags, gate warning)
      blocked_by_secret   — requires API secrets; cannot run keyless
      blocked_by_high_risk — automation_risk is "high"
      complete            — all files present but other blocker
      generated           — pack exists but required files are missing
      not_generated       — pack directory does not exist
    """
    if pack_dir is None or not pack_dir.exists():
        return {
            "development_pack_status": "not_generated",
            "claude_code_ready": False,
            "missing_files": _REQUIRED_FILES[:],
            "blocking_reasons": ["pack_directory_not_created"],
        }

    missing_files = [
        fname for fname in _REQUIRED_FILES
        if not (pack_dir / fname).exists() or (pack_dir / fname).stat().st_size == 0
    ]

    if missing_files:
        return {
            "development_pack_status": "generated",
            "claude_code_ready": False,
            "development_pack_files": [f for f in _REQUIRED_FILES if f not in missing_files],
            "missing_files": missing_files,
            "blocking_reasons": [f"missing_files:{','.join(missing_files)}"],
        }

    # All files present — check blockers in priority order
    blocking_reasons: list[str] = []

    automation_risk = candidate.get("automation_risk", "high")
    required_secrets = candidate.get("required_secrets") or []
    tech_tags = set(candidate.get("technology_tags") or [])
    experimental_tags = tech_tags & _HIGH_RISK_TAGS
    gate_overall = gate_status.get("overall")
    shortlist = gate_status.get("shortlist_status")

    has_high_risk = automation_risk == "high"
    has_secrets = bool(required_secrets)
    has_experimental_tags = bool(experimental_tags)
    gate_failed = gate_overall == "fail"
    shortlist_blocked = shortlist == "blocked"

    if gate_failed:
        blocking_reasons.append("gate_failed")
    if shortlist_blocked:
        blocking_reasons.append("shortlist_blocked")
    if has_high_risk:
        blocking_reasons.append("high_automation_risk")
    if has_secrets:
        blocking_reasons.append(f"required_secrets:{','.join(required_secrets)}")
    if has_experimental_tags:
        blocking_reasons.append(f"experimental_tags:{','.join(sorted(experimental_tags))}")

    claude_code_ready = len(blocking_reasons) == 0

    # Determine granular status
    if claude_code_ready:
        status = "claude_code_ready"
    elif has_high_risk and not gate_failed and not shortlist_blocked:
        status = "blocked_by_high_risk"
    elif has_secrets and not has_high_risk and not gate_failed and not shortlist_blocked:
        status = "blocked_by_secret"
    elif has_experimental_tags and not has_secrets and not has_high_risk:
        status = "review_required"
    elif gate_overall == "warning" and not has_high_risk and not has_secrets:
        status = "review_required"
    elif gate_failed or shortlist_blocked:
        status = "complete"  # files OK but gate blocks deployment
    else:
        status = "complete"

    return {
        "development_pack_status": status,
        "claude_code_ready": claude_code_ready,
        "development_pack_files": list(_REQUIRED_FILES),
        "missing_files": [],
        "blocking_reasons": blocking_reasons,
    }
