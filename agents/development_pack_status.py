"""Evaluates whether a candidate's development pack is Claude Code Ready."""
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
    """Return readiness status for a candidate's development pack.

    A pack is ``claude_code_ready`` only when:
    - all required files exist in pack_dir
    - automation_risk is not "high"
    - required_secrets is empty
    - no experimental/high-risk technology tags
    - gate overall is "pass" or "warning"
    """
    blocking_reasons: list[str] = []
    missing_files: list[str] = []

    if pack_dir is None or not pack_dir.exists():
        return {
            "development_pack_status": "not_generated",
            "claude_code_ready": False,
            "missing_files": _REQUIRED_FILES[:],
            "blocking_reasons": ["pack_directory_not_created"],
        }

    for fname in _REQUIRED_FILES:
        fpath = pack_dir / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing_files.append(fname)

    automation_risk = candidate.get("automation_risk", "high")
    if automation_risk == "high":
        blocking_reasons.append("high_automation_risk")

    required_secrets = candidate.get("required_secrets") or []
    if required_secrets:
        blocking_reasons.append(f"required_secrets:{','.join(required_secrets)}")

    tech_tags = set(candidate.get("technology_tags") or [])
    experimental_tags = tech_tags & _HIGH_RISK_TAGS
    if experimental_tags:
        blocking_reasons.append(f"experimental_tags:{','.join(sorted(experimental_tags))}")

    gate_overall = gate_status.get("overall")
    if gate_overall == "fail":
        blocking_reasons.append("gate_failed")

    shortlist = gate_status.get("shortlist_status")
    if shortlist == "blocked":
        blocking_reasons.append("shortlist_blocked")

    if missing_files:
        blocking_reasons.append(f"missing_files:{','.join(missing_files)}")

    claude_code_ready = len(blocking_reasons) == 0

    if claude_code_ready:
        status = "ready"
    elif shortlist == "review" and not any(
        r.startswith("missing_files") or r == "gate_failed" or r == "shortlist_blocked"
        for r in blocking_reasons
    ):
        status = "review_required"
    else:
        status = "blocked"

    return {
        "development_pack_status": status,
        "claude_code_ready": claude_code_ready,
        "development_pack_files": [f for f in _REQUIRED_FILES if f not in missing_files],
        "missing_files": missing_files,
        "blocking_reasons": blocking_reasons,
    }
