from __future__ import annotations

from copy import deepcopy
from typing import Any

from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposedCandidate


def validate_candidate_export_contract(
    candidate: ComposedCandidate,
    gate_status: dict[str, Any],
    *,
    no_paid_api: bool = True,
) -> dict[str, Any]:
    """Apply strict export contract before any candidate is marked ready."""
    registry = SourceRegistry.load()
    status = deepcopy(gate_status or {})
    reasons: list[str] = list(status.get("reasons") or [])
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    exp_sid = registry.resolve(candidate.exposure_source)
    out_sid = registry.resolve(candidate.outcome_source)

    if not exp_sid or not out_sid:
        blocking_reasons.append("source_not_in_registry")

    exp_spec = registry.sources.get(exp_sid, {}) if exp_sid else {}
    out_spec = registry.sources.get(out_sid, {}) if out_sid else {}

    if exp_sid and "exposure" not in (exp_spec.get("roles") or []):
        blocking_reasons.append("missing_exposure_role_source")
    if out_sid and "outcome" not in (out_spec.get("roles") or []):
        blocking_reasons.append("missing_outcome_role_source")

    core_source_ids: list[str] = [sid for sid in [exp_sid, out_sid] if sid]
    for source_name in (candidate.join_plan.get("controls") or []):
        sid = registry.resolve(source_name)
        if sid:
            core_source_ids.append(sid)
    for source_name in (candidate.join_plan.get("boundary_source") or []):
        sid = registry.resolve(source_name)
        if sid:
            core_source_ids.append(sid)

    if any(not bool(registry.sources.get(sid, {}).get("machine_readable")) for sid in core_source_ids):
        blocking_reasons.append("missing_machine_readable_source")

    controls = [registry.resolve(s) or s for s in (candidate.join_plan.get("controls") or [])]
    if "ACS" not in controls:
        warnings.append("missing_standard_control_source")

    boundary_ids = [registry.resolve(s) or s for s in (candidate.join_plan.get("boundary_source") or [])]
    if "TIGER_Lines" not in boundary_ids:
        has_boundary_role = any(
            "boundary" in ((registry.sources.get(sid, {}) or {}).get("roles") or [])
            for sid in boundary_ids
        )
        if not has_boundary_role:
            blocking_reasons.append("missing_join_path")

    threats = list(candidate.key_threats or [])
    mitigations = dict(candidate.mitigations or {})
    if len(threats) < 3:
        blocking_reasons.append("missing_identification_threats")
    if threats and any(t not in mitigations for t in threats):
        blocking_reasons.append("missing_threat_mitigation")

    if candidate.required_secrets:
        warnings.append("required_secrets_present")

    if no_paid_api and any(bool(registry.sources.get(sid, {}).get("cost_required")) for sid in core_source_ids):
        blocking_reasons.append("paid_source_not_allowed")

    reasons.extend([r for r in blocking_reasons + warnings if r not in reasons])

    if blocking_reasons:
        shortlist_status = "blocked"
    elif warnings or status.get("shortlist_status") == "review":
        shortlist_status = "review"
    else:
        shortlist_status = "ready"

    overall = "fail" if blocking_reasons else ("warning" if warnings else status.get("overall", "pass"))

    status.update(
        {
            "overall": overall,
            "shortlist_status": shortlist_status,
            "claude_code_ready": shortlist_status == "ready",
            "blocking_reasons": blocking_reasons,
            "repairable_warnings": warnings,
            "reasons": reasons,
        }
    )
    return status
