"""Flag classification and user-facing readiness computation for composed candidates.

Raw gate flags from precheck_candidate() and validate_candidate_export_contract()
are internal lint results. This module converts them into four tiers:

  INFO_FLAGS        — system working as expected; silent to users
  AUTO_FIXABLE_FLAGS — repair loop already fixed these; show as "auto-fixed"
  REVIEW_FLAGS      — needs human attention; show as action reasons
  BLOCKING_FLAGS    — prevents automated execution; always shown

The compute_candidate_readiness() function maps gate_status + repair_history
into a single readiness value:
  "ready"               — all pass, no fixes needed
  "ready_after_auto_fix"— repair loop applied fixes; result is clean
  "needs_review"        — shortlist_status="review"; human attention needed
  "blocked"             — shortlist_status="blocked" or blocking_reasons present
"""
from __future__ import annotations

FLAG_MESSAGES: dict[str, str] = {
    # Blocking flags
    "source_not_in_registry": "数据源未收录于 source catalog，无法验证可用性",
    "missing_exposure_role_source": "暴露变量未绑定可机读 exposure source（请检查数据目录）",
    "missing_outcome_role_source": "结局变量未绑定可机读 outcome source（请检查数据目录）",
    "missing_machine_readable_source": "未找到可机读（API 或公开下载）数据源",
    "missing_join_path": "暴露与结局数据缺少可自动执行的空间连接路径",
    "paid_source_not_allowed": "所用数据源需要付费 API，违反 no_paid_api 约束",
    "required_secrets_blocks_ready": "候选依赖需鉴权的密钥，无法全自动执行",
    "high_automation_risk_blocks_ready": "自动化风险为 high，不符合当前自动化约束",
    "missing_threat_mitigation": "未为所有已识别的识别威胁提供缓解方案",
    "threat_mitigation_coverage_low": "识别威胁覆盖率偏低（< 60%）",
    # Review flags
    "manual_download_required": "至少一个数据源需要手动下载（非全自动）",
    "semi_automated_source": "数据源为半自动（需人工介入）",
    "aggregation_plan_required": "数据空间聚合方案未完整记录",
    "experimental_source_in_use": "使用了实验性数据源，稳定性未经验证",
    "experimental_source_requires_key": "实验性数据源需要 API 密钥",
    "causal_assumption_weak": "因果识别假设较弱，建议降为关联研究声明",
    "time_overlap_insufficient": "数据源时间范围与目标时间窗口不完全重叠",
    # Auto-fixable flags (shown as "already fixed")
    "missing_default_controls": "缺少默认控制变量（已自动补充 ACS）",
    "missing_boundary_source": "缺少边界数据源（已自动补充 TIGER_Lines）",
    "missing_identification_threats": "识别威胁未填写（已从方法模板自动填充）",
    "aggregation_required": "需要空间聚合步骤（已自动生成聚合方案）",
}


def _translate_flag(flag: str) -> str:
    """Return human-readable Chinese message for a raw gate flag.

    Falls back to the raw flag string when no translation is registered, so
    new flags are always surfaced rather than silently dropped.
    """
    base = flag.split(":")[0]
    # Include colon-suffix detail for context (e.g. "threat_mitigation_coverage_low:40%")
    detail = flag[len(base):] if len(flag) > len(base) else ""
    msg = FLAG_MESSAGES.get(base) or FLAG_MESSAGES.get(flag)
    if msg:
        return f"{msg}{detail}" if detail else msg
    return flag


INFO_FLAGS: frozenset[str] = frozenset({
    "source_alias_resolved",
    "non_canonical_source_name",
    "canonicalize_source_name",
    "partial_registry_match",   # fuzzy alias hit in legacy registry; not a real data risk
})

AUTO_FIXABLE_FLAGS: frozenset[str] = frozenset({
    "missing_default_controls",
    "missing_boundary_source",
    "missing_identification_threats",
    "aggregation_required",
})

REVIEW_FLAGS: frozenset[str] = frozenset({
    "manual_download_required",
    "semi_automated_source",
    "aggregation_plan_required",
    "experimental_source_in_use",
    "experimental_source_requires_key",
    "causal_assumption_weak",
    "time_overlap_insufficient",
})

BLOCKING_FLAGS: frozenset[str] = frozenset({
    "source_not_in_registry",
    "missing_exposure_role_source",
    "missing_outcome_role_source",
    "missing_machine_readable_source",
    "missing_join_path",
    "paid_source_not_allowed",
    "required_secrets_blocks_ready",
    "high_automation_risk_blocks_ready",
    "missing_threat_mitigation",
    "threat_mitigation_coverage_low",  # colon-suffixed variant from precheck
})

# Flags that directly indicate a data-path failure
_DATA_FAIL_FLAGS: frozenset[str] = frozenset({
    "source_not_in_registry",
    "missing_exposure_role_source",
    "missing_outcome_role_source",
    "missing_machine_readable_source",
    "missing_join_path",
})


def classify_flags(raw_flags: list[str]) -> dict:
    """Classify raw gate flags into four user-facing tiers.

    Unknown flags (not in any set) fall into review_reasons so they
    surface rather than being silently dropped.
    """
    info_notes: list[str] = []
    auto_fixes: list[str] = []
    review_reasons: list[str] = []
    blocking_reasons: list[str] = []

    for flag in raw_flags:
        # Strip any colon-suffixed detail (e.g. "threat_mitigation_coverage_low:40%")
        base = flag.split(":")[0]
        if base in INFO_FLAGS or flag in INFO_FLAGS:
            info_notes.append(flag)
        elif base in AUTO_FIXABLE_FLAGS or flag in AUTO_FIXABLE_FLAGS:
            auto_fixes.append(flag)
        elif base in REVIEW_FLAGS or flag in REVIEW_FLAGS:
            review_reasons.append(flag)
        elif base in BLOCKING_FLAGS or flag in BLOCKING_FLAGS:
            blocking_reasons.append(flag)
        else:
            review_reasons.append(flag)

    return {
        "info_notes": info_notes,
        "auto_fixes": auto_fixes,
        "review_reasons": review_reasons,
        "blocking_reasons": blocking_reasons,
    }


def compute_candidate_readiness(
    candidate_dict: dict,
    gate_status: dict,
    repair_history: list[dict],
) -> dict:
    """Compute a structured user-facing readiness summary.

    Args:
        candidate_dict:  ComposedCandidate.model_dump() or equivalent dict.
        gate_status:     Final gate status from validate_candidate_export_contract().
        repair_history:  Repair entries from repair_candidate().

    Returns a dict with keys:
        readiness              : "ready" | "ready_after_auto_fix" | "needs_review" | "blocked"
        data_status            : "ok" | "failed"
        automation_status      : "full" | "partial" | "blocked"
        identification_status  : descriptive string
        auto_fix_actions       : list of action names applied by the repair loop
        user_visible_reasons   : blocking + review reasons only (no info/auto-fix noise)
        debug_flags            : raw reasons list for diagnostic traces
    """
    blocking = list(gate_status.get("blocking_reasons") or [])
    shortlist = gate_status.get("shortlist_status", "blocked")

    auto_fix_actions = [
        h["action"] for h in repair_history
        if h.get("result") in ("repaired", "normalized", "canonicalized")
    ]

    if blocking or shortlist == "blocked":
        readiness = "blocked"
    elif shortlist == "review":
        readiness = "needs_review"
    elif auto_fix_actions:
        readiness = "ready_after_auto_fix"
    else:
        readiness = "ready"

    # Classify raw reasons for the user_visible_reasons field; translate to human-readable
    raw_reasons: list[str] = gate_status.get("reasons") or []
    classified = classify_flags(raw_reasons)
    user_visible_reasons = [
        _translate_flag(f)
        for f in classified["blocking_reasons"] + classified["review_reasons"]
    ]

    # Identification quality
    claim_strength = candidate_dict.get("claim_strength", "associational")
    threats = list(candidate_dict.get("key_threats") or [])
    mitigations = dict(candidate_dict.get("mitigations") or {})
    threats_complete = len(threats) >= 3 and all(t in mitigations for t in threats)

    if claim_strength == "causal":
        identification_status = (
            "documented_causal" if threats_complete else "causal_claim_underdocumented"
        )
    else:
        identification_status = (
            "documented_associational" if threats_complete else "associational_threats_partial"
        )

    # Automation reachability
    risk = candidate_dict.get("automation_risk", "low")
    required_secrets = (
        gate_status.get("required_secrets")
        or candidate_dict.get("required_secrets")
        or []
    )
    if blocking or shortlist == "blocked" or risk == "high":
        automation_status = "blocked"
    elif risk == "medium" or required_secrets:
        automation_status = "partial"
    else:
        automation_status = "full"

    # Data path health
    data_ok = not any(
        (flag.split(":")[0] in _DATA_FAIL_FLAGS or flag in _DATA_FAIL_FLAGS)
        for flag in raw_reasons
    )

    return {
        "readiness": readiness,
        "data_status": "ok" if data_ok else "failed",
        "automation_status": automation_status,
        "identification_status": identification_status,
        "auto_fix_actions": auto_fix_actions,
        "user_visible_reasons": user_visible_reasons,
        "debug_flags": raw_reasons,
    }
