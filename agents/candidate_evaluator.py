from __future__ import annotations

import difflib
from datetime import datetime, timezone
from typing import Any

# Reasons that represent successful normalization, not real data-availability risk.
# These must not degrade the overall verdict or count as repairable warnings.
_INFO_FLAGS: frozenset[str] = frozenset({
    "source_alias_resolved",
    "non_canonical_source_name",
    "canonicalize_source_name",
    "partial_registry_match",   # alias resolution success; not a real data risk
})

from agents import settings
from agents.data_accessibility import evaluate_data_sources, summarize_data_access
from agents.openalex_utils import search_openalex
from agents.logging_config import get_logger
from agents.source_registry import SourceRegistry
from models.candidate_schema import CandidateEvaluation
from models.research_plan_schema import ResearchPlan

logger = get_logger(__name__)

# Lazy-loaded singleton; shared across all evaluate_candidate calls within a process.
_REGISTRY: SourceRegistry | None = None


def _get_registry() -> SourceRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SourceRegistry.load()
    return _REGISTRY


def _data_registry_verdict(plan: ResearchPlan) -> tuple[str, list[str], dict]:
    """Check each declared source against SourceRegistry using priority-ordered matching.

    Match priority:
      1. Exact source_id lookup (registry.resolve)
      2. Alias / canonical-name lookup (also via registry.resolve)
      3. Fuzzy string match against alias_to_id keys (threshold 0.65)

    Unresolved sources are flagged as data risks; alias resolutions are info-only.
    """
    registry = _get_registry()

    if not plan.data_sources:
        return "fail", ["no_data_sources_declared"], {"registry_matches": []}

    match_rows: list[dict[str, Any]] = []
    matched = 0
    alias_resolved = 0

    for source in plan.data_sources:
        name = (source.name or "").strip()
        canonical = registry.resolve(name)
        is_exact = canonical == name  # exact source_id hit

        if canonical is None:
            # Fuzzy fallback against all known aliases
            key = name.lower()
            best_ratio = 0.0
            best_alias: str | None = None
            for alias in registry.alias_to_id:
                ratio = difflib.SequenceMatcher(None, key, alias).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_alias = alias
            if best_ratio >= 0.65 and best_alias:
                canonical = registry.alias_to_id[best_alias]
                alias_resolved += 1

        is_match = canonical is not None
        if is_match:
            matched += 1

        match_rows.append({
            "source_name": name,
            "matched_registry_name": canonical or "",
            "matched": is_match,
            "exact": is_exact,
        })

    reasons: list[str] = []
    if matched == len(plan.data_sources):
        verdict = "pass"
        if alias_resolved > 0:
            # All sources resolved but some needed alias lookup — purely informational
            reasons = ["partial_registry_match"]
    elif matched == 0:
        verdict = "fail"
        reasons = ["source_not_in_registry"]
    else:
        verdict = "warning"
        reasons = ["partial_registry_match"]

    return verdict, reasons, {"registry_matches": match_rows}


def _identification_verdict(plan: ResearchPlan) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not plan.identification.primary_method.strip():
        reasons.append("missing_identification_method")
    if not plan.identification.key_threats:
        reasons.append("missing_identification_threats")
    if reasons:
        return "warning", reasons
    return "pass", []


def _contribution_verdict(plan: ResearchPlan) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if len(plan.short_rationale.strip()) < 20:
        reasons.append("short_rationale_too_brief")
    if len(plan.research_question.strip()) < 15:
        reasons.append("research_question_too_brief")
    if reasons:
        return "warning", reasons
    return "pass", []


def _novelty_verdict(
    llm: Any,
    plan: ResearchPlan,
    years: int = 2,
) -> tuple[str, list[str], dict[str, Any]]:
    from_year = datetime.now(timezone.utc).year - years
    query = plan.project_title
    papers = search_openalex(query, limit=8, from_year=from_year)
    if not papers:
        return "unknown", ["novelty_evidence_limited"], {"query": query, "papers": []}

    title_key = "".join(c for c in plan.project_title.lower() if c.isalnum())
    best = 0.0
    for paper in papers:
        paper_title = str(paper.get("title") or "")
        paper_key = "".join(c for c in paper_title.lower() if c.isalnum())
        if not paper_key:
            continue
        best = max(best, difflib.SequenceMatcher(None, title_key, paper_key).ratio())

    if best >= 0.85:
        verdict = "already_published"
        reasons = ["already_published_overlap"]
    elif best >= 0.65:
        verdict = "partially_overlapping"
        reasons = ["partial_literature_overlap"]
    else:
        verdict = "novel"
        reasons = []
    evidence = {
        "query": query,
        "max_similarity": round(best, 3),
        "papers": papers[:5],
    }
    return verdict, reasons, evidence


def evaluate_candidate(
    candidate: dict[str, Any],
    plan: ResearchPlan,
    llm: Any = None,
) -> CandidateEvaluation:
    candidate_id = str(
        candidate.get("topic_id")
        or candidate.get("reflection_trace_id")
        or candidate.get("candidate_id")
        or f"candidate_{candidate.get('rank', 0)}"
    )
    title = str(candidate.get("title") or plan.project_title)

    schema_valid = True
    data_registry_verdict, registry_reasons, registry_evidence = _data_registry_verdict(plan)
    checks = evaluate_data_sources(plan)
    data_access_verdict, data_access_reasons = summarize_data_access(checks)
    identification_verdict, identification_reasons = _identification_verdict(plan)
    contribution_verdict, contribution_reasons = _contribution_verdict(plan)
    novelty_verdict, novelty_reasons, novelty_evidence = _novelty_verdict(llm, plan)

    reasons = (
        registry_reasons
        + data_access_reasons
        + identification_reasons
        + contribution_reasons
        + novelty_reasons
    )

    verdict_bucket = {
        data_registry_verdict,
        data_access_verdict,
        identification_verdict,
        contribution_verdict,
    }
    if "fail" in verdict_bucket:
        overall = "fail"
    elif "already_published" == novelty_verdict:
        overall = "warning"
    elif "warning" in verdict_bucket or novelty_verdict == "partially_overlapping":
        overall = "warning"
    else:
        overall = "pass"

    base_score = {"pass": 1.0, "warning": 0.6, "fail": 0.2}[overall]
    penalty_reasons = [r for r in reasons if r not in _INFO_FLAGS]
    score = max(0.0, min(1.0, base_score - 0.05 * len(penalty_reasons)))

    return CandidateEvaluation(
        candidate_id=candidate_id,
        title=title,
        rank=int(candidate.get("rank") or 0),
        schema_valid=schema_valid,
        data_registry_verdict=data_registry_verdict,
        data_access_verdict=data_access_verdict,
        novelty_verdict=novelty_verdict,
        identification_verdict=identification_verdict,
        contribution_verdict=contribution_verdict,
        overall_verdict=overall,
        score=round(score, 3),
        reasons=reasons,
        evidence={
            **registry_evidence,
            "data_access_checks": [c.model_dump() for c in checks],
            "novelty": novelty_evidence,
        },
    )
