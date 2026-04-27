from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from typing import Any

from agents import settings
from agents.data_accessibility import evaluate_data_sources, summarize_data_access
from agents.openalex_utils import search_openalex
from agents.logging_config import get_logger
from models.candidate_schema import CandidateEvaluation
from models.research_plan_schema import ResearchPlan

logger = get_logger(__name__)


def _load_registry() -> list[dict[str, Any]]:
    path = settings.data_source_registry_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logger.warning("Data source registry unavailable at %s", path)
        return []


def _registry_match_score(source_name: str, registry_entry: dict[str, Any]) -> float:
    candidates = [registry_entry.get("canonical_name", "")] + list(registry_entry.get("aliases", []))
    source = (source_name or "").strip().lower()
    best = 0.0
    for candidate in candidates:
        score = difflib.SequenceMatcher(None, source, str(candidate).lower()).ratio()
        best = max(best, score)
    return best


def _data_registry_verdict(plan: ResearchPlan, threshold: float = 0.6) -> tuple[str, list[str], dict]:
    registry = _load_registry()
    if not plan.data_sources:
        return "fail", ["no_data_sources_declared"], {"registry_matches": []}
    if not registry:
        return "warning", ["registry_unavailable"], {"registry_matches": []}

    match_rows: list[dict[str, Any]] = []
    matched = 0
    for source in plan.data_sources:
        best_score = 0.0
        best_name = ""
        for entry in registry:
            score = _registry_match_score(source.name, entry)
            if score > best_score:
                best_score = score
                best_name = str(entry.get("canonical_name", ""))
        is_match = best_score >= threshold
        if is_match:
            matched += 1
        match_rows.append(
            {
                "source_name": source.name,
                "matched_registry_name": best_name if is_match else "",
                "score": round(best_score, 3),
                "matched": is_match,
            }
        )

    if matched == len(plan.data_sources):
        verdict = "pass"
        reasons: list[str] = []
    elif matched == 0:
        verdict = "fail"
        reasons = ["source_not_in_registry"]
    else:
        verdict = "warning"
        reasons = ["source_alias_resolved"]
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
    score = max(0.0, min(1.0, base_score - 0.05 * len(reasons)))

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
