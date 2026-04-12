"""
IdeaValidatorAgent: validates top research ideas for novelty (via OpenAlex
recent-paper search) and data availability (via registry matching).
Auto-substitutes failed ideas from the backup candidate pool.

Environment variables (all optional):
  NOVELTY_CHECK_YEARS             – how many years back to search (default 2)
  NOVELTY_QUERIES_PER_IDEA        – search queries per idea (default 3)
  NOVELTY_RESULTS_PER_QUERY       – papers per query (default 10)
  DATA_REGISTRY_FUZZY_THRESHOLD   – fuzzy match cutoff 0-1 (default 0.6)
  VALIDATION_MAX_SUBSTITUTIONS    – max substitute rounds (default 2)
"""

from __future__ import annotations

import difflib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from agents import settings
from agents.logging_config import get_logger
from agents.memory_retriever import MemoryRetriever
from agents.openalex_utils import search_openalex
from agents.orchestrator import ResearchState

logger = get_logger(__name__)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class NoveltySearchQueries(BaseModel):
    """LLM-generated targeted search queries for novelty checking."""
    queries: List[str] = Field(
        description=(
            "2-3 short, precise English search queries (4-8 words each) "
            "designed to find existing papers that overlap with this research idea. "
            "Focus on the specific method + data + phenomenon combination."
        )
    )


class SimilarPaper(BaseModel):
    title: str = Field(description="Title of the potentially overlapping paper.")
    year: int = Field(description="Publication year.")
    doi: str | None = Field(default=None, description="DOI if available.")
    similarity_verdict: str = Field(
        description="One of: highly_similar, partially_similar, different"
    )
    overlap_explanation: str = Field(
        description="Brief explanation of how this paper relates to the idea."
    )


class NoveltyAssessment(BaseModel):
    """LLM assessment of novelty given a set of candidate papers."""
    verdict: str = Field(
        description="One of: novel, partially_overlapping, already_published"
    )
    similar_papers: List[SimilarPaper] = Field(
        description="Papers with non-trivial overlap, ordered by similarity."
    )


class NoveltyResult(BaseModel):
    verdict: str
    similar_papers: List[Dict[str, Any]]
    search_queries_used: List[str]
    was_llm_fallback: bool = False  # True when LLM was unavailable or failed


class DataSourceCheck(BaseModel):
    name: str
    registry_match: str | None
    status: str  # "verified" / "unverified"


class IdeaValidation(BaseModel):
    title: str
    rank: int
    brief_rationale: str = ""
    novelty: NoveltyResult
    data_availability: List[DataSourceCheck]
    overall_verdict: str  # "passed" / "failed" / "warning"
    failure_reasons: List[str]


class ValidationReport(BaseModel):
    run_id: str
    validated_ideas: List[IdeaValidation]
    substitutions_made: int
    final_top3_titles: List[str]


# ---------------------------------------------------------------------------
# Data source registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> List[Dict[str, Any]]:
    """Load the data source registry from disk."""
    registry_path = settings.data_source_registry_path()
    if not os.path.exists(registry_path):
        logger.warning("Data source registry not found at %s", registry_path)
        return []
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _match_data_source(
    name: str,
    registry: List[Dict[str, Any]],
    threshold: float = 0.6,
) -> str | None:
    """
    Fuzzy-match a claimed data source name against the registry.
    Returns the canonical_name if matched, else None.
    """
    name_lower = name.strip().lower()
    best_score = 0.0
    best_match: str | None = None

    for entry in registry:
        candidates = [entry["canonical_name"]] + entry.get("aliases", [])
        for candidate in candidates:
            score = difflib.SequenceMatcher(
                None, name_lower, candidate.lower()
            ).ratio()
            if score > best_score:
                best_score = score
                best_match = entry["canonical_name"]

    if best_score >= threshold:
        return best_match
    return None


def check_data_availability(
    data_sources: List[Dict[str, Any]],
    registry: List[Dict[str, Any]],
    threshold: float = 0.6,
) -> List[DataSourceCheck]:
    """Check each claimed data source against the registry."""
    results = []
    for source in data_sources:
        name = source.get("name", source.get("source", "Unknown"))
        match = _match_data_source(name, registry, threshold)
        results.append(DataSourceCheck(
            name=name,
            registry_match=match,
            status="verified" if match else "unverified",
        ))
    return results


# ---------------------------------------------------------------------------
# Novelty check helpers
# ---------------------------------------------------------------------------

def _generate_novelty_queries(
    llm: Any,
    title: str,
    rationale: str,
    n_queries: int = 3,
) -> List[str]:
    """Use LLM to generate targeted search queries for novelty checking."""
    if llm is None or not isinstance(llm, ChatGoogleGenerativeAI):
        # Fallback: use title words as query
        return [title]

    structured_llm = llm.with_structured_output(NoveltySearchQueries)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
    def _call():
        return structured_llm.invoke(
            f"Generate {n_queries} precise search queries to find existing published "
            f"papers that may overlap with this research idea.\n\n"
            f"Title: {title}\n"
            f"Rationale: {rationale}\n\n"
            f"The queries should target the specific combination of method, data, "
            f"and phenomenon. Use short English phrases (4-8 words), no boolean operators."
        )

    try:
        result = _call()
        return result.queries[:n_queries]
    except Exception as e:
        logger.warning("Novelty query generation failed: %s; using title as fallback", e)
        return [title]


def _assess_novelty(
    llm: Any,
    idea_title: str,
    idea_rationale: str,
    papers: List[Dict[str, Any]],
) -> tuple["NoveltyAssessment", bool]:
    """Use LLM to assess novelty given a set of retrieved papers.

    Returns (assessment, was_fallback) where was_fallback is True when the LLM
    was unavailable or threw an exception and a safe default was used instead.
    """
    if llm is None or not isinstance(llm, ChatGoogleGenerativeAI):
        return NoveltyAssessment(verdict="novel", similar_papers=[]), True

    if not papers:
        return NoveltyAssessment(verdict="novel", similar_papers=[]), False

    papers_text = "\n".join(
        f"- [{p.get('year', '?')}] {p.get('title', 'No title')} "
        f"(cited: {p.get('citationCount', 0)}, doi: {p.get('doi', 'N/A')})\n"
        f"  Abstract: {(p.get('abstract') or 'N/A')[:300]}"
        for p in papers[:15]  # limit to avoid token overflow
    )

    structured_llm = llm.with_structured_output(NoveltyAssessment)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
    def _call():
        return structured_llm.invoke(
            f"You are a senior research evaluator. Compare this research idea against "
            f"recently published papers and assess its novelty.\n\n"
            f"RESEARCH IDEA:\n"
            f"Title: {idea_title}\n"
            f"Rationale: {idea_rationale}\n\n"
            f"RECENTLY PUBLISHED PAPERS:\n{papers_text}\n\n"
            f"Determine the verdict:\n"
            f"- 'novel': No paper substantially overlaps with this idea's core contribution\n"
            f"- 'partially_overlapping': Some papers address similar questions but with "
            f"different methods, data, or scope — the idea still has a publishable gap\n"
            f"- 'already_published': A paper already covers the same question with similar "
            f"method and data — minimal remaining gap\n\n"
            f"List papers with non-trivial overlap (highly_similar or partially_similar)."
        )

    try:
        return _call(), False
    except Exception as e:
        logger.warning("Novelty assessment failed: %s; defaulting to novel", e)
        return NoveltyAssessment(verdict="novel", similar_papers=[]), True


def check_novelty(
    llm: Any,
    title: str,
    rationale: str,
    from_year: int,
    n_queries: int = 3,
    results_per_query: int = 10,
) -> NoveltyResult:
    """Full novelty check: generate queries → search OpenAlex → assess."""
    queries = _generate_novelty_queries(llm, title, rationale, n_queries)
    logger.info("Novelty queries for '%s': %s", title[:50], queries)

    all_papers: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for query in queries:
        try:
            papers = search_openalex(query, limit=results_per_query, from_year=from_year)
            for p in papers:
                pid = p.get("openalex_id") or p.get("doi") or p.get("title", "")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)
        except Exception as e:
            logger.warning("OpenAlex search failed for query '%s': %s", query, e)

    logger.info("Found %d unique recent papers for '%s'", len(all_papers), title[:50])

    assessment, was_fallback = _assess_novelty(llm, title, rationale, all_papers)

    return NoveltyResult(
        verdict=assessment.verdict,
        similar_papers=[sp.model_dump() for sp in assessment.similar_papers],
        search_queries_used=queries,
        was_llm_fallback=was_fallback,
    )


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class IdeaValidatorAgent:
    def __init__(self):
        fast_model = os.getenv("GEMINI_FAST_MODEL", "gemini-2.0-flash-lite")
        self.llm = None
        self._degraded_nodes: list[str] = []
        if ChatGoogleGenerativeAI is not None:
            try:
                self.llm = ChatGoogleGenerativeAI(model=fast_model, temperature=0.1)
            except Exception as e:
                logger.warning("Could not init LLM for validation: %s", e)
                self._degraded_nodes.append("idea_validator:llm_unavailable")

        self.memory = MemoryRetriever()

    def _validate_idea(
        self,
        idea: Dict[str, Any],
        registry: List[Dict[str, Any]],
        from_year: int,
        n_queries: int,
        results_per_query: int,
        fuzzy_threshold: float,
    ) -> IdeaValidation:
        """Validate a single idea for novelty and data availability."""
        title = idea.get("title", "")
        rationale = idea.get("brief_rationale", "")
        rank = idea.get("rank", 0)

        # Novelty check
        logger.info("Checking novelty for: %s", title[:70])
        novelty = check_novelty(
            self.llm, title, rationale, from_year, n_queries, results_per_query
        )
        if novelty.was_llm_fallback:
            tag = f"idea_validator:novelty_fallback:{title[:40]}"
            if tag not in self._degraded_nodes:
                self._degraded_nodes.append(tag)
                logger.warning("LLM novelty assessment fell back to default for: %s", title[:70])

        # Data availability check
        data_sources = idea.get("data_sources", [])
        data_checks = check_data_availability(data_sources, registry, fuzzy_threshold)

        # Determine verdict
        failure_reasons: List[str] = []
        if novelty.verdict == "already_published":
            failure_reasons.append(
                "Idea appears already published — similar paper(s) found in recent literature"
            )
        all_unverified = (
            len(data_checks) > 0
            and all(dc.status == "unverified" for dc in data_checks)
        )
        if all_unverified:
            failure_reasons.append(
                "All claimed data sources are unverified — not found in known public data registry"
            )

        if failure_reasons:
            overall = "failed"
        elif novelty.verdict == "partially_overlapping" or any(
            dc.status == "unverified" for dc in data_checks
        ):
            overall = "warning"
        else:
            overall = "passed"

        return IdeaValidation(
            title=title,
            rank=rank,
            brief_rationale=rationale,
            novelty=novelty,
            data_availability=[dc.model_dump() for dc in data_checks],
            overall_verdict=overall,
            failure_reasons=failure_reasons,
        )

    def run(self, state: ResearchState) -> ResearchState:
        logger.info("--- Idea Validator: Checking novelty & data availability ---")

        # Load config
        from_year = datetime.now(timezone.utc).year - int(
            os.getenv("NOVELTY_CHECK_YEARS", "2")
        )
        n_queries = int(os.getenv("NOVELTY_QUERIES_PER_IDEA", "3"))
        results_per_query = int(os.getenv("NOVELTY_RESULTS_PER_QUERY", "10"))
        fuzzy_threshold = float(os.getenv("DATA_REGISTRY_FUZZY_THRESHOLD", "0.6"))
        max_subs = int(os.getenv("VALIDATION_MAX_SUBSTITUTIONS", "2"))

        # Load screening data
        screening_path = state.get("candidate_topics_path", settings.topic_screening_path())
        if not os.path.exists(screening_path):
            logger.warning("No topic_screening.json found; skipping validation.")
            return {
                "validation_report_path": settings.idea_validation_path(),
                "execution_status": "harvesting",
            }

        with open(screening_path, "r", encoding="utf-8") as f:
            screening_data = json.load(f)

        candidates = screening_data.get("candidates", [])
        backup_pool = list(screening_data.get("backup_candidates", []))
        run_id = screening_data.get("run_id", "unknown")
        domain = state.get("domain_input", "")

        # Load registry
        registry = _load_registry()

        # Validate each candidate
        validated: List[IdeaValidation] = []
        substitutions_made = 0

        for idea in candidates:
            validation = self._validate_idea(
                idea, registry, from_year, n_queries, results_per_query, fuzzy_threshold
            )
            if validation.overall_verdict == "failed" and backup_pool and substitutions_made < max_subs:
                logger.info(
                    "Idea '%s' failed validation (%s); substituting...",
                    validation.title[:50],
                    ", ".join(validation.failure_reasons),
                )
                # Record failure in memory
                self._record_failure(validation, domain, run_id, screening_path)
                # Add failed validation to report
                validated.append(validation)

                # Substitute
                substitute = backup_pool.pop(0)
                substitute["rank"] = idea.get("rank", 0)
                sub_validation = self._validate_idea(
                    substitute, registry, from_year, n_queries, results_per_query, fuzzy_threshold
                )
                # Replace in candidates list
                idx = candidates.index(idea)
                candidates[idx] = substitute
                validated.append(sub_validation)
                substitutions_made += 1
            else:
                if validation.overall_verdict == "failed":
                    self._record_failure(validation, domain, run_id, screening_path)
                validated.append(validation)

        # Rebuild final top 3 (only non-failed)
        final_titles = [
            v.title for v in validated
            if v.overall_verdict in ("passed", "warning")
        ]

        # Save validation report
        report = ValidationReport(
            run_id=run_id,
            validated_ideas=validated,
            substitutions_made=substitutions_made,
            final_top3_titles=final_titles[:3],
        )
        validation_path = settings.idea_validation_path()
        os.makedirs(os.path.dirname(validation_path), exist_ok=True)
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("Validation report saved to %s", validation_path)

        # Update screening if substitutions were made
        if substitutions_made > 0:
            screening_data["candidates"] = candidates
            screening_data["backup_candidates"] = backup_pool
            with open(screening_path, "w", encoding="utf-8") as f:
                json.dump(screening_data, f, indent=2, ensure_ascii=False)
            logger.info("Updated topic_screening.json with %d substitution(s)", substitutions_made)

            # Regenerate research_plan.json for the new top-1
            plan_path = state.get("current_plan_path", settings.research_plan_path())
            self._update_plan_if_needed(plan_path, candidates)

        return {
            "validation_report_path": validation_path,
            "execution_status": "harvesting",
            "degraded_nodes": self._degraded_nodes,
        }

    def _record_failure(
        self,
        validation: IdeaValidation,
        domain: str,
        run_id: str,
        screening_path: str,
    ) -> None:
        """Write failed idea to memory for future avoidance."""
        status = "failed_novelty_check" if any(
            "already published" in r.lower() for r in validation.failure_reasons
        ) else "failed_data_check"
        try:
            self.memory.store_idea(
                topic=validation.title,
                domain=domain,
                status=status,
                rejection_reason="; ".join(validation.failure_reasons),
                metadata={
                    "run_id": run_id,
                    "similar_papers": [
                        sp.get("title", "") for sp in validation.novelty.similar_papers[:3]
                    ],
                    "unverified_sources": [
                        (dc.name if isinstance(dc, DataSourceCheck) else dc.get("name", ""))
                        for dc in validation.data_availability
                        if (dc.status if isinstance(dc, DataSourceCheck) else dc.get("status")) == "unverified"
                    ],
                },
                source_file=screening_path,
            )
        except Exception as e:
            logger.warning("Failed to record validation failure in memory: %s", e)

    def _update_plan_if_needed(
        self,
        plan_path: str,
        candidates: List[Dict[str, Any]],
    ) -> None:
        """Update research_plan.json title if top-1 changed."""
        if not candidates or not os.path.exists(plan_path):
            return
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            new_top1 = candidates[0]
            plan["project_title"] = new_top1.get("title", plan.get("project_title", ""))
            plan["topic_screening"] = {
                "top_candidate_title": new_top1.get("title", ""),
                "validation_status": "substituted",
            }
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
            logger.info("Updated research plan for substituted top-1")
        except Exception as e:
            logger.warning("Could not update research plan: %s", e)


def idea_validator_node(state: ResearchState) -> ResearchState:
    agent = IdeaValidatorAgent()
    return agent.run(state)
