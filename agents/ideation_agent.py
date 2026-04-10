import json
import os
import csv
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential

from agents import settings
from agents.logging_config import get_logger
from agents.memory_retriever import MemoryRetriever
from agents.orchestrator import ResearchState
from agents.field_scanner_agent import summarize_field_scan
from agents.cache_utils import build_cache_key, load_json_cache, save_json_cache
from agents.keyword_planner import KeywordPlanner
from agents.openalex_utils import multi_search_openalex

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1 schemas: lightweight generation only
# ---------------------------------------------------------------------------

class LightCandidateTopic(BaseModel):
    title: str = Field(description="The formal title of the research topic.")
    brief_rationale: str = Field(
        description=(
            "A concise 1-2 sentence rationale explaining why this topic is impactful, novel, "
            "and feasible with public quantitative data. Hint at method type, data source, and gap filled."
        )
    )


class LightCandidateTopicsList(BaseModel):
    candidates: List[LightCandidateTopic]


# ---------------------------------------------------------------------------
# Screening schemas
# ---------------------------------------------------------------------------

class TopicScore(BaseModel):
    title: str = Field(description="Title of the topic being scored.")
    score: int = Field(description="Overall quality score out of 100.")
    passed_gates: bool = Field(
        description="Whether the topic passes all hard gates (impact, quantitative, novelty, etc.)."
    )
    rejection_reason: str = Field(
        description="If passed_gates is false, strictly explain which gate it failed and why. Otherwise, empty string.",
        default="",
    )
    rank: int = Field(
        description="Final ranking position among passed topics (1 = best). Set to 0 if passed_gates is false.",
        default=0,
    )


class TopicScoresList(BaseModel):
    scores: List[TopicScore]


# ---------------------------------------------------------------------------
# Enrichment schemas
# ---------------------------------------------------------------------------

class QuantitativeSpecs(BaseModel):
    unit_of_analysis: str = Field(description="Sample unit and spatial/time scale.")
    outcomes: List[str] = Field(description="Clear dependent variables constructable from public data.")
    exposures: List[str] = Field(description="Clear core explanatory/treatment variables.")
    estimand_and_strategy: str = Field(description="Estimand and identification strategy/assumptions.")
    model_family: str = Field(description="Specific, scriptable model family.")
    robustness_checks: List[str] = Field(description="At least 6 scriptable robustness/heterogeneity checks.")
    expected_tables_figures: List[str] = Field(description="Expected table/figure types and required statistics.")


class DataSource(BaseModel):
    name: str = Field(description="Name of the public data source.")
    accessibility: str = Field(description="Why this data is freely accessible without paywalls.")


class RawCandidateTopic(BaseModel):
    title: str = Field(description="The formal title of the research topic (keep exactly as given).")
    impact_evidence: str = Field(description="Realistic justification of impact.")
    novelty_gap_type: str = Field(description="Specific type of gap this fills.")
    publishability: str = Field(description="2-3 specific target journals with brief matching justification.")
    quantitative_specs: QuantitativeSpecs
    data_sources: List[DataSource]


class ResearchPlanSchema(BaseModel):
    project_title: str
    study_type: str
    topic_screening: dict
    research_questions: List[str]
    hypotheses: List[str]
    unit_of_analysis: str
    outcomes: List[dict]
    exposures: List[dict]
    keywords: List[str]
    data_sources: List[dict]
    methodology: dict


class NoveltyQueryPlan(BaseModel):
    queries: List[str] = Field(description="2-3 precise OpenAlex queries for novelty check.")


class PaperOverlapAssessment(BaseModel):
    paper_openalex_id: str = Field(description="OpenAlex ID of the paper being assessed.")
    overlap_score: int = Field(description="0-100 overlap score. 100 means almost identical research contribution.")
    rationale: str = Field(description="Short explanation of overlap judgment.")


class NoveltyAssessment(BaseModel):
    novelty_verdict: str = Field(description="One of: novel, partially_overlapping, already_published.")
    assessments: List[PaperOverlapAssessment] = Field(description="Per-paper overlap assessments.")


# ---------------------------------------------------------------------------
# Long-term archive helper
# ---------------------------------------------------------------------------

def _persist_enriched_top3(
    enriched_top3: List[Dict[str, Any]],
    domain: str,
    run_id: str,
    archive_path: str | None = None,
) -> None:
    """Append-only JSONL archive of enriched top-3 candidates for long-term review."""
    if archive_path is None:
        archive_path = settings.enriched_top_candidates_path()
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with open(archive_path, "a", encoding="utf-8") as f:
            for candidate in enriched_top3:
                record = {
                    "created_at": created_at,
                    "domain": domain,
                    "run_id": run_id,
                    "rank": candidate.get("rank"),
                    "title": candidate.get("title", ""),
                    "brief_rationale": candidate.get("brief_rationale", ""),
                    "initial_score": candidate.get("initial_score"),
                    "final_score": candidate.get("final_score"),
                    "impact_evidence": candidate.get("impact_evidence", ""),
                    "novelty_gap_type": candidate.get("novelty_gap_type", ""),
                    "publishability": candidate.get("publishability", ""),
                    "quantitative_specs": candidate.get("quantitative_specs", {}),
                    "data_sources": candidate.get("data_sources", []),
                    "source_snapshot": settings.topic_screening_path(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Enriched top-3 appended to %s", archive_path)
    except Exception as e:
        logger.warning("Failed to append enriched top-3 to archive: %s", e)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class IdeationAgent:
    def __init__(self, use_strict_models: bool = False):
        fast_model = os.getenv("GEMINI_FAST_MODEL", "gemini-2.0-flash-lite")
        pro_model = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
        self.fast_llm = ChatGoogleGenerativeAI(model=fast_model, temperature=0.7)
        self.pro_llm = ChatGoogleGenerativeAI(model=pro_model, temperature=0.2)
        self.memory = MemoryRetriever()
        self.keyword_planner = KeywordPlanner()
        self._degraded_nodes: list[str] = []

    def _generate_novelty_queries(self, candidate: Dict[str, Any], domain: str) -> List[str]:
        title = str(candidate.get("title", "")).strip()
        rationale = str(candidate.get("brief_rationale", "")).strip()
        base = f"{title}. {rationale}".strip()

        try:
            if self.fast_llm:
                structured_llm = self.fast_llm.with_structured_output(NoveltyQueryPlan)
                prompt = (
                    "Generate exactly 2-3 precise OpenAlex search queries for novelty check.\n"
                    "Each query should be concise and targeted to detect prior near-identical studies.\n"
                    "Avoid Boolean operators and quotes.\n\n"
                    f"Domain: {domain}\n"
                    f"Idea title: {title}\n"
                    f"Idea summary: {base}\n"
                )
                result = structured_llm.invoke(prompt)
                queries = [q.strip() for q in result.queries if q.strip()]
                if queries:
                    return queries[:3]
        except Exception as exc:
            logger.warning("Novelty query generation via schema failed for '%s': %s", title, exc)

        try:
            planned = self.keyword_planner.plan(base or title or domain, extra_context=domain)
            query_pool = [q.strip() for q in planned.get("query_pool", []) if q and q.strip()]
            if query_pool:
                return query_pool[:3]
        except Exception as exc:
            logger.warning("KeywordPlanner fallback failed for '%s': %s", title, exc)

        fallback = [q for q in [title, f"{title} causal inference", domain] if q]
        return fallback[:3]

    def _run_novelty_check(
        self,
        candidate: Dict[str, Any],
        domain: str,
        months_back: int = 15,
    ) -> Dict[str, Any]:
        title = str(candidate.get("title", "")).strip()
        queries = self._generate_novelty_queries(candidate, domain)

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=30 * months_back)
        papers, _ = multi_search_openalex(
            queries,
            per_query_limit=int(os.getenv("NOVELTY_PER_QUERY_LIMIT", "8")),
            final_limit=int(os.getenv("NOVELTY_FINAL_LIMIT", "10")),
            cache_namespace="openalex_novelty",
            cache_prefix="openalex_novelty_query",
            from_publication_date=start_date.isoformat(),
            to_publication_date=end_date.isoformat(),
        )

        if not papers:
            return {
                "novelty_verdict": "novel",
                "novelty_queries": queries,
                "novelty_date_window": {
                    "from_publication_date": start_date.isoformat(),
                    "to_publication_date": end_date.isoformat(),
                },
                "novelty_top_papers": [],
                "novelty_assessment_notes": "No recent OpenAlex results; defaulted to novel.",
            }

        shortlist = papers[: min(len(papers), 6)]
        paper_payload = [
            {
                "paper_openalex_id": p.get("openalex_id", ""),
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "year": p.get("year"),
                "doi": p.get("doi"),
            }
            for p in shortlist
        ]

        assessment: NoveltyAssessment | None = None
        try:
            if self.pro_llm:
                structured_llm = self.pro_llm.with_structured_output(NoveltyAssessment)
                prompt = (
                    "You are evaluating novelty risk for a research idea.\n"
                    "Compare the idea with each candidate paper (title + abstract) and judge substantive overlap.\n"
                    "Return one verdict in {novel, partially_overlapping, already_published}.\n\n"
                    f"Idea title: {title}\n"
                    f"Idea rationale: {candidate.get('brief_rationale', '')}\n"
                    f"Candidate papers: {json.dumps(paper_payload, ensure_ascii=False)}\n"
                )
                assessment = structured_llm.invoke(prompt)
        except Exception as exc:
            logger.warning("Novelty assessment LLM failed for '%s': %s", title, exc)

        if assessment is None:
            self._degraded_nodes.append(f"ideation:novelty_fallback:{title[:40]}")
            logger.warning("Novelty assessment LLM unavailable for '%s'; using fallback verdict.", title)
            return {
                "novelty_verdict": "partially_overlapping",
                "novelty_queries": queries,
                "novelty_date_window": {
                    "from_publication_date": start_date.isoformat(),
                    "to_publication_date": end_date.isoformat(),
                },
                "novelty_top_papers": paper_payload[:2],
                "novelty_assessment_notes": "Fallback verdict due to LLM assessment failure.",
            }

        scored = sorted(assessment.assessments, key=lambda x: x.overlap_score, reverse=True)
        best_ids = [s.paper_openalex_id for s in scored[:2] if s.paper_openalex_id]
        top_refs = [p for p in paper_payload if p.get("paper_openalex_id") in best_ids]
        if not top_refs:
            top_refs = paper_payload[:2]

        return {
            "novelty_verdict": assessment.novelty_verdict,
            "novelty_queries": queries,
            "novelty_date_window": {
                "from_publication_date": start_date.isoformat(),
                "to_publication_date": end_date.isoformat(),
            },
            "novelty_top_papers": top_refs,
            "novelty_assessment_notes": [a.model_dump() for a in scored[:3]],
        }

    def run(self, state: ResearchState) -> ResearchState:
        domain = state.get("domain_input", "Urban Green Space and Inequality")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        logger.info("--- Module 1: Ideation & Design for '%s' ---", domain)

        # 0. Load Field Scan
        field_scan_path = state.get("field_scan_path", settings.field_scan_path())
        field_scan_context = "No previous field scan data available for constraints."
        field_scan_data: Dict[str, Any] | None = None
        if os.path.exists(field_scan_path):
            try:
                with open(field_scan_path, "r", encoding="utf-8") as f:
                    field_scan_data = json.load(f)
                    field_scan_context = json.dumps(
                        summarize_field_scan(field_scan_data), indent=2
                    )
            except Exception as e:
                logger.warning("Could not read field scan: %s", e)

        # 1. Memory RAG (recent CSV + enriched archive + graveyard)
        memory_context = self.memory.build_prompt_context(
            domain=domain,
            enriched_jsonl_path=settings.enriched_top_candidates_path(),
            graveyard_path=settings.ideas_graveyard_path(domain=domain),
        )
        if (
            memory_context.get("summary", {}).get("recent_count", 0) == 0
            and memory_context.get("summary", {}).get("archive_count", 0) == 0
            and memory_context.get("summary", {}).get("rejected_count", 0) == 0
        ):
            past_context = "No prior memory for this domain."
        else:
            past_context = json.dumps(memory_context, indent=2, ensure_ascii=False)

        # Step 1: Lightweight idea generation
        logger.info("Scouting field and generating candidate topics...")

        generation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert academic research designer. Using the current 'Field Scan' data as strict "
                "constraints and realistic grounding, generate 30 highly impactful, novel, and quantitative "
                "candidate research topics for the given domain.\n\n"
                "Field Scan Data (Current Trends & Highly Cited Papers):\n{field_scan_context}\n\n"
                "For EACH topic, internally consider (but do NOT output these details):\n"
                "- Impact: policy/industry/academic relevance\n"
                "- Novelty: specific gap filled (Problem/Measurement/Scale/Method/Reproducibility Gap)\n"
                "- Publishability: top-tier target journals\n"
                "- Quantitative feasibility: measurable outcomes and exposures, suitable model family\n"
                "- Data availability: publicly accessible, free datasets\n\n"
                "Historical Memory Context (recent selections/rejections + enriched archive):\n{past_ideas}\n\n"
                "Output ONLY: a formal title and a 1-2 sentence rationale hinting at method, data, and gap.\n"
                "Ensure HIGH DIVERSITY across spatial scales, methods, and data sources.\n"
                "Avoid repeating rejected ideas and near-duplicates from memory context."
            ),
            ("user", "Domain: {domain}"),
        ])

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def _generate_light_batch():
            if not self.fast_llm:
                raise ValueError("fast_llm not initialized")
            structured_llm = self.fast_llm.with_structured_output(LightCandidateTopicsList)
            prompt_value = generation_prompt.invoke(
                {"domain": domain, "past_ideas": past_context, "field_scan_context": field_scan_context}
            )
            result = structured_llm.invoke(prompt_value)
            return [c.model_dump() for c in result.candidates]

        target_candidates = int(os.getenv("MIN_CANDIDATE_TOPICS", "30"))
        logger.info("Calling LLM for topic generation (target >= %d topics)...", target_candidates)

        light_candidates: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = 6
        seen_titles: set[str] = set()

        while len(light_candidates) < target_candidates and attempts < max_attempts:
            attempts += 1
            try:
                batch = _generate_light_batch()
            except Exception as e:
                logger.warning("Generation batch %d failed: %s", attempts, e)
                continue
            for c in batch:
                title = str(c.get("title", "")).strip()
                key = title.lower()
                if not title or key in seen_titles:
                    continue
                seen_titles.add(key)
                light_candidates.append(c)

        logger.info("Step 1 complete: %d unique candidates after %d attempts.", len(light_candidates), attempts)
        if len(light_candidates) < target_candidates:
            logger.warning(
                "Requested %d topics but only got %d after %d attempts.",
                target_candidates, len(light_candidates), attempts,
            )

        screening_path = settings.topic_screening_path()
        plan_path = settings.research_plan_path()
        context_path = settings.research_context_path()

        if not light_candidates:
            logger.warning("No candidates generated; aborting ideation pipeline.")
            return {
                "execution_status": "harvesting",
                "candidate_topics_path": screening_path,
                "current_plan_path": plan_path,
                "field_scan_path": field_scan_path,
                "research_context_path": context_path,
            }

        # Step 2: Combined screening + ranking (single LLM call)
        final_top_n = int(os.getenv("FINAL_TOP_N", "3"))
        logger.info("Scoring, screening, and ranking candidates (Step 2) — selecting top %d...", final_top_n)

        def _screen_and_rank() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
            scoring_model_choice = os.getenv("SCORING_MODEL", "pro").lower()
            scoring_llm = self.fast_llm if scoring_model_choice == "fast" else (self.pro_llm or self.fast_llm)
            if not scoring_llm:
                raise ValueError("No LLM available for screening")

            scoring_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a senior research committee member and expert academic evaluator.\n\n"
                    "For EACH candidate topic:\n"
                    "1. Score it out of 100 considering: impact, quantitative operability, novelty, "
                    "publishability, data availability, method automatability, policy relevance, "
                    "methodological rigor, and top-tier outlet fit.\n"
                    "2. Decide whether it passes all hard gates (passed_gates=true/false). "
                    "If it fails, provide a rejection_reason.\n"
                    f"3. Among the topics that pass gates, rank the BEST {final_top_n} "
                    f"(rank 1 = best, rank 2 = second best, ..., rank {final_top_n}). "
                    "Set rank=0 for all topics that fail gates or are not in the top selection.\n\n"
                    "Return structured scores for ALL candidates."
                ),
                (
                    "user",
                    "Domain: {domain}\n\nField Scan Context:\n{field_scan_context}\n\n"
                    "Past Ideas (avoid repeating):\n{past_ideas}\n\nCandidates to evaluate:\n{candidates_json}\n",
                ),
            ])

            structured_llm = scoring_llm.with_structured_output(TopicScoresList)
            prompt_value = scoring_prompt.invoke({
                "domain": domain,
                "past_ideas": past_context,
                "field_scan_context": field_scan_context,
                "candidates_json": json.dumps(light_candidates, ensure_ascii=False),
            })
            result = structured_llm.invoke(prompt_value)
            scores_by_title = {s.title.strip().lower(): s for s in result.scores}

            scored_all = []
            for c in light_candidates:
                title = str(c.get("title", "")).strip()
                s = scores_by_title.get(title.lower())
                entry = dict(c)
                if s:
                    entry["initial_score"] = s.score
                    entry["final_score"] = s.score
                    entry["passed_gates"] = s.passed_gates
                    entry["rejection_reason"] = s.rejection_reason or ""
                    entry["rank"] = s.rank
                else:
                    entry["initial_score"] = 0
                    entry["final_score"] = 0
                    entry["passed_gates"] = False
                    entry["rejection_reason"] = "Not evaluated by screening model."
                    entry["rank"] = 0
                scored_all.append(entry)

            rejected = [c for c in scored_all if not c.get("passed_gates")]
            ranked = [c for c in scored_all if c.get("rank", 0) > 0]
            ranked.sort(key=lambda x: x.get("rank", 999))
            passed_all = [c for c in scored_all if c.get("passed_gates")]
            passed_all.sort(key=lambda x: x.get("initial_score", 0), reverse=True)
            return ranked[:final_top_n], rejected, passed_all

        top3, rejected_candidates, passed_candidates_pool = _screen_and_rank()
        backup_candidates = passed_candidates_pool

        if not top3:
            logger.warning("No ranked candidates from screening; falling back to top scorers.")
            all_scored = [c for c in light_candidates]
            all_scored.sort(key=lambda x: x.get("initial_score", 0), reverse=True)
            top3 = all_scored[:final_top_n]
            backup_candidates = all_scored[final_top_n:]
            for i, c in enumerate(top3):
                c["rank"] = i + 1
                c["final_score"] = c.get("initial_score", 0)

        # Derive finalists for downstream use (all passed candidates)
        finalists = top3

        logger.info("Step 2 complete: %d candidates selected for enrichment.", len(top3))

        # Step 3: Enrichment
        logger.info("Step 3: Enriching top candidates with full research details...")

        enrichment_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert academic research designer. Given a selected research topic, "
                "produce the FULL research specification.\n\nField Scan Context:\n{field_scan_context}"
            ),
            (
                "user",
                "Domain: {domain}\n\nTopic title: {title}\nBrief rationale: {brief_rationale}\n\n"
                "Provide: impact_evidence, novelty_gap_type, publishability (2-3 specific journals), "
                "quantitative_specs (with ≥6 robustness checks), and data_sources (freely accessible only)."
            ),
        ])

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def _enrich_single(candidate: Dict[str, Any]) -> Dict[str, Any]:
            enrichment_llm = self.pro_llm or self.fast_llm
            if not isinstance(enrichment_llm, ChatGoogleGenerativeAI):
                return {
                    **candidate,
                    "impact_evidence": "High impact (offline fallback)",
                    "novelty_gap_type": "Problem Gap",
                    "publishability": "Journal A, Journal B",
                    "quantitative_specs": {
                        "unit_of_analysis": "cities",
                        "outcomes": ["Outcome1"],
                        "exposures": ["Exposure1"],
                        "estimand_and_strategy": "ATE via DiD",
                        "model_family": "OLS",
                        "robustness_checks": ["check1", "check2", "check3", "check4", "check5", "check6"],
                        "expected_tables_figures": ["Table1"],
                    },
                    "data_sources": [{"name": "US Census", "accessibility": "Public API"}],
                }
            structured_llm = enrichment_llm.with_structured_output(RawCandidateTopic)
            prompt_value = enrichment_prompt.invoke({
                "domain": domain,
                "field_scan_context": field_scan_context,
                "title": candidate.get("title", ""),
                "brief_rationale": candidate.get("brief_rationale", ""),
            })
            result = structured_llm.invoke(prompt_value)
            enriched_fields = result.model_dump()
            return {**candidate, **enriched_fields}

        enriched_top3: List[Dict[str, Any]] = []
        for i, candidate in enumerate(top3):
            try:
                logger.info("Enriching candidate %d/%d: %s...", i + 1, len(top3), candidate.get("title", "")[:70])
                enriched = _enrich_single(candidate)
                enriched_top3.append(enriched)
            except Exception as e:
                logger.error("Enrichment failed for candidate %d: %s", i + 1, e)
                raise RuntimeError(f"Enrichment failed for candidate {i + 1}: {e}") from e

        logger.info("Step 3 complete: %d candidates enriched.", len(enriched_top3))

        # Step 3.5: Novelty check for top ideas with targeted recent OpenAlex search
        logger.info("Step 3.5: Running targeted novelty checks on selected ideas...")
        novelty_months_back = int(os.getenv("NOVELTY_MONTHS_BACK", "15"))
        selected_titles = {str(c.get("title", "")).strip().lower() for c in enriched_top3}
        novelty_rejected: List[Dict[str, Any]] = []

        for idx, candidate in enumerate(enriched_top3):
            novelty_info = self._run_novelty_check(candidate, domain=domain, months_back=novelty_months_back)
            candidate.update(novelty_info)
            candidate["novelty_checked"] = True
            candidate["novelty_checked_at"] = datetime.now(timezone.utc).isoformat()
            if candidate.get("novelty_verdict") == "already_published":
                novelty_rejected.append(candidate)

        if novelty_rejected:
            logger.info("Novelty gate removed %d ideas as already_published. Attempting replacements...", len(novelty_rejected))
            enriched_top3 = [c for c in enriched_top3 if c.get("novelty_verdict") != "already_published"]
            for removed in novelty_rejected:
                removed["passed_gates"] = False
                removed["rejection_reason"] = "Failed novelty gate: already_published."
                removed["rejected_by_novelty_gate"] = True
                rejected_candidates.append(removed)

            for backup in passed_candidates_pool:
                backup_title = str(backup.get("title", "")).strip().lower()
                if backup_title in selected_titles:
                    continue
                try:
                    enriched_backup = _enrich_single(backup)
                    novelty_info = self._run_novelty_check(enriched_backup, domain=domain, months_back=novelty_months_back)
                    enriched_backup.update(novelty_info)
                    enriched_backup["novelty_checked"] = True
                    enriched_backup["novelty_checked_at"] = datetime.now(timezone.utc).isoformat()
                    selected_titles.add(backup_title)
                    if enriched_backup.get("novelty_verdict") == "already_published":
                        enriched_backup["passed_gates"] = False
                        enriched_backup["rejection_reason"] = "Failed novelty gate: already_published."
                        enriched_backup["rejected_by_novelty_gate"] = True
                        rejected_candidates.append(enriched_backup)
                        continue
                    enriched_top3.append(enriched_backup)
                    if len(enriched_top3) >= final_top_n:
                        break
                except Exception as exc:
                    logger.warning("Failed to evaluate replacement candidate '%s': %s", backup.get("title", ""), exc)

            enriched_top3 = enriched_top3[:final_top_n]

        for i, c in enumerate(enriched_top3):
            c["rank"] = i + 1

        # Save outputs
        with open(screening_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "candidates": enriched_top3,
                    "gates_passed": len(enriched_top3),
                    "backup_candidates": backup_candidates,
                },
                f, indent=2, ensure_ascii=False,
            )

        all_rejected = rejected_candidates

        graveyard_path = settings.ideas_graveyard_path(domain=domain)
        existing_graveyard: list = []
        if os.path.exists(graveyard_path):
            try:
                with open(graveyard_path, "r", encoding="utf-8") as f:
                    existing_graveyard = json.load(f)
            except json.JSONDecodeError:
                pass
        existing_graveyard.extend(all_rejected)
        with open(graveyard_path, "w", encoding="utf-8") as f:
            json.dump(existing_graveyard, f, indent=2, ensure_ascii=False)

        for candidate in enriched_top3:
            try:
                self.memory.store_idea(
                    topic=candidate.get("title", ""),
                    domain=domain,
                    status="selected",
                    rejection_reason="",
                    metadata={
                        "score": candidate.get("final_score", candidate.get("initial_score")),
                        "gap_type": candidate.get("novelty_gap_type", ""),
                        "run_id": run_id,
                    },
                    source_file=screening_path,
                )
            except Exception as e:
                logger.warning("Failed to store selected idea in memory: %s", e)

        for candidate in rejected_candidates:
            try:
                self.memory.store_idea(
                    topic=candidate.get("title", ""),
                    domain=domain,
                    status="discarded",
                    rejection_reason=candidate.get("rejection_reason", ""),
                    metadata={"score": candidate.get("initial_score", 0), "run_id": run_id},
                    source_file=graveyard_path,
                )
            except Exception as e:
                logger.warning("Failed to store rejected idea in memory: %s", e)

        ranking_path = settings.topic_ranking_path()
        with open(ranking_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Topic", "InitialScore", "FinalScore"])
            for c in enriched_top3:
                writer.writerow([c.get("rank", ""), c.get("title", ""), c.get("initial_score", ""), c.get("final_score", "")])

        # Step 4: Final plan generation
        logger.info("Generating final research_plan.json using Pro model...")
        top1 = enriched_top3[0] if enriched_top3 else {}
        top3_context = json.dumps(
            [
                {
                    "rank": c.get("rank"),
                    "title": c.get("title", ""),
                    "brief_rationale": c.get("brief_rationale", ""),
                    "quantitative_specs": c.get("quantitative_specs", {}),
                    "data_sources": c.get("data_sources", []),
                }
                for c in enriched_top3[:3]
            ],
            ensure_ascii=False,
            indent=2,
        )
        top1_context = json.dumps(
            {
                "title": top1.get("title", ""),
                "brief_rationale": top1.get("brief_rationale", ""),
                "quantitative_specs": top1.get("quantitative_specs", {}),
                "data_sources": top1.get("data_sources", []),
                "publishability": top1.get("publishability", ""),
            },
            ensure_ascii=False, indent=2,
        )

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def generate_plan():
            if not self.pro_llm:
                raise ValueError("LLM not initialized")
            if not isinstance(self.pro_llm, ChatGoogleGenerativeAI):
                title = top1.get("title", "Generated Topic")
                self._degraded_nodes.append("ideation:plan_placeholder")
                logger.warning("Pro LLM unavailable; research plan generated from placeholder template.")
                return {
                    "project_title": title,
                    "study_type": "quantitative",
                    "topic_screening": {"top_candidate_title": title},
                    "research_questions": [f"RQ1: What drives outcomes for {title}?"],
                    "hypotheses": ["H1: There is a measurable effect."],
                    "unit_of_analysis": "units",
                    "outcomes": [],
                    "exposures": [],
                    "keywords": [domain],
                    "data_sources": [],
                    "methodology": {"design": "placeholder"},
                }
            structured_llm = self.pro_llm.with_structured_output(ResearchPlanSchema)
            plan_prompt_cache_hours = float(os.getenv("PLAN_PROMPT_CACHE_HOURS", "24"))
            plan_cache_key = build_cache_key(
                "ideation_research_plan",
                {"domain": domain, "top1_context": top1_context, "top3_context": top3_context},
            )
            cached_plan = load_json_cache("research_plans", plan_cache_key, max_age_hours=plan_prompt_cache_hours)
            if isinstance(cached_plan, dict) and cached_plan:
                return cached_plan

            plan = structured_llm.invoke(
                "Design a comprehensive quantitative research plan based on the selected top candidates.\n\n"
                "Primary selected topic (rank 1):\n"
                + top1_context
                + "\n\nAdditional ranked context (top 3 shortlist):\n"
                + top3_context
            )
            plan_data_cached = plan.model_dump()
            save_json_cache("research_plans", plan_cache_key, plan_data_cached)
            return plan_data_cached

        plan_data = generate_plan()
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)

        context: Dict[str, Any] = {
            "domain": domain,
            "run_id": run_id,
            "field_scan_summary": summarize_field_scan(field_scan_data or {}) if field_scan_data else {},
            "selected_topic": {
                "title": top1.get("title", ""),
                "score": top1.get("final_score", top1.get("initial_score")),
                "quantitative_specs": top1.get("quantitative_specs", {}),
                "data_sources": top1.get("data_sources", []),
                "publishability": top1.get("publishability", ""),
            },
            "rejection_summary": {"count": len(rejected_candidates)},
            "plan_essentials": {
                "research_questions": plan_data.get("research_questions", []),
                "hypotheses": plan_data.get("hypotheses", []),
                "outcomes": plan_data.get("outcomes", []),
                "exposures": plan_data.get("exposures", []),
                "methodology": plan_data.get("methodology", {}),
                "keywords": plan_data.get("keywords", []),
            },
        }

        try:
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(context, f, indent=2, ensure_ascii=False)
            logger.info("Research context summary saved to %s", context_path)
        except Exception as e:
            logger.warning("Failed to write research context: %s", e)

        # Step 5: Persist enriched top-3 to long-term JSONL archive
        _persist_enriched_top3(enriched_top3, domain=domain, run_id=run_id)

        logger.info("Ideation complete. Plan saved to %s", plan_path)

        return {
            "execution_status": "harvesting",
            "candidate_topics_path": screening_path,
            "current_plan_path": plan_path,
            "field_scan_path": field_scan_path,
            "research_context_path": context_path,
            "degraded_nodes": self._degraded_nodes,
        }


def ideation_node(state: ResearchState) -> ResearchState:
    agent = IdeationAgent()
    return agent.run(state)
