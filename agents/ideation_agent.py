import json
import os
import csv
import uuid
from datetime import datetime, timezone
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


class TopicScoresList(BaseModel):
    scores: List[TopicScore]


class FinalRankScore(BaseModel):
    title: str = Field(description="Title of the candidate topic.")
    rank: int = Field(description="Final ranking position (1 = best overall).")
    final_score: int = Field(description="Refined quality score out of 100 after careful re-evaluation.")


class FinalRankScoresList(BaseModel):
    top_candidates: List[FinalRankScore]


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


class RawCandidateTopicsList(BaseModel):
    candidates: List[RawCandidateTopic]


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

        # 1. Memory RAG
        past_ideas = self.memory.retrieve_domain_context(domain)
        past_context = json.dumps(past_ideas, indent=2) if past_ideas else "No prior memory for this domain."

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
                "Output ONLY: a formal title and a 1-2 sentence rationale hinting at method, data, and gap.\n"
                "Ensure HIGH DIVERSITY across spatial scales, methods, and data sources.\n"
                "Avoid repeating past ideas: {past_ideas}"
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

        # Step 2: Initial screening
        logger.info("Scoring and screening candidate topics (Step 2)...")
        initial_screen_topn = int(os.getenv("INITIAL_SCREEN_TOPN", "10"))

        def _initial_screen() -> List[Dict[str, Any]]:
            scoring_model_choice = os.getenv("SCORING_MODEL", "pro").lower()
            scoring_llm = self.fast_llm if scoring_model_choice == "fast" else (self.pro_llm or self.fast_llm)
            if not scoring_llm:
                raise ValueError("No LLM available for initial screening")

            scoring_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an expert academic evaluator. Score each candidate research topic out of 100 and "
                    "decide whether it passes all hard gates. Consider: impact, quantitative operability, "
                    "novelty, publishability, data availability, method automatability. "
                    "Return ONLY structured scores per topic."
                ),
                (
                    "user",
                    "Domain: {domain}\n\nField Scan Context:\n{field_scan_context}\n\n"
                    "Past Ideas (avoid repeating):\n{past_ideas}\n\nCandidates to score:\n{candidates_json}\n",
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

            scored = []
            for c in light_candidates:
                title = str(c.get("title", "")).strip()
                s = scores_by_title.get(title.lower())
                entry = dict(c)
                if s:
                    entry["initial_score"] = s.score
                    entry["passed_gates"] = s.passed_gates
                    entry["rejection_reason"] = s.rejection_reason or ""
                else:
                    entry["initial_score"] = 0
                    entry["passed_gates"] = False
                    entry["rejection_reason"] = "Not evaluated by initial screening model."
                scored.append(entry)
            return scored

        initially_scored = _initial_screen()
        logger.info("Step 2 complete: scored %d candidates.", len(initially_scored))

        passed_initial = [c for c in initially_scored if c.get("passed_gates")]
        rejected_candidates = [c for c in initially_scored if not c.get("passed_gates")]

        passed_initial.sort(key=lambda x: x.get("initial_score", 0), reverse=True)
        finalists = passed_initial[:initial_screen_topn]

        if not finalists:
            finalists = sorted(initially_scored, key=lambda x: x.get("initial_score", 0), reverse=True)[:initial_screen_topn]
            logger.warning("All candidates failed initial gates; using top-scoring as finalists anyway.")

        logger.info("%d candidates advanced to final selection.", len(finalists))

        # Step 3: Final selection
        final_top_n = int(os.getenv("FINAL_TOP_N", "3"))
        logger.info("Step 3: Re-ranking %d finalists to choose top %d...", len(finalists), final_top_n)

        def _final_selection() -> List[Dict[str, Any]]:
            scoring_model_choice = os.getenv("SCORING_MODEL", "pro").lower()
            scoring_llm = self.fast_llm if scoring_model_choice == "fast" else (self.pro_llm or self.fast_llm)

            selection_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a senior research committee member. Apply stricter criteria: "
                    "policy relevance, methodological rigor, novel gap, top-tier outlet fit, automatable pipeline. "
                    f"Select and rank EXACTLY the top {final_top_n} candidates (rank 1 = best). "
                    "Assign refined final_score out of 100. Return ONLY structured data."
                ),
                (
                    "user",
                    "Domain: {domain}\n\nField Scan Context:\n{field_scan_context}\n\n"
                    "Shortlisted candidates:\n{candidates_json}\n",
                ),
            ])

            structured_llm = scoring_llm.with_structured_output(FinalRankScoresList)
            prompt_value = selection_prompt.invoke({
                "domain": domain,
                "field_scan_context": field_scan_context,
                "candidates_json": json.dumps(finalists, ensure_ascii=False),
            })
            result = structured_llm.invoke(prompt_value)

            final_by_title = {r.title.strip().lower(): r for r in result.top_candidates}
            ranked = []
            for c in finalists:
                title = str(c.get("title", "")).strip()
                r = final_by_title.get(title.lower())
                if r:
                    entry = dict(c)
                    entry["rank"] = r.rank
                    entry["final_score"] = r.final_score
                    ranked.append(entry)

            ranked.sort(key=lambda x: x.get("rank", 999))
            return ranked[:final_top_n]

        top3 = _final_selection()
        if not top3:
            logger.warning("Final selection returned no ranked candidates; falling back to top initial scorers.")
            top3 = [dict(c) for c in finalists[:final_top_n]]
            for i, c in enumerate(top3):
                c["rank"] = i + 1
                c["final_score"] = c.get("initial_score", 0)

        logger.info("Step 3 complete: %d candidates selected for enrichment.", len(top3))

        # Step 4: Enrichment
        logger.info("Step 4: Enriching top candidates with full research details...")

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
            structured_llm = enrichment_llm.with_structured_output(RawCandidateTopicsList)
            prompt_value = enrichment_prompt.invoke({
                "domain": domain,
                "field_scan_context": field_scan_context,
                "title": candidate.get("title", ""),
                "brief_rationale": candidate.get("brief_rationale", ""),
            })
            result = structured_llm.invoke(prompt_value)
            if result.candidates:
                enriched_fields = result.candidates[0].model_dump()
                return {**candidate, **enriched_fields}
            return candidate

        enriched_top3: List[Dict[str, Any]] = []
        for i, candidate in enumerate(top3):
            try:
                logger.info("Enriching candidate %d/%d: %s...", i + 1, len(top3), candidate.get("title", "")[:70])
                enriched = _enrich_single(candidate)
                enriched_top3.append(enriched)
            except Exception as e:
                logger.warning("Enrichment failed for candidate %d: %s", i + 1, e)
                enriched_top3.append(candidate)

        logger.info("Step 4 complete: %d candidates enriched.", len(enriched_top3))

        # Save outputs
        with open(screening_path, "w", encoding="utf-8") as f:
            json.dump(
                {"run_id": run_id, "candidates": enriched_top3, "gates_passed": len(enriched_top3)},
                f, indent=2, ensure_ascii=False,
            )

        finalist_titles = {str(c.get("title", "")).strip().lower() for c in finalists}
        non_finalist_passers = [c for c in passed_initial if str(c.get("title", "")).strip().lower() not in finalist_titles]
        all_rejected = rejected_candidates + non_finalist_passers

        graveyard_path = settings.ideas_graveyard_path()
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

        # Step 5: Final plan generation
        logger.info("Generating final research_plan.json using Pro model...")
        top1 = enriched_top3[0] if enriched_top3 else {}
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
            plan = structured_llm.invoke(
                "Design a comprehensive quantitative research plan based on the following selected topic:\n\n"
                + top1_context
            )
            return plan.model_dump()

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

        # Step 6: Persist enriched top-3 to long-term JSONL archive
        _persist_enriched_top3(enriched_top3, domain=domain, run_id=run_id)

        logger.info("Ideation complete. Plan saved to %s", plan_path)

        return {
            "execution_status": "harvesting",
            "candidate_topics_path": screening_path,
            "current_plan_path": plan_path,
            "field_scan_path": field_scan_path,
            "research_context_path": context_path,
        }


def ideation_node(state: ResearchState) -> ResearchState:
    agent = IdeationAgent()
    return agent.run(state)
