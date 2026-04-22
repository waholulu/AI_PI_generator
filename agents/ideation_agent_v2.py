"""
IdeationAgentV2 — dual-mode topic ideation (Level 1: user topic, Level 2: domain search).

Level 1 (--mode level_1 / --user-topic path.yaml):
  - Loads user-provided structured YAML, validates as Topic
  - Runs hard-blockers (G2/G3/G6) synchronously — raises HITLInterruption on any fail
  - Calls run_reflection_loop with max_rounds=1
  - ACCEPTED → produces research_plan.json + topic_screening.json
  - TENTATIVE after 1 round → raises HITLInterruption

Level 2 (--mode level_2 / default):
  - Generates 30 SeedCandidates from ideation_seed prompt
  - Runs reflection loop per topic (parallel, max 5 workers)
  - Splits ACCEPTED / TENTATIVE / REJECTED
  - ACCEPTED → ranks → top-3 → data verification → research_plan.json
  - TENTATIVE → writes tentative_pool.json
  - Writes ideation_run_summary.json
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Optional

import yaml

from agents.budget_tracker import BudgetExceededError, BudgetTracker
from agents.logging_config import get_logger
from agents.memory_retriever import MemoryRetriever
from agents.openalex_verifier import OpenAlexVerifier
from agents.reflection_loop import ReflectionTrace, run_reflection_loop
from agents.rule_engine import RuleEngine
from agents import settings
from models.topic_schema import (
    FinalStatus,
    HITLInterruption,
    SeedCandidate,
    Topic,
)

logger = get_logger(__name__)


def _build_legacy_gates_map(trace: ReflectionTrace) -> dict:
    """Map seven-gate results to the legacy six-gate format expected by downstream agents."""
    gate_map = {
        "G1": {"gate": "mechanism_plausibility", "passed": True, "score": None},
        "G2": {"gate": "scale_alignment", "passed": True},
        "G3": {"gate": "data_availability", "passed": True},
        "G4": {"gate": "identification_validity", "passed": True, "score": None},
        "G5": {"gate": "novelty", "passed": True, "score": None},
        "G6": {"gate": "automation_feasibility", "passed": True},
        "G7": {"gate": "contribution_clarity", "passed": True, "score": None},
    }
    for rnd in trace.rounds:
        for gr in rnd.gate_results:
            if gr.gate_id in gate_map:
                gate_map[gr.gate_id]["passed"] = gr.passed
                if gr.score is not None:
                    gate_map[gr.gate_id]["score"] = gr.score

    # Collapse to legacy six-gate format (impact, data, novelty, feasibility, gap, contribution)
    return {
        "impact": gate_map["G1"]["passed"],
        "data": gate_map["G3"]["passed"],
        "novelty": gate_map["G5"]["passed"],
        "feasibility": gate_map["G6"]["passed"],
        "gap": gate_map["G7"]["passed"],
        "contribution": gate_map["G7"]["passed"],
        "full_seven_gates": {k: v for k, v in gate_map.items()},
    }


class _UserInputError(Exception):
    pass


class IdeationAgentV2:
    """Dual-mode ideation agent using structured Topic slot schema + reflection loop."""

    def __init__(
        self,
        budget: Optional[BudgetTracker] = None,
        budget_override_usd: Optional[float] = None,
        skip_reflection: bool = False,
    ) -> None:
        self.rule_engine = RuleEngine()
        self.verifier = OpenAlexVerifier()
        self.memory = MemoryRetriever()
        self.skip_reflection = skip_reflection
        self.budget = budget or BudgetTracker(
            per_run_budget_usd=budget_override_usd
        )
        self._llm = self._init_llm()

    def _init_llm(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            fast_model = os.getenv("GEMINI_FAST_MODEL", "gemini-2.0-flash-lite")
            return ChatGoogleGenerativeAI(model=fast_model, temperature=0.7)
        except Exception as e:
            logger.warning("LLM init failed: %s — running without LLM", e)
            return None

    # ── Level 1 ───────────────────────────────────────────────────────────────

    def run_level1(self, state: dict) -> dict:
        """Level 1: validate user-provided topic YAML and run single-round reflection."""
        user_topic_path = state.get("user_topic_path", "")
        if not user_topic_path:
            raise _UserInputError("user_topic_path not set in state")

        # Load and validate topic YAML
        try:
            with open(user_topic_path) as f:
                raw = yaml.safe_load(f)
            topic = Topic.model_validate(raw)
        except FileNotFoundError:
            raise _UserInputError(f"user_topic_path not found: {user_topic_path}")
        except Exception as e:
            raise _UserInputError(f"Topic YAML validation failed: {e}")

        declared_sources: list[str] = raw.get("declared_sources", [])
        declared_rationale: str = raw.get("declared_sources_rationale", "")

        # Memory check (non-blocking)
        try:
            mem_ctx = self.memory.build_prompt_context(
                domain=topic.spatial_scope.geography,
                enriched_jsonl_path=settings.enriched_top_candidates_path(),
                graveyard_path=settings.ideas_graveyard_path(
                    domain=topic.spatial_scope.geography
                ),
            )
            if mem_ctx.get("summary", {}).get("recent_count", 0) > 0:
                logger.info("Memory: %d similar topics found (informational only)",
                            mem_ctx["summary"]["recent_count"])
        except Exception:
            pass

        # Hard-blockers (no LLM cost)
        hard_results = self.rule_engine.run_hard_blockers(topic, declared_sources)
        failed_hard = [r for r in hard_results if not r.passed]
        if failed_hard:
            suggested_ops = [
                {"op": "change_geography", "description": "try a geography with better data coverage"}
                if r.gate_id == "G3" else
                {"op": "change_spatial_unit", "description": "align spatial units"}
                if r.gate_id == "G2" else
                {"op": "declare_additional_sources", "description": "add more sources"}
                for r in failed_hard
            ]
            raise HITLInterruption(
                kind="hard_blocker_failed",
                message="; ".join(r.reason for r in failed_hard),
                failed_gates=[r.gate_id for r in failed_hard],
                suggested_operations=suggested_ops,
            )

        seed = SeedCandidate(
            topic=topic,
            declared_sources=declared_sources,
            declared_sources_rationale=declared_rationale,
        )

        trace = run_reflection_loop(
            seed,
            self.budget,
            rule_engine=self.rule_engine,
            verifier=self.verifier,
            llm=self._llm,
            max_rounds=1,
        )

        if trace.final_status != FinalStatus.ACCEPTED:
            raise HITLInterruption(
                kind="refinable_still_failing_after_one_round",
                message=f"Topic is {trace.final_status.value} after 1 round",
                failed_gates=[
                    r.gate_id
                    for rnd in trace.rounds
                    for r in rnd.gate_results
                    if not r.passed
                ],
                diff_from_original={
                    "original_topic_id": topic.meta.topic_id,
                    "final_status": trace.final_status.value,
                },
                suggested_next_operations=[],
            )

        return self._emit_level1_outputs(seed, trace, state)

    def _emit_level1_outputs(self, seed: SeedCandidate, trace: ReflectionTrace, state: dict) -> dict:
        run_id = state.get("run_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        legacy_gates = _build_legacy_gates_map(trace)
        topic = seed.topic
        legacy_dict = topic.to_legacy_dict()

        candidate_entry = {
            **legacy_dict,
            "rank": 1,
            "final_status": FinalStatus.ACCEPTED.value,
            "declared_sources": seed.declared_sources,
            "legacy_six_gates": legacy_gates,
            "reflection_trace_id": topic.meta.topic_id,
        }

        screening = {
            "run_id": run_id,
            "input_mode": "level_1",
            "candidates": [candidate_entry],
        }
        screening_path = settings.topic_screening_path()
        with open(screening_path, "w") as f:
            json.dump(screening, f, indent=2, ensure_ascii=False)

        plan_path = settings.research_plan_path()
        plan = _build_research_plan(candidate_entry, run_id)
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        return {
            "execution_status": "harvesting",
            "candidate_topics_path": screening_path,
            "current_plan_path": plan_path,
            "degraded_nodes": [],
        }

    # ── Level 2 ───────────────────────────────────────────────────────────────

    def run_level2(self, state: dict) -> dict:
        """Level 2: generate 30 seed candidates, run parallel reflection loop."""
        domain = state.get("domain_input", "Urban Planning and Health")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        t_start_dt = datetime.now(timezone.utc)
        t_start = t_start_dt.timestamp()

        field_scan_path = state.get("field_scan_path", settings.field_scan_path())
        field_scan_context = _load_field_scan_context(field_scan_path)
        memory_context = _load_memory_context(self.memory, domain)

        # Generate 30 seed candidates
        seeds = self._generate_seeds(domain, field_scan_context, memory_context)
        logger.info("Generated %d seed candidates", len(seeds))

        # Parallel reflection loop
        accepted: list[tuple[SeedCandidate, ReflectionTrace]] = []
        tentative: list[tuple[SeedCandidate, ReflectionTrace]] = []
        rejected: list[tuple[SeedCandidate, ReflectionTrace]] = []

        max_workers = int(os.getenv("IDEATION_V2_WORKERS", "5"))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for seed in seeds:
                if not self.budget.can_start_new_topic():
                    logger.warning("Budget 90%% reached — skipping remaining seeds")
                    break
                fut = executor.submit(self._run_one_seed, seed)
                futures[fut] = seed

            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    trace = fut.result()
                    if trace.final_status == FinalStatus.ACCEPTED:
                        accepted.append((seed, trace))
                    elif trace.final_status == FinalStatus.TENTATIVE:
                        tentative.append((seed, trace))
                    else:
                        rejected.append((seed, trace))
                except BudgetExceededError as e:
                    logger.warning("Budget exceeded for %s: %s", seed.topic.meta.topic_id, e)
                    tentative.append((seed, _make_budget_exceeded_trace(seed, str(e))))
                except Exception as e:
                    logger.warning("Reflection loop error for %s: %s",
                                   seed.topic.meta.topic_id, e)

        logger.info(
            "Reflection done — ACCEPTED: %d, TENTATIVE: %d, REJECTED: %d",
            len(accepted), len(tentative), len(rejected),
        )

        # Rank accepted → top-3
        top_candidates = self._rank_and_select(accepted, domain)[:3]

        # Write tentative pool
        self._write_tentative_pool(tentative, run_id)

        # Build outputs
        t_end = datetime.now(timezone.utc).timestamp()
        return self._emit_level2_outputs(
            top_candidates, tentative, rejected, domain, run_id, t_start_dt,
            wallclock=t_end - t_start,
            total_attempted=len(seeds),
        )

    def _run_one_seed(self, seed: SeedCandidate) -> ReflectionTrace:
        return run_reflection_loop(
            seed,
            self.budget,
            rule_engine=self.rule_engine,
            verifier=self.verifier,
            llm=None if self.skip_reflection else self._llm,
            max_rounds=1 if self.skip_reflection else None,
        )

    def _generate_seeds(
        self, domain: str, field_scan_context: str, memory_context: str
    ) -> list[SeedCandidate]:
        """Generate 30 SeedCandidates from LLM or fallback stubs."""
        try:
            seeds = self._llm_generate_seeds(domain, field_scan_context, memory_context)
            if seeds:
                return seeds
        except Exception as e:
            logger.warning("LLM seed generation failed: %s — using fallback", e)
        return self._fallback_seeds(domain)

    def _llm_generate_seeds(
        self, domain: str, field_scan_context: str, memory_context: str
    ) -> list[SeedCandidate]:
        if self._llm is None:
            return []

        prompt_path = settings.prompts_dir() / "ideation_seed.txt"
        if prompt_path.exists():
            template = prompt_path.read_text()
        else:
            template = (
                "Generate 30 research topic seeds for domain: {domain}.\n"
                "Field context: {field_scan_context}\n"
                "Memory: {memory_context}\n"
                "Return JSON array of objects with keys: topic_id, title, exposure, outcome, "
                "geography, method, declared_sources, notes."
            )

        prompt = template.format(
            domain=domain,
            field_scan_context=field_scan_context[:2000],
            memory_context=memory_context[:1000],
        )
        from langchain_core.messages import HumanMessage

        response = self._llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        items = json.loads(raw)
        seeds = []
        for i, item in enumerate(items[:30]):
            try:
                seed = _dict_to_seed_candidate(item, domain, i)
                seeds.append(seed)
            except Exception as e:
                logger.warning("Seed %d parse failed: %s", i, e)
        return seeds

    def _fallback_seeds(self, domain: str) -> list[SeedCandidate]:
        """Minimal stub seeds used when LLM is unavailable."""
        from models.topic_schema import (
            Contribution, ContributionPrimary, ExposureFamily, ExposureX,
            Frequency, IdentificationPrimary, IdentificationStrategy, OutcomeFamily,
            OutcomeY, SamplingMode, SpatialScope, TemporalScope, TopicMeta,
        )
        seeds = []
        for i in range(3):
            tid = f"fallback_{i:03d}"
            topic = Topic(
                meta=TopicMeta(topic_id=tid),
                exposure_X=ExposureX(
                    family=ExposureFamily.BUILT_ENVIRONMENT,
                    specific_variable=f"built_env_var_{i}",
                    spatial_unit="tract",
                ),
                outcome_Y=OutcomeY(
                    family=OutcomeFamily.HEALTH,
                    specific_variable=f"health_outcome_{i}",
                    spatial_unit="tract",
                ),
                spatial_scope=SpatialScope(
                    geography=domain[:50],
                    spatial_unit="tract",
                    sampling_mode=SamplingMode.PANEL,
                ),
                temporal_scope=TemporalScope(
                    start_year=2010, end_year=2020, frequency=Frequency.ANNUAL
                ),
                identification=IdentificationStrategy(
                    primary=IdentificationPrimary.FE,
                    key_threats=["confounding"],
                    mitigations=["fixed_effects"],
                ),
                contribution=Contribution(
                    primary=ContributionPrimary.NOVEL_CONTEXT,
                    statement=f"Fallback topic {i} for {domain}.",
                ),
                free_form_title=f"Fallback topic {i}",
            )
            seeds.append(SeedCandidate(topic=topic, declared_sources=["NHGIS"]))
        return seeds

    def _rank_and_select(
        self, accepted: list[tuple[SeedCandidate, ReflectionTrace]], domain: str
    ) -> list[dict]:
        candidates = []
        for seed, trace in accepted:
            legacy_gates = _build_legacy_gates_map(trace)
            last_round = trace.rounds[-1] if trace.rounds else None
            score = last_round.round_score if last_round else 3.0
            entry = {
                **seed.topic.to_legacy_dict(),
                "rank": 0,
                "final_status": FinalStatus.ACCEPTED.value,
                "declared_sources": seed.declared_sources,
                "legacy_six_gates": legacy_gates,
                "reflection_trace_id": seed.topic.meta.topic_id,
                "_sort_score": score,
            }
            candidates.append(entry)

        candidates.sort(key=lambda x: x.get("_sort_score", 0), reverse=True)
        for i, c in enumerate(candidates):
            c["rank"] = i + 1
            c.pop("_sort_score", None)
        return candidates

    def _write_tentative_pool(
        self, tentative: list[tuple[SeedCandidate, ReflectionTrace]], run_id: str
    ) -> None:
        pool = []
        for seed, trace in tentative:
            last_round = trace.rounds[-1] if trace.rounds else None
            failed_gates = []
            if last_round:
                failed_gates = [r.gate_id for r in last_round.gate_results if not r.passed]
            pool.append({
                "topic_id": seed.topic.meta.topic_id,
                "title": seed.topic.free_form_title or seed.topic.exposure_X.specific_variable,
                "run_id": run_id,
                "failed_gates": failed_gates,
                "declared_sources": seed.declared_sources,
                "legacy_six_gates": _build_legacy_gates_map(trace),
                "topic_dict": seed.topic.model_dump(),
                "trace_rounds": len(trace.rounds),
            })

        path = settings.tentative_pool_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"run_id": run_id, "tentative": pool}, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %d tentative topics to %s", len(pool), path)

    def _emit_level2_outputs(
        self,
        top_candidates: list[dict],
        tentative: list,
        rejected: list,
        domain: str,
        run_id: str,
        started_at: datetime,
        wallclock: float,
        total_attempted: int,
    ) -> dict:
        if not top_candidates:
            logger.warning("No ACCEPTED candidates — falling back to degraded plan")
            top_candidates = [{
                "title": f"No accepted topics for: {domain}",
                "rank": 1,
                "final_status": "DEGRADED",
                "legacy_six_gates": {},
            }]

        screening = {
            "run_id": run_id,
            "input_mode": "level_2",
            "domain": domain,
            "candidates": top_candidates,
        }
        screening_path = settings.topic_screening_path()
        with open(screening_path, "w") as f:
            json.dump(screening, f, indent=2, ensure_ascii=False)

        plan = _build_research_plan(top_candidates[0], run_id)
        plan_path = settings.research_plan_path()
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        # ideation_run_summary.json
        summary = {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "input_mode": "level_2",
            "total_topics_attempted": total_attempted,
            "status_breakdown": {
                "ACCEPTED": len(top_candidates),
                "TENTATIVE": len(tentative),
                "REJECTED": len(rejected),
            },
            "total_cost_usd": self.budget.snapshot()["per_run_spent_usd"],
            "total_wallclock_seconds": wallclock,
            "trace_files": [
                str(settings.ideation_traces_dir() / f"{c.get('topic_id', c.get('reflection_trace_id', '?'))}_trace.json")
                for c in top_candidates
            ],
        }
        summary_path = settings.ideation_run_summary_path()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        context_path = settings.research_context_path()
        context = {
            "domain": domain,
            "run_id": run_id,
            "selected_topic": top_candidates[0] if top_candidates else {},
        }
        with open(context_path, "w") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)

        return {
            "execution_status": "harvesting",
            "candidate_topics_path": screening_path,
            "current_plan_path": plan_path,
            "research_context_path": context_path,
            "degraded_nodes": [],
        }

    def run(self, state: dict) -> dict:
        mode = state.get("ideation_mode", "level_2")
        if mode == "level_1" or state.get("user_topic_path"):
            try:
                return self.run_level1(state)
            except (HITLInterruption, _UserInputError):
                raise
            except Exception as e:
                if _should_fallback_level2(e):
                    logger.warning("Level 1 failed (%s) — falling back to Level 2", e)
                    return self.run_level2(state)
                raise
        return self.run_level2(state)


# ── Helper functions ──────────────────────────────────────────────────────────

def _load_field_scan_context(path: str) -> str:
    if not os.path.exists(path):
        return "No field scan available."
    try:
        import json as _json
        from agents.field_scanner_agent import summarize_field_scan
        with open(path) as f:
            data = _json.load(f)
        return _json.dumps(summarize_field_scan(data), indent=2)
    except Exception as e:
        logger.warning("Field scan load failed: %s", e)
        return "Field scan unavailable."


def _load_memory_context(memory: MemoryRetriever, domain: str) -> str:
    try:
        ctx = memory.build_prompt_context(
            domain=domain,
            enriched_jsonl_path=settings.enriched_top_candidates_path(),
            graveyard_path=settings.ideas_graveyard_path(domain=domain),
        )
        return json.dumps(ctx, indent=2, ensure_ascii=False)
    except Exception:
        return "Memory unavailable."


def _build_research_plan(candidate: dict, run_id: str) -> dict:
    return {
        "project_title": candidate.get("title", "Research Topic"),
        "run_id": run_id,
        "study_type": "quantitative_causal",
        "topic_screening": {
            "top_candidate_title": candidate.get("title", ""),
            "legacy_six_gates": candidate.get("legacy_six_gates", {}),
        },
        "research_questions": [],
        "hypotheses": [],
        "unit_of_analysis": candidate.get("exposure_variable", ""),
        "outcomes": [{"name": candidate.get("outcome_variable", "")}],
        "exposures": [{"name": candidate.get("exposure_variable", "")}],
        "keywords": [],
        "data_sources": [{"name": s} for s in candidate.get("declared_sources", [])],
        "methodology": {"identification": candidate.get("method", "")},
    }


def _dict_to_seed_candidate(item: dict, domain: str, idx: int) -> SeedCandidate:
    from models.topic_schema import (
        Contribution, ContributionPrimary, ExposureFamily, ExposureX,
        Frequency, IdentificationPrimary, IdentificationStrategy, OutcomeFamily,
        OutcomeY, SamplingMode, SpatialScope, TemporalScope, TopicMeta,
    )

    tid = item.get("topic_id") or f"seed_{idx:03d}"
    topic = Topic(
        meta=TopicMeta(topic_id=tid),
        exposure_X=ExposureX(
            family=ExposureFamily.OTHER,
            specific_variable=item.get("exposure", f"exposure_{idx}"),
            spatial_unit=item.get("spatial_unit", "tract"),
        ),
        outcome_Y=OutcomeY(
            family=OutcomeFamily.OTHER,
            specific_variable=item.get("outcome", f"outcome_{idx}"),
            spatial_unit=item.get("spatial_unit", "tract"),
        ),
        spatial_scope=SpatialScope(
            geography=item.get("geography", domain[:50]),
            spatial_unit=item.get("spatial_unit", "tract"),
            sampling_mode=SamplingMode.PANEL,
        ),
        temporal_scope=TemporalScope(
            start_year=item.get("start_year", 2010),
            end_year=item.get("end_year", 2020),
            frequency=Frequency.ANNUAL,
        ),
        identification=IdentificationStrategy(
            primary=IdentificationPrimary(item.get("method", "fixed_effects")),
            key_threats=item.get("threats", ["confounding"]),
            mitigations=item.get("mitigations", ["fixed_effects"]),
        ),
        contribution=Contribution(
            primary=ContributionPrimary.NOVEL_CONTEXT,
            statement=item.get("notes", f"Topic {idx} for {domain}"),
        ),
        free_form_title=item.get("title", f"Topic {idx}"),
    )
    return SeedCandidate(
        topic=topic,
        declared_sources=item.get("declared_sources", []),
        declared_sources_rationale=item.get("sources_rationale", ""),
    )


def _make_budget_exceeded_trace(seed: SeedCandidate, reason: str) -> ReflectionTrace:
    return ReflectionTrace(
        topic_id=seed.topic.meta.topic_id,
        final_status=FinalStatus.TENTATIVE,
        rounds=[],
        total_cost_usd=0.0,
        total_wallclock_seconds=0.0,
    )


def _should_fallback_level2(exc: Exception) -> bool:
    import yaml as _yaml
    return isinstance(exc, (ValueError, KeyError, _yaml.YAMLError))
