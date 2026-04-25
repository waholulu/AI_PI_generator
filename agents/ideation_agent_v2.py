"""
IdeationAgentV2 — dual-mode topic ideation (Level 1 + Level 2).

Level 1:
  - validate user topic
  - run one reflection pass
  - evaluate candidate once
  - emit contract-valid research_plan.json

Level 2:
  - default AUTOPI_IDEATION_MODE=simple: one-pass candidate generation + evaluation
  - optional AUTOPI_IDEATION_MODE=reflection: keep legacy reflection path
  - all emitted candidates include `evaluation`
  - research_plan.json is always validated against ResearchPlan schema
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Optional

import yaml

from agents.candidate_evaluator import evaluate_candidate
from agents.budget_tracker import BudgetExceededError, BudgetTracker
from agents.logging_config import get_logger
from agents.memory_retriever import MemoryRetriever
from agents.openalex_verifier import OpenAlexVerifier
from agents.reflection_loop import ReflectionTrace, run_reflection_loop
from agents.research_plan_builder import build_research_plan_from_candidate
from agents.rule_engine import RuleEngine
from agents import settings
from models.topic_schema import (
    FinalStatus,
    HITLInterruption,
    SeedCandidate,
    Topic,
)
from models.research_plan_schema import ResearchPlan

logger = get_logger(__name__)


def _load_reflection_models_config() -> dict:
    try:
        with open(settings.reflection_config_path(), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("models", {})
    except Exception as exc:
        logger.warning("Failed to load reflection model config: %s", exc)
        return {}


def _load_prompt_enums_block() -> str:
    enums_path = settings.prompts_dir() / "_enums.txt"
    if enums_path.exists():
        return enums_path.read_text(encoding="utf-8")

    try:
        from models.topic_schema import (
            ContributionPrimary,
            ExposureFamily,
            Frequency,
            IdentificationPrimary,
            OutcomeFamily,
            SamplingMode,
        )
        blocks = []
        for enum_cls in (
            ExposureFamily,
            OutcomeFamily,
            SamplingMode,
            Frequency,
            IdentificationPrimary,
            ContributionPrimary,
        ):
            values = ", ".join(member.value for member in enum_cls)
            blocks.append(f"{enum_cls.__name__}: {values}")
        return "\n".join(blocks)
    except Exception:
        return ""


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

    # Collapse to legacy six-gate format required by the upgrade plan.
    return {
        "impact": gate_map["G1"]["passed"],
        "quantitative": True,
        "novelty": gate_map["G5"]["passed"],
        "publishability": gate_map["G7"]["passed"],
        "automation": gate_map["G6"]["passed"],
        "data_availability": gate_map["G3"]["passed"],
        "full_seven_gates": {k: v for k, v in gate_map.items()},
    }


class _UserInputError(Exception):
    pass


class IdeationSeedGenerationError(RuntimeError):
    """Raised when Level 2 cannot produce any seed candidates from the LLM.

    Surfaces a clear cause (missing API key, unparseable model output, etc.)
    instead of silently emitting placeholder "Fallback topic" stubs that would
    then pass the reflection loop and leak into the HITL picker.
    """


_PLACEHOLDER_TITLE_MARKERS = ("fallback topic", "fallback_")


def _is_placeholder_candidate(candidate: dict) -> bool:
    title = str(candidate.get("title", "")).lower()
    topic_id = str(candidate.get("topic_id") or candidate.get("reflection_trace_id") or "").lower()
    return any(m in title for m in _PLACEHOLDER_TITLE_MARKERS) or topic_id.startswith("fallback_")


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
        self.ideation_mode = os.getenv("AUTOPI_IDEATION_MODE", "simple").strip().lower()
        if self.ideation_mode not in {"simple", "reflection"}:
            self.ideation_mode = "simple"

    def _init_llm(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            models_cfg = _load_reflection_models_config()
            seed_model = models_cfg.get(
                "seed_generation", os.getenv("GEMINI_FAST_MODEL", "gemini-2.0-flash-lite")
            )
            return ChatGoogleGenerativeAI(model=seed_model, temperature=0.7)
        except Exception as e:
            logger.warning("LLM init failed: %s — running without LLM", e)
            return None

    @staticmethod
    def _verdict_rank(verdict: str) -> int:
        return {"pass": 2, "warning": 1, "fail": 0}.get(verdict, 0)

    def _evaluate_candidate_entry(self, candidate: dict, run_id: str) -> tuple[dict, dict]:
        """Build a contract-valid plan + deterministic candidate evaluation."""
        bootstrap_plan = build_research_plan_from_candidate(candidate, evaluation=None, run_id=run_id)
        evaluation = evaluate_candidate(candidate, bootstrap_plan, llm=self._llm)
        candidate["evaluation"] = evaluation.model_dump()
        final_plan = build_research_plan_from_candidate(
            candidate, evaluation=candidate["evaluation"], run_id=run_id
        )
        validated = ResearchPlan.model_validate(final_plan.model_dump())
        return candidate, validated.model_dump()

    def _run_simple_screening(
        self,
        seeds: list[SeedCandidate],
        run_id: str,
        top_k: int = 3,
    ) -> tuple[list[dict], list[dict]]:
        ranked: list[dict] = []
        for seed in seeds:
            entry = {
                **seed.topic.to_legacy_dict(),
                "rank": 0,
                "final_status": "EVALUATED",
                "declared_sources": seed.declared_sources,
                "legacy_six_gates": {},
                "reflection_trace_id": seed.topic.meta.topic_id,
            }
            if _is_placeholder_candidate(entry):
                continue
            evaluated, _ = self._evaluate_candidate_entry(entry, run_id)
            ranked.append(evaluated)

        ranked.sort(
            key=lambda c: (
                self._verdict_rank((c.get("evaluation") or {}).get("overall_verdict", "fail")),
                float((c.get("evaluation") or {}).get("score", 0.0)),
            ),
            reverse=True,
        )
        for i, candidate in enumerate(ranked):
            candidate["rank"] = i + 1
        return ranked[:top_k], ranked

    def _persist_screening_and_plan(self, run_id: str, screening: dict, top_candidate: dict) -> tuple[str, str]:
        screening_path = settings.topic_screening_path()
        with open(screening_path, "w", encoding="utf-8") as f:
            json.dump(screening, f, indent=2, ensure_ascii=False)

        _, plan_dict = self._evaluate_candidate_entry(top_candidate, run_id)
        plan_path = settings.research_plan_path()
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_dict, f, indent=2, ensure_ascii=False)
        return screening_path, plan_path

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

        _, plan = self._evaluate_candidate_entry(candidate_entry, run_id)
        plan_path = settings.research_plan_path()
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        return {
            "execution_status": "harvesting",
            "candidate_topics_path": screening_path,
            "current_plan_path": plan_path,
            "degraded_nodes": [],
        }

    # ── Level 2 ───────────────────────────────────────────────────────────────

    def run_level2(self, state: dict) -> dict:
        """Level 2: default simple one-pass screening, optional reflection mode."""
        domain = state.get("domain_input", "Urban Planning and Health")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        t_start_dt = datetime.now(timezone.utc)
        t_start = t_start_dt.timestamp()

        field_scan_path = state.get("field_scan_path", settings.field_scan_path())
        field_scan_context = _load_field_scan_context(field_scan_path)
        memory_context = _load_memory_context(self.memory, domain)

        seeds = self._generate_seeds(domain, field_scan_context, memory_context)
        simple_limit = int(os.getenv("IDEATION_SIMPLE_CANDIDATE_COUNT", "20"))
        seeds = seeds[: max(1, simple_limit)]

        if self.ideation_mode != "reflection":
            ranked_top, ranked_all = self._run_simple_screening(seeds, run_id=run_id, top_k=3)
            t_end = datetime.now(timezone.utc).timestamp()
            return self._emit_level2_outputs_simple(
                top_candidates=ranked_top,
                all_candidates=ranked_all,
                domain=domain,
                run_id=run_id,
                started_at=t_start_dt,
                wallclock=t_end - t_start,
            )

        accepted: list[tuple[SeedCandidate, ReflectionTrace]] = []
        tentative: list[tuple[SeedCandidate, ReflectionTrace]] = []
        rejected: list[tuple[SeedCandidate, ReflectionTrace]] = []
        accepted, tentative, rejected = self._run_reflection_batch(seeds)
        top_candidates = self._rank_and_select(accepted, domain)[:3]
        self._write_tentative_pool(tentative, run_id)

        t_end = datetime.now(timezone.utc).timestamp()
        return self._emit_level2_outputs_reflection(
            top_candidates, tentative, rejected, len(accepted), domain, run_id, t_start_dt,
            wallclock=t_end - t_start,
            total_attempted=len(seeds),
        )

    def _run_reflection_batch(
        self, seeds: list[SeedCandidate]
    ) -> tuple[
        list[tuple[SeedCandidate, ReflectionTrace]],
        list[tuple[SeedCandidate, ReflectionTrace]],
        list[tuple[SeedCandidate, ReflectionTrace]],
    ]:
        """Run one reflection pass over all seeds."""
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
                    logger.warning("Reflection loop error for %s: %s", seed.topic.meta.topic_id, e)
        return accepted, tentative, rejected

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
        """Generate SeedCandidates from the LLM.

        Raises IdeationSeedGenerationError when no valid seeds can be produced,
        so the pipeline surfaces a clear failure instead of silently emitting
        placeholder stubs.
        """
        if self._llm is None:
            raise IdeationSeedGenerationError(
                "Ideation LLM is not configured — cannot generate seed topics. "
                "Verify GEMINI_API_KEY is set in the runtime environment."
            )
        try:
            seeds = self._llm_generate_seeds(domain, field_scan_context, memory_context)
        except IdeationSeedGenerationError:
            raise
        except Exception as e:
            raise IdeationSeedGenerationError(
                f"LLM seed generation failed: {e}"
            ) from e
        if not seeds:
            raise IdeationSeedGenerationError(
                "LLM returned no parseable seed candidates for the given domain."
            )
        return seeds

    def _llm_generate_seeds(
        self, domain: str, field_scan_context: str, memory_context: str
    ) -> list[SeedCandidate]:
        if self._llm is None:
            raise IdeationSeedGenerationError("LLM unavailable for seed generation.")

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
            enum_block=_load_prompt_enums_block(),
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
            if _is_placeholder_candidate(entry):
                logger.warning(
                    "Dropping placeholder candidate from ranking: %s",
                    entry.get("title") or entry.get("topic_id"),
                )
                continue
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

    def _emit_level2_outputs_reflection(
        self,
        top_candidates: list[dict],
        tentative: list,
        rejected: list,
        accepted_count: int,
        domain: str,
        run_id: str,
        started_at: datetime,
        wallclock: float,
        total_attempted: int,
    ) -> dict:
        if not top_candidates:
            if tentative or rejected:
                logger.warning(
                    "No ACCEPTED candidates after rerun — evaluating near-pass candidates"
                )
                fallback_candidates: list[dict] = []
                for seed, trace in (tentative + rejected):
                    legacy_gates = _build_legacy_gates_map(trace)
                    last_round = trace.rounds[-1] if trace.rounds else None
                    score = last_round.round_score if last_round else 0.0
                    entry = {
                        **seed.topic.to_legacy_dict(),
                        "rank": 0,
                        "final_status": trace.final_status.value,
                        "declared_sources": seed.declared_sources,
                        "legacy_six_gates": legacy_gates,
                        "reflection_trace_id": seed.topic.meta.topic_id,
                        "_reflection_score": score,
                    }
                    if _is_placeholder_candidate(entry):
                        continue
                    evaluated, _ = self._evaluate_candidate_entry(entry, run_id)
                    fallback_candidates.append(evaluated)
                fallback_candidates.sort(
                    key=lambda c: (
                        self._verdict_rank((c.get("evaluation") or {}).get("overall_verdict", "fail")),
                        float((c.get("evaluation") or {}).get("score", 0.0)),
                        float(c.get("_reflection_score", 0.0)),
                    ),
                    reverse=True,
                )
                for i, candidate in enumerate(fallback_candidates):
                    candidate["rank"] = i + 1
                    candidate.pop("_reflection_score", None)
                top_candidates = fallback_candidates[:3]

        if not top_candidates:
            raise IdeationSeedGenerationError(
                f"No usable topics generated for domain '{domain}' after all retries — "
                "aborting rather than emitting a placeholder."
            )

        evaluated_candidates: list[dict] = []
        first_plan: dict | None = None
        for candidate in top_candidates:
            evaluated, plan_dict = self._evaluate_candidate_entry(candidate, run_id)
            evaluated_candidates.append(evaluated)
            if first_plan is None:
                first_plan = plan_dict

        screening = {
            "run_id": run_id,
            "input_mode": "level_2",
            "ideation_mode": self.ideation_mode,
            "domain": domain,
            "candidates": evaluated_candidates,
        }
        screening_path = settings.topic_screening_path()
        with open(screening_path, "w") as f:
            json.dump(screening, f, indent=2, ensure_ascii=False)

        plan = first_plan or {}
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
            "input_summary": domain,
            "total_topics_attempted": total_attempted,
            "status_breakdown": {
                "ACCEPTED": accepted_count,
                "TENTATIVE": len(tentative),
                "REJECTED": len(rejected),
            },
            "total_cost_usd": self.budget.snapshot()["per_run_spent_usd"],
            "total_wallclock_seconds": wallclock,
            "trace_files": [
                str(settings.ideation_traces_dir() / f"{c.get('topic_id', c.get('reflection_trace_id', '?'))}_trace.json")
                for c in evaluated_candidates
            ],
        }
        summary_path = settings.ideation_run_summary_path()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        context_path = settings.research_context_path()
        context = {
            "domain": domain,
            "run_id": run_id,
            "selected_topic": evaluated_candidates[0] if evaluated_candidates else {},
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

    def _emit_level2_outputs_simple(
        self,
        top_candidates: list[dict],
        all_candidates: list[dict],
        domain: str,
        run_id: str,
        started_at: datetime,
        wallclock: float,
    ) -> dict:
        if not top_candidates:
            raise IdeationSeedGenerationError(
                f"No usable topics generated for domain '{domain}' in simple mode."
            )

        evaluated_candidates: list[dict] = []
        first_plan: dict | None = None
        for candidate in top_candidates:
            evaluated, plan_dict = self._evaluate_candidate_entry(candidate, run_id)
            evaluated_candidates.append(evaluated)
            if first_plan is None:
                first_plan = plan_dict

        screening = {
            "run_id": run_id,
            "input_mode": "level_2",
            "ideation_mode": self.ideation_mode,
            "domain": domain,
            "candidates": evaluated_candidates,
            "all_candidates_count": len(all_candidates),
        }
        screening_path = settings.topic_screening_path()
        with open(screening_path, "w", encoding="utf-8") as f:
            json.dump(screening, f, indent=2, ensure_ascii=False)

        plan_path = settings.research_plan_path()
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(first_plan or {}, f, indent=2, ensure_ascii=False)

        status_counts = {"pass": 0, "warning": 0, "fail": 0}
        for candidate in all_candidates:
            verdict = (candidate.get("evaluation") or {}).get("overall_verdict", "fail")
            if verdict in status_counts:
                status_counts[verdict] += 1

        summary = {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "input_mode": "level_2",
            "ideation_mode": self.ideation_mode,
            "input_summary": domain,
            "total_topics_attempted": len(all_candidates),
            "status_breakdown": status_counts,
            "total_cost_usd": self.budget.snapshot()["per_run_spent_usd"],
            "total_wallclock_seconds": wallclock,
        }
        summary_path = settings.ideation_run_summary_path()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        context_path = settings.research_context_path()
        context = {
            "domain": domain,
            "run_id": run_id,
            "selected_topic": evaluated_candidates[0] if evaluated_candidates else {},
        }
        with open(context_path, "w", encoding="utf-8") as f:
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


def _dict_to_seed_candidate(item: dict, domain: str, idx: int) -> SeedCandidate:
    from agents.seed_normalizer import (
        SeedNormalizationError,
        normalize_family,
        normalize_method,
        normalize_mitigations,
    )
    from models.topic_schema import (
        Contribution, ContributionPrimary, ExposureFamily, ExposureX,
        Frequency, IdentificationPrimary, IdentificationStrategy, OutcomeFamily,
        OutcomeY, SamplingMode, SpatialScope, TemporalScope, TopicMeta,
    )

    def _normalize_enum(enum_cls, value: str):
        normalized = str(value or "").strip().lower().replace("-", "_")
        return enum_cls(normalized)

    def _as_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]

    try:
        x_family = normalize_family(
            item.get("exposure_family") or item.get("exposure", ""),
            ExposureFamily,
        )
        y_family = normalize_family(
            item.get("outcome_family") or item.get("outcome", ""),
            OutcomeFamily,
        )
        method = normalize_method(item.get("method", ""))
        contribution_primary = normalize_family(
            item.get("contribution_type", "novel_context"),
            ContributionPrimary,
        )
        sampling = _normalize_enum(
            SamplingMode,
            item.get("sampling_mode", "panel"),
        )
        frequency = _normalize_enum(
            Frequency,
            item.get("frequency", "annual"),
        )
    except (SeedNormalizationError, ValueError) as e:
        raise SeedNormalizationError(f"seed[{idx}] normalization failed: {e}") from e

    x_spatial_unit = (
        item.get("exposure_spatial_unit")
        or item.get("spatial_unit")
        or "tract"
    )
    y_spatial_unit = (
        item.get("outcome_spatial_unit")
        or item.get("spatial_unit")
        or "tract"
    )
    key_threats = _as_list(
        item.get("key_threats") or item.get("threats") or ["confounding"]
    )
    mitigations = normalize_mitigations(item.get("mitigations"), key_threats)
    target_venues = _as_list(item.get("target_venues"))[:5]

    tid = item.get("topic_id") or f"seed_{idx:03d}"
    topic = Topic(
        meta=TopicMeta(topic_id=tid),
        exposure_X=ExposureX(
            family=x_family,
            specific_variable=item.get("exposure_specific", item.get("exposure", f"exposure_{idx}")),
            spatial_unit=x_spatial_unit,
            measurement_proxy=item.get("exposure_proxy", ""),
        ),
        outcome_Y=OutcomeY(
            family=y_family,
            specific_variable=item.get("outcome_specific", item.get("outcome", f"outcome_{idx}")),
            spatial_unit=y_spatial_unit,
            measurement_proxy=item.get("outcome_proxy", ""),
        ),
        spatial_scope=SpatialScope(
            geography=item.get("geography", domain[:50]),
            spatial_unit=item.get("spatial_unit", "tract"),
            sampling_mode=sampling,
        ),
        temporal_scope=TemporalScope(
            start_year=item.get("start_year", 2010),
            end_year=item.get("end_year", 2020),
            frequency=frequency,
        ),
        identification=IdentificationStrategy(
            primary=method,
            key_threats=key_threats,
            mitigations=mitigations,
            requires_exogenous_shock=bool(item.get("requires_exogenous_shock", False)),
        ),
        contribution=Contribution(
            primary=contribution_primary,
            statement=item.get("contribution_statement", item.get("notes", f"Topic {idx} for {domain}")),
            gap_addressed=item.get("gap_addressed", ""),
        ),
        target_venues=target_venues,
        free_form_title=item.get("title", f"Topic {idx}"),
    )
    return SeedCandidate(
        topic=topic,
        declared_sources=item.get("declared_sources", []),
        declared_sources_rationale=item.get("sources_rationale", ""),
    )


def _make_budget_exceeded_trace(seed: SeedCandidate, reason: str) -> ReflectionTrace:
    from agents.reflection_loop import ReflectionTrace

    return ReflectionTrace(
        topic_id=seed.topic.meta.topic_id,
        seed_version=seed.topic.model_dump(),
        final_status=FinalStatus.TENTATIVE,
        rounds=[],
        reject_reasons=[reason],
        convergence={"score_trajectory": [], "signature_history": [], "early_stop_reason": reason},
        design_alternatives_considered=[],
        total_cost_usd=0.0,
        total_wallclock_seconds=0.0,
        final_topic=seed.topic.model_dump(),
    )


def _should_fallback_level2(exc: Exception) -> bool:
    import yaml as _yaml
    return isinstance(exc, (ValueError, KeyError, _yaml.YAMLError))
