"""Domain-grounded task seed generator for training_research templates.

Given a free-text `domain_input` (e.g. "Built environment exposure and health
outcomes"), produces 5-10 concrete supervised-learning tasks that an LLM
could be fine-tuned on, e.g.:

  - Classify clinical-note spans that describe built-environment exposures
  - Predict asthma risk from neighborhood walkability and air-pollution text
  - Extract green-space mentions from social-media posts

The candidate composer then takes the Cartesian product of
{sft_full_finetune, lora_adapter, qlora_4bit} × <generated tasks>, so the
research questions surfaced at the HITL checkpoint are about applying LLM
techniques *to the user's domain* rather than about LLM training in the
abstract.

Falls back to a single generic task derived from `domain_input` when Gemini
is unavailable, the key is missing, or the call fails — keeps the pipeline
deterministic and the unit tests green without API keys.
"""
from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.logging_config import get_logger

logger = get_logger(__name__)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore[assignment, misc]


MetricFamily = Literal["task_accuracy", "instruction_following", "generation_quality"]
Modality = Literal[
    "text_classification",
    "sequence_labeling",
    "extraction",
    "generation",
    "summarization",
    "question_answering",
]


class TaskSeed(BaseModel):
    """One concrete supervised-learning task derived from a domain description."""

    task_id: str = Field(
        description=(
            "Short snake_case identifier, max 40 chars. Must be unique within "
            "the generated list. Example: 'be_exposure_extraction_clinical_notes'."
        )
    )
    task_label: str = Field(
        description=(
            "Human-readable title for the UI card. Title-case, 4-12 words. "
            "Example: 'Built-Environment Exposure Extraction from Clinical Notes'."
        )
    )
    task_description: str = Field(
        description=(
            "One or two sentences explaining what the model predicts, what "
            "inputs it consumes, and why it matters for the user's domain. "
            "No mention of training mechanics — that lives on the X axis."
        )
    )
    modality: Modality = Field(
        description=(
            "Coarse task type. Drives the eval-harness mapping in the dev pack."
        )
    )
    metric_family: MetricFamily = Field(
        description=(
            "Which evaluation metric family applies. classification/extraction "
            "→ task_accuracy; open-ended-instruction → instruction_following; "
            "summarization/generation → generation_quality."
        )
    )
    dataset_hint: str = Field(
        description=(
            "Free-text hint about a candidate dataset on HuggingFace Hub or a "
            "public corpus the user could fine-tune on. May be empty if no "
            "obvious match. Example: 'i2b2 2010 clinical concept extraction, "
            "MIMIC-III discharge summaries'."
        ),
        default="",
    )


class TaskSeedPlan(BaseModel):
    """Structured plan returned by the LLM."""

    tasks: list[TaskSeed] = Field(
        description="5 to 10 diverse supervised-learning tasks for the domain."
    )


_SYSTEM_PROMPT = """\
You are a research methods adviser for a small lab that fine-tunes open-source
language models on domain-specific tasks. The user has named a research
domain. Your job is to enumerate 5-10 *concrete supervised-learning tasks*
inside that domain that an LLM could plausibly be fine-tuned on with a small
public dataset and a Colab GPU.

Rules:
- Each task must be a specific predictive problem, not a research theme.
  Bad: "study built-environment effects on health".
  Good: "Classify mentions of built-environment exposures in clinical notes".
- Each task should have inputs that are plausibly available as public or
  semi-public text/tabular data on HuggingFace Hub.
- Stay inside the user's domain. Do NOT generalize to LLM benchmarking
  tasks (MMLU, IFEval, etc.) unless the user explicitly asked about those.
- Spread coverage across modalities: at least two of
  {classification, extraction, generation/summarization}.
- Choose metric_family honestly: f1/accuracy → task_accuracy; following an
  instruction template → instruction_following; free-form generation /
  summarization → generation_quality.
- Be concise. Each task_description is one or two sentences max.
- task_id is snake_case, 40 chars max, unique.
"""

_USER_PROMPT = """\
Research domain (free text):
{domain_input}

Optional context:
{extra_context}

Produce a TaskSeedPlan with 5-10 tasks the lab could fine-tune an open LM on.
"""


def _slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len] or "domain_task"


def _fallback_seeds(domain_input: str) -> list[TaskSeed]:
    """One generic task per metric family when the LLM is unavailable.

    Three placeholder tasks (classification, instruction-following,
    generation) — labels make it clear they're fallback so the user knows to
    re-run with GEMINI_API_KEY set for real domain grounding. Keeps the
    candidate pool large enough to be useful for review even without an LLM.
    """
    base_slug = _slugify(domain_input) or "domain"
    domain_label = (domain_input.strip() or "your domain").rstrip(".")
    return [
        TaskSeed(
            task_id=f"{base_slug}_classification",
            task_label=f"Classification Task in {domain_label.title()}",
            task_description=(
                f"Train an LM to classify text inputs related to "
                f"\"{domain_label}\". Placeholder task — Gemini was unavailable, "
                f"so no domain-specific decomposition was performed."
            ),
            modality="text_classification",
            metric_family="task_accuracy",
            dataset_hint="",
        ),
        TaskSeed(
            task_id=f"{base_slug}_instruction_following",
            task_label=f"Instruction-Following Task in {domain_label.title()}",
            task_description=(
                f"Train an LM to follow constrained instructions in "
                f"\"{domain_label}\" (placeholder — set GEMINI_API_KEY for "
                f"domain-grounded task generation)."
            ),
            modality="generation",
            metric_family="instruction_following",
            dataset_hint="",
        ),
        TaskSeed(
            task_id=f"{base_slug}_generation",
            task_label=f"Generation Task in {domain_label.title()}",
            task_description=(
                f"Train an LM to produce free-form text in "
                f"\"{domain_label}\" (placeholder — set GEMINI_API_KEY for "
                f"domain-grounded task generation)."
            ),
            modality="generation",
            metric_family="generation_quality",
            dataset_hint="",
        ),
    ]


class TaskSeedGenerator:
    """LLM-backed domain-to-tasks expander.

    Public API:
      generate(domain_input, extra_context="") -> list[TaskSeed]

    Always returns at least one TaskSeed. Never raises — falls back to a
    single placeholder task on any failure so the pipeline stays
    deterministic.
    """

    def __init__(self) -> None:
        self._enabled: bool = os.getenv(
            "LLM_DOMAIN_TASK_GENERATION_ENABLED", "true"
        ).lower() not in ("false", "0", "no")
        model_name = os.getenv(
            "LLM_DOMAIN_TASK_MODEL",
            os.getenv("GEMINI_FAST_MODEL", "gemini-2.0-flash-lite"),
        )
        self._llm: Any | None = None
        if (
            self._enabled
            and ChatGoogleGenerativeAI is not None
            and os.getenv("GEMINI_API_KEY")
        ):
            try:
                self._llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.4)
            except Exception as exc:
                logger.warning("TaskSeedGenerator: LLM init failed (%s)", exc)
                self._enabled = False

    def generate(
        self, domain_input: str, extra_context: str = ""
    ) -> list[TaskSeed]:
        if not domain_input or not domain_input.strip():
            return _fallback_seeds("")

        if not self._enabled or self._llm is None:
            logger.info(
                "TaskSeedGenerator: disabled or no LLM — falling back to "
                "single generic task for domain=%r",
                domain_input,
            )
            return _fallback_seeds(domain_input)

        try:
            structured = self._llm.with_structured_output(TaskSeedPlan)
            prompt = _SYSTEM_PROMPT + "\n\n" + _USER_PROMPT.format(
                domain_input=domain_input.strip(),
                extra_context=extra_context.strip() or "(none)",
            )
            result = structured.invoke(prompt)
            tasks = list(result.tasks) if result and result.tasks else []
            if not tasks:
                logger.warning(
                    "TaskSeedGenerator: LLM returned 0 tasks — using fallback"
                )
                return _fallback_seeds(domain_input)
            # Dedupe task_id; keep first occurrence.
            seen: set[str] = set()
            deduped: list[TaskSeed] = []
            for t in tasks:
                tid = _slugify(t.task_id or t.task_label)
                if tid in seen:
                    continue
                seen.add(tid)
                t.task_id = tid
                deduped.append(t)
            return deduped[:10]
        except Exception as exc:
            logger.warning(
                "TaskSeedGenerator: LLM call failed (%s) — using fallback", exc
            )
            return _fallback_seeds(domain_input)
