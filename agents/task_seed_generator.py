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

    requires_credentialing: bool = Field(
        default=False,
        description=(
            "Set True when the proposed dataset requires sign-up, IRB, DUA, "
            "or paid licence (MIMIC, UK Biobank, i2b2, EHR data, …). Only "
            "non-False values are allowed when the user has opted in via "
            "--allow-credentialed-data."
        ),
    )
    credentialing_note: str = Field(
        default="",
        description=(
            "When requires_credentialing is True, a one-line summary of the "
            "access process (e.g. 'PhysioNet credentialed access + CITI "
            "training', 'UK Biobank application via AMS portal')."
        ),
    )
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
            "obvious match. MUST point to data that downloads without sign-up, "
            "credentialing, IRB approval, or a paid licence. Example: "
            "'yelp_polarity train split' or 'HuggingFaceH4/no_robots (mit)'."
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

Hard data-access constraint (every task must satisfy this):
- The dataset must download from a public URL with no sign-up, no
  credentialing, no IRB approval, and no paid licence.
- FORBIDDEN sources: clinical notes, electronic health records (EHR/EMR),
  discharge summaries, medical or insurance claims, PhysioNet datasets
  (MIMIC-III/IV, eICU, etc.), i2b2 / n2c2, ADNI, UK Biobank, All of Us,
  individual-level survey microdata, restricted-use research files,
  proprietary news/legal corpora behind paywalls, gated HF Hub repos.
- ALLOWED sources: public HuggingFace Hub datasets under apache-2.0, mit,
  cc-by-4.0, cc-by-sa-4.0, odc-by, or bsd-3-clause; Wikipedia / Wikidata
  dumps; Common Crawl derivatives; Stack Exchange data dumps; PubMed
  Central open-access subset; public consumer-health or social-media
  forum dumps; government open-data portals (CDC PLACES tract-level,
  data.gov, data.gov.uk); curated public benchmarks already on HF Hub.
- When the user's domain naturally implies clinical/EHR data, reach for
  public proxies (open consumer-health forum text, PubMed Central
  abstracts, CDC PLACES outcomes joined to OpenStreetMap features) — do
  not propose tasks that would need credentialed access even if the
  research question would be more clinically authoritative.

Other rules:
- Each task must be a specific predictive problem, not a research theme.
  Bad:  "study built-environment effects on health".
  Good: "Classify built-environment factors mentioned in r/urbanplanning
         Reddit posts (Pushshift dump)".
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

# Opt-in addendum appended to the system prompt when the user has set
# --allow-credentialed-data. Relaxes the hard ban on PhysioNet / UKB / i2b2
# style datasets and instead asks the LLM to mark each such task with
# `requires_credentialing: true` plus a one-line `credentialing_note` so
# the user knows what access process is required.
_CREDENTIALED_OPT_IN_ADDENDUM = """\

OPT-IN OVERRIDE (the user has confirmed they have access):
- The hard data-access constraint above is RELAXED. You MAY propose tasks
  backed by credentialed / gated biomedical datasets when they fit the
  domain — e.g. MIMIC-III/IV (PhysioNet), eICU, i2b2 / n2c2, UK Biobank,
  All of Us, ADNI. These are the dominant benchmarks in top venues
  (NEJM, Lancet Digital Health, JAMIA, Nature Medicine), so prefer them
  over weaker public proxies when the research question would clearly
  benefit.
- For every such task, you MUST set `requires_credentialing: true` and
  fill `credentialing_note` with the access process in <=120 chars
  (examples: "PhysioNet credentialed access + CITI training",
  "UK Biobank application via Access Management System",
  "n2c2 DUA via Harvard DBMI"). Tasks that omit this signalling will be
  rejected downstream.
- You may still propose public-data tasks alongside credentialed ones —
  diversity is preferred. Public-only tasks leave `requires_credentialing`
  at False.
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


# Defence-in-depth against the LLM ignoring the prompt's data-access rule.
# Patterns are tuned to the failure modes observed in production: clinical
# notes / EHRs / credentialed health datasets / individual-level surveys.
# The prompt is the primary guardrail — this filter just stops the worst
# obvious mistakes from reaching the user.
_NON_PUBLIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmimic[\-_ ]?(?:iii|iv|3|4)?\b",
        r"\bphysionet\b",
        r"\beicu\b",
        r"\behrshot\b",
        r"\bi2b2\b",
        r"\bn2c2\b",
        r"\badni\b",
        r"\bukb\b",
        r"\buk[\- ]?biobank\b",
        r"\ball[\- ]of[\- ]us\b",
        r"\bclinical[\- ]notes?\b",
        r"\bdischarge[\- ]summar(y|ies)\b",
        r"\belectronic[\- ]health[\- ]record",
        r"\behr\b",
        r"\bemr\b",
        r"\bmedical[\- ]claims?\b",
        r"\binsurance[\- ]claims?\b",
        r"\birb[\- ]approval\b",
        r"\bcredentialed\b",
        r"\bgated[\- ]dataset\b",
        r"\bindividual[\- ]level[\- ]survey\b",
        r"\bmicrodata\b",
        r"\brestricted[\- ]use\b",
    )
]


def _seed_uses_public_data(seed: TaskSeed) -> tuple[bool, str | None]:
    """Return (is_public, matched_pattern). Inspects label, description, and hint."""
    blob = " ".join((seed.task_label or "", seed.task_description or "", seed.dataset_hint or ""))
    for pat in _NON_PUBLIC_PATTERNS:
        m = pat.search(blob)
        if m:
            return False, m.group(0)
    return True, None


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

    def __init__(self, allow_credentialed_data: bool = False) -> None:
        self._enabled: bool = os.getenv(
            "LLM_DOMAIN_TASK_GENERATION_ENABLED", "true"
        ).lower() not in ("false", "0", "no")
        # CLI flag wins; env var is a fallback for non-CLI callers.
        self._allow_credentialed: bool = bool(allow_credentialed_data) or os.getenv(
            "AUTOPI_ALLOW_CREDENTIALED_DATA", "0"
        ).strip().lower() in ("1", "true", "yes")
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
            system = _SYSTEM_PROMPT + (
                _CREDENTIALED_OPT_IN_ADDENDUM if self._allow_credentialed else ""
            )
            prompt = system + "\n\n" + _USER_PROMPT.format(
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

            # Inspect every seed against the denylist. Strict mode drops
            # matches; opt-in mode labels them so the UI / dev pack can
            # surface the credentialing step.
            kept: list[TaskSeed] = []
            for t in deduped:
                ok, hit = _seed_uses_public_data(t)
                if ok:
                    kept.append(t)
                    continue
                if self._allow_credentialed:
                    t.requires_credentialing = True
                    if not t.credentialing_note:
                        t.credentialing_note = (
                            f"Credentialed access required (matched: {hit})"
                        )
                    logger.info(
                        "TaskSeedGenerator: labelled credentialed seed "
                        "task_id=%r (matched %r)",
                        t.task_id, hit,
                    )
                    kept.append(t)
                else:
                    logger.warning(
                        "TaskSeedGenerator: dropped non-public seed "
                        "task_id=%r (matched %r) — set "
                        "--allow-credentialed-data to keep these",
                        t.task_id, hit,
                    )

            if not kept:
                logger.warning(
                    "TaskSeedGenerator: every LLM seed referenced "
                    "non-public data — using fallback for domain=%r",
                    domain_input,
                )
                return _fallback_seeds(domain_input)

            # If too few seeds survived, top up from the generic fallback so
            # the user still gets coverage across the three metric families.
            if len(kept) < 3:
                existing_ids = {s.task_id for s in kept}
                for fb in _fallback_seeds(domain_input):
                    if fb.task_id not in existing_ids:
                        kept.append(fb)

            return kept[:10]
        except Exception as exc:
            logger.warning(
                "TaskSeedGenerator: LLM call failed (%s) — using fallback", exc
            )
            return _fallback_seeds(domain_input)
