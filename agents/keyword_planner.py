"""
KeywordPlanner: uses Gemini fast to reformulate broad domain descriptions
into structured, multi-query search plans for OpenAlex.

Environment variables (all optional, with safe defaults):
  OPENALEX_QUERY_REWRITE_ENABLED   – "true" (default) / "false" to bypass LLM
  OPENALEX_QUERY_REWRITE_MAX_QUERIES – max queries to include in the pool (default 10)
  OPENALEX_QUERY_REWRITE_MODEL     – model name override (default: GEMINI_FAST_MODEL)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class TopicQueryGroup(BaseModel):
    """One thematic sub-topic with its dedicated OpenAlex queries."""

    label: str = Field(description="Short English label for this sub-topic (e.g. 'Causal inference in urban health').")
    queries: List[str] = Field(
        description=(
            "2–4 short English search queries (3–6 words each) optimised for "
            "the OpenAlex full-text search API. No boolean operators."
        )
    )


class KeywordPlan(BaseModel):
    """Structured keyword plan produced by the LLM reformulation step."""

    primary_domains: List[str] = Field(
        description="2–5 English domain/field labels extracted from the input."
    )
    methods: List[str] = Field(
        description="Up to 5 relevant methodological keywords (e.g. 'causal inference', 'GNN')."
    )
    topics: List[TopicQueryGroup] = Field(
        description="One TopicQueryGroup per identifiable sub-topic in the input."
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert research librarian. Your task is to reformulate a broad research
area description into concise, high-quality search queries for the OpenAlex academic
search API.

Rules:
- Always output valid structured data matching the requested schema.
- Extract distinct thematic sub-topics from the input (one TopicQueryGroup each).
- For each sub-topic, generate 2–4 short English queries (3–6 content words each).
- Queries should reflect typical phrasing found in academic titles and abstracts.
- Prefer combinations of [topic] + [method] + [outcome] or [phenomenon].
- Do NOT use boolean operators (AND/OR/NOT) or quotation marks.
- Even if the input is in Chinese or another language, output queries in English.
"""

_USER_PROMPT = """\
Research area description:
{domain_input}

Additional context (may be empty):
{extra_context}

Please produce a KeywordPlan with sub-topics and OpenAlex-ready queries.
"""


class KeywordPlanner:
    """
    Reformulates a free-text domain description into a structured query pool
    using Gemini fast model. Falls back to a single-element pool on failure.
    """

    def __init__(self) -> None:
        self._enabled: bool = os.getenv("OPENALEX_QUERY_REWRITE_ENABLED", "true").lower() not in (
            "false", "0", "no"
        )
        self._max_queries: int = int(os.getenv("OPENALEX_QUERY_REWRITE_MAX_QUERIES", "10"))

        model_name = os.getenv(
            "OPENALEX_QUERY_REWRITE_MODEL",
            os.getenv("GEMINI_FAST_MODEL", "gemini-2.0-flash-lite"),
        )
        self._llm: Optional[Any] = None
        if self._enabled and ChatGoogleGenerativeAI is not None:
            try:
                self._llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
            except Exception as exc:
                print(f"[KeywordPlanner] Failed to initialise LLM ({exc}); will use fallback.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        domain_input: str,
        extra_context: str = "",
    ) -> Dict[str, Any]:
        """
        Returns a planning result dict with keys:
          - query_pool   : List[str]  – deduplicated, length-capped query list
          - topics       : List[dict] – raw TopicQueryGroup dicts (label + queries)
          - primary_domains: List[str]
          - methods      : List[str]
          - used_fallback: bool       – True when LLM was skipped / failed
        """
        if not self._enabled or self._llm is None:
            return self._fallback(domain_input, reason="disabled or LLM unavailable")

        try:
            return self._call_llm(domain_input, extra_context)
        except Exception as exc:
            print(f"[KeywordPlanner] LLM call failed ({exc}); using fallback.")
            return self._fallback(domain_input, reason=str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, domain_input: str, extra_context: str) -> Dict[str, Any]:
        structured_llm = self._llm.with_structured_output(KeywordPlan)  # type: ignore[union-attr]
        prompt = _SYSTEM_PROMPT + "\n\n" + _USER_PROMPT.format(
            domain_input=domain_input,
            extra_context=extra_context or "(none)",
        )
        result: KeywordPlan = structured_llm.invoke(prompt)

        # Collect and deduplicate queries
        seen: set[str] = set()
        pool: List[str] = []
        for tg in result.topics:
            for q in tg.queries:
                q_clean = q.strip()
                if q_clean and q_clean.lower() not in seen:
                    seen.add(q_clean.lower())
                    pool.append(q_clean)

        pool = pool[: self._max_queries]

        return {
            "query_pool": pool,
            "topics": [t.model_dump() for t in result.topics],
            "primary_domains": result.primary_domains,
            "methods": result.methods,
            "used_fallback": False,
        }

    def _fallback(self, domain_input: str, reason: str = "") -> Dict[str, Any]:
        """Return a single-element query pool using the raw domain_input."""
        if reason:
            print(f"[KeywordPlanner] Fallback activated: {reason}")
        return {
            "query_pool": [domain_input],
            "topics": [{"label": domain_input, "queries": [domain_input]}],
            "primary_domains": [],
            "methods": [],
            "used_fallback": True,
        }
