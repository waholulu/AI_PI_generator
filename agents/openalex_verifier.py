"""
OpenAlex four-tuple novelty verifier for Module 1 G5 gate.

Uses pyalex.Works() directly (not multi_search_openalex wrapper) with a
simple rate limiter.  Keyword expansion maps Topic slot enums to natural-
language terms for dictionary matching against paper titles/abstracts.
Results are cached per topic to avoid redundant API calls across rounds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from agents.cache_utils import build_cache_key, load_json_cache, save_json_cache
from agents.logging_config import get_logger
from models.topic_schema import (
    ExposureFamily,
    IdentificationPrimary,
    OutcomeFamily,
    Topic,
)

logger = get_logger(__name__)

_RATE_LIMIT_INTERVAL = 0.15  # seconds between API calls


@dataclass
class NoveltyEvidence:
    total_hits: int
    top_k_papers: list[dict]
    four_tuple_match_count: int
    queries_log: list[str]
    was_fallback: bool = False


# ── Keyword expansion dictionaries ───────────────────────────────────────────

_EXPOSURE_KEYWORDS: dict[str, list[str]] = {
    ExposureFamily.BUILT_ENVIRONMENT.value: [
        "built environment", "urban form", "neighborhood design",
        "walkability", "street connectivity", "land use",
    ],
    ExposureFamily.TRANSPORT_INFRA.value: [
        "transport infrastructure", "transit access", "road network",
        "cycling infrastructure", "public transit",
    ],
    ExposureFamily.GREEN_BLUE_SPACE.value: [
        "green space", "park access", "urban greenery", "blue space",
        "vegetation", "tree canopy",
    ],
    ExposureFamily.LAND_USE_MIX.value: [
        "land use mix", "mixed use", "zoning", "land cover",
    ],
    ExposureFamily.DENSITY.value: [
        "density", "urban density", "population density", "building density",
    ],
    ExposureFamily.AIR_QUALITY.value: [
        "air quality", "air pollution", "PM2.5", "NO2", "particulate matter",
    ],
    ExposureFamily.NOISE.value: [
        "noise", "noise pollution", "traffic noise", "environmental noise",
    ],
    ExposureFamily.HEAT_ISLAND.value: [
        "heat island", "urban heat", "surface temperature", "thermal comfort",
    ],
    ExposureFamily.DIGITAL_INFRA.value: [
        "digital infrastructure", "broadband", "internet access", "ICT",
    ],
    ExposureFamily.GOVERNANCE.value: [
        "governance", "policy", "regulation", "planning", "zoning reform",
    ],
    ExposureFamily.ECONOMIC_ACTIVITY.value: [
        "economic activity", "employment", "business density", "retail",
    ],
    ExposureFamily.SOCIAL_CAPITAL.value: [
        "social capital", "social cohesion", "community", "trust",
    ],
}

_OUTCOME_KEYWORDS: dict[str, list[str]] = {
    OutcomeFamily.HEALTH.value: [
        "health", "obesity", "cardiovascular", "mental health", "mortality",
        "physical activity", "diabetes",
    ],
    OutcomeFamily.MOBILITY.value: [
        "mobility", "travel behavior", "mode choice", "commuting",
        "walking", "cycling", "transit use",
    ],
    OutcomeFamily.WELLBEING.value: [
        "wellbeing", "well-being", "life satisfaction", "happiness",
        "quality of life",
    ],
    OutcomeFamily.ECONOMIC.value: [
        "economic", "income", "employment", "property value", "rent",
        "housing price",
    ],
    OutcomeFamily.ENVIRONMENT.value: [
        "environment", "carbon", "emissions", "energy use", "sustainability",
    ],
    OutcomeFamily.SAFETY.value: [
        "safety", "crime", "accident", "injury", "violence",
    ],
    OutcomeFamily.EQUITY.value: [
        "equity", "inequality", "disparities", "access", "social justice",
    ],
    OutcomeFamily.COGNITION.value: [
        "cognition", "cognitive", "attention", "academic performance",
        "school outcomes",
    ],
    OutcomeFamily.BEHAVIOR.value: [
        "behavior", "behaviour", "physical activity", "lifestyle",
    ],
}

_METHOD_KEYWORDS: dict[str, list[str]] = {
    IdentificationPrimary.DID.value: [
        "difference-in-differences", "diff-in-diff", "DiD",
        "difference in differences",
    ],
    IdentificationPrimary.RDD.value: [
        "regression discontinuity", "RDD", "sharp discontinuity",
    ],
    IdentificationPrimary.IV.value: [
        "instrumental variable", "IV estimation", "two-stage least squares",
        "2SLS",
    ],
    IdentificationPrimary.PSM.value: [
        "propensity score matching", "PSM", "matching estimator",
    ],
    IdentificationPrimary.FE.value: [
        "fixed effects", "panel data", "within estimator",
    ],
    IdentificationPrimary.SYNTHETIC_CONTROL.value: [
        "synthetic control", "Abadie", "comparative case study",
    ],
    IdentificationPrimary.EVENT_STUDY.value: [
        "event study", "event-study", "dynamic treatment effects",
    ],
    IdentificationPrimary.CAUSAL_FOREST.value: [
        "causal forest", "generalized random forest", "GRF",
        "heterogeneous treatment effects",
    ],
    IdentificationPrimary.SPATIAL_REGRESSION.value: [
        "spatial regression", "spatial econometrics", "spatial lag",
        "spatial error", "geographically weighted regression",
    ],
}


def _expand_family_keywords(family_value: str, mapping: dict) -> list[str]:
    return mapping.get(family_value, [family_value])


def _papers_match_four_tuple(paper: dict, topic: Topic) -> bool:
    """Return True if paper title/abstract contains keywords from all 4 dimensions."""
    text = " ".join(filter(None, [
        paper.get("title", ""),
        (paper.get("abstract") or ""),
    ])).lower()

    x_kws = _expand_family_keywords(topic.exposure_X.family.value, _EXPOSURE_KEYWORDS)
    y_kws = _expand_family_keywords(topic.outcome_Y.family.value, _OUTCOME_KEYWORDS)
    geo_kws = [topic.spatial_scope.geography.lower().strip()]
    method_kws = _expand_family_keywords(
        topic.identification.primary.value, _METHOD_KEYWORDS
    )

    def any_hit(kws: list[str]) -> bool:
        return any(kw.lower() in text for kw in kws)

    return any_hit(x_kws) and any_hit(y_kws) and any_hit(geo_kws) and any_hit(method_kws)


class OpenAlexVerifier:
    """Checks four-tuple novelty via pyalex with caching and rate limiting."""

    CACHE_NS = "openalex_novelty"

    def __init__(self, top_k: int = 50) -> None:
        self._top_k = top_k
        self._last_call: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _RATE_LIMIT_INTERVAL:
            time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    def _compose_query(self, topic: Topic) -> str:
        x_terms = _expand_family_keywords(
            topic.exposure_X.family.value, _EXPOSURE_KEYWORDS
        )[:2]
        y_terms = _expand_family_keywords(
            topic.outcome_Y.family.value, _OUTCOME_KEYWORDS
        )[:2]
        geo = topic.spatial_scope.geography
        parts = x_terms[:1] + y_terms[:1] + [geo]
        return " ".join(parts)

    def verify_novelty_four_tuple(self, topic: Topic) -> NoveltyEvidence:
        cache_key = build_cache_key(self.CACHE_NS, topic.four_tuple_signature())
        cached = load_json_cache(cache_key)
        if cached:
            logger.debug("OpenAlex novelty cache hit for %s", topic.meta.topic_id)
            return NoveltyEvidence(**cached)

        query = self._compose_query(topic)
        queries_log = [query]

        try:
            from pyalex import Works  # type: ignore[import]

            self._rate_limit()
            results = (
                Works()
                .search(query)
                .sort(cited_by_count="desc")
                .get(per_page=self._top_k)
            )
            papers = list(results)
        except Exception as e:
            logger.warning("OpenAlex API unavailable: %s — returning fallback", e)
            evidence = NoveltyEvidence(
                total_hits=0,
                top_k_papers=[],
                four_tuple_match_count=0,
                queries_log=queries_log,
                was_fallback=True,
            )
            return evidence

        top_k = papers[: self._top_k]
        four_tuple_count = sum(1 for p in top_k if _papers_match_four_tuple(p, topic))

        evidence = NoveltyEvidence(
            total_hits=len(papers),
            top_k_papers=[
                {
                    "id": p.get("id", ""),
                    "title": p.get("title", ""),
                    "publication_year": p.get("publication_year"),
                    "cited_by_count": p.get("cited_by_count", 0),
                }
                for p in top_k
            ],
            four_tuple_match_count=four_tuple_count,
            queries_log=queries_log,
        )

        save_json_cache(cache_key, {
            "total_hits": evidence.total_hits,
            "top_k_papers": evidence.top_k_papers,
            "four_tuple_match_count": evidence.four_tuple_match_count,
            "queries_log": evidence.queries_log,
            "was_fallback": evidence.was_fallback,
        })
        return evidence
