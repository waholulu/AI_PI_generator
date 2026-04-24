"""OpenAlex four-tuple novelty verifier for Module 1 G5 gate."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from agents.cache_utils import build_cache_key, load_json_cache, save_json_cache
from agents.logging_config import get_logger
from models.topic_schema import ExposureFamily, IdentificationPrimary, OutcomeFamily, Topic

logger = get_logger(__name__)

_RATE_LIMIT_INTERVAL = 0.15
_CACHE_HOURS = 24


@dataclass
class NoveltyEvidence:
    total_hits: int
    top_k_papers: list[dict]
    four_tuple_match_count: Optional[int]
    queries_log: list[str]
    was_fallback: bool = False


_EXPOSURE_KEYWORDS: dict[str, list[str]] = {
    ExposureFamily.BUILT_ENVIRONMENT.value: [
        "built environment",
        "urban form",
        "neighborhood design",
        "walkability",
        "street connectivity",
        "land use",
    ],
    ExposureFamily.TRANSPORT_INFRA.value: [
        "transport infrastructure",
        "transit access",
        "road network",
        "cycling infrastructure",
        "public transit",
    ],
    ExposureFamily.GREEN_BLUE_SPACE.value: [
        "green space",
        "park access",
        "urban greenery",
        "blue space",
        "vegetation",
        "tree canopy",
    ],
    ExposureFamily.LAND_USE_MIX.value: ["land use mix", "mixed use", "zoning", "land cover"],
    ExposureFamily.DENSITY.value: ["density", "urban density", "population density", "building density"],
    ExposureFamily.AIR_QUALITY.value: ["air quality", "air pollution", "PM2.5", "NO2", "particulate matter"],
    ExposureFamily.NOISE.value: ["noise", "noise pollution", "traffic noise", "environmental noise"],
    ExposureFamily.HEAT_ISLAND.value: ["heat island", "urban heat", "surface temperature", "thermal comfort"],
    ExposureFamily.DIGITAL_INFRA.value: ["digital infrastructure", "broadband", "internet access", "ICT"],
    ExposureFamily.GOVERNANCE.value: ["governance", "policy", "regulation", "planning", "zoning reform"],
    ExposureFamily.ECONOMIC_ACTIVITY.value: ["economic activity", "employment", "business density", "retail"],
    ExposureFamily.SOCIAL_CAPITAL.value: ["social capital", "social cohesion", "community", "trust"],
}

_OUTCOME_KEYWORDS: dict[str, list[str]] = {
    OutcomeFamily.HEALTH.value: ["health", "obesity", "cardiovascular", "mental health", "mortality", "diabetes"],
    OutcomeFamily.MOBILITY.value: ["mobility", "travel behavior", "mode choice", "commuting", "walking", "cycling"],
    OutcomeFamily.WELLBEING.value: ["wellbeing", "well-being", "life satisfaction", "happiness", "quality of life"],
    OutcomeFamily.ECONOMIC.value: ["economic", "income", "employment", "property value", "rent", "housing price"],
    OutcomeFamily.ENVIRONMENT.value: ["environment", "carbon", "emissions", "energy use", "sustainability"],
    OutcomeFamily.SAFETY.value: ["safety", "crime", "accident", "injury", "violence"],
    OutcomeFamily.EQUITY.value: ["equity", "inequality", "disparities", "access", "social justice"],
    OutcomeFamily.COGNITION.value: ["cognition", "cognitive", "attention", "academic performance", "school outcomes"],
    OutcomeFamily.BEHAVIOR.value: ["behavior", "behaviour", "physical activity", "lifestyle"],
}

_METHOD_KEYWORDS: dict[str, list[str]] = {
    IdentificationPrimary.DID.value: [
        "difference-in-differences",
        "diff-in-diff",
        "difference in differences",
        "did",
    ],
    IdentificationPrimary.RDD.value: ["regression discontinuity", "rdd", "sharp discontinuity"],
    IdentificationPrimary.IV.value: ["instrumental variable", "iv estimation", "two-stage least squares", "2SLS"],
    IdentificationPrimary.PSM.value: ["propensity score matching", "psm", "matching estimator"],
    IdentificationPrimary.FE.value: ["fixed effects", "panel data", "within estimator"],
    IdentificationPrimary.SYNTHETIC_CONTROL.value: ["synthetic control", "abadie", "comparative case study"],
    IdentificationPrimary.EVENT_STUDY.value: ["event study", "event-study", "dynamic treatment effects"],
    IdentificationPrimary.CAUSAL_FOREST.value: ["causal forest", "generalized random forest", "grf"],
    IdentificationPrimary.SPATIAL_REGRESSION.value: ["spatial regression", "spatial econometrics", "spatial lag"],
}


def _expand_family_keywords(family_value: str, mapping: dict[str, list[str]]) -> list[str]:
    return mapping.get(family_value, [family_value])


def _expand_geo_keywords(geography: str) -> list[str]:
    g = (geography or "").lower().strip()
    if g.startswith("metro:") or "metro" in g or "msa" in g or "cbsa" in g:
        return ["metropolitan area", "MSA", "CBSA"]
    if g.startswith("state:"):
        return ["US state", "state"]
    if g in ("conus", "us", "united states"):
        return ["United States", "US"]
    if g.startswith("country:"):
        return [g.split(":", 1)[1]]
    return [geography]


def _papers_match_four_tuple(paper: dict, topic: Topic) -> bool:
    text = " ".join(filter(None, [paper.get("title", ""), paper.get("abstract") or ""])).lower()
    x_kws = _expand_family_keywords(topic.exposure_X.family.value, _EXPOSURE_KEYWORDS)
    y_kws = _expand_family_keywords(topic.outcome_Y.family.value, _OUTCOME_KEYWORDS)
    geo_kws = _expand_geo_keywords(topic.spatial_scope.geography)
    method_kws = _expand_family_keywords(topic.identification.primary.value, _METHOD_KEYWORDS)

    def any_hit(keywords: list[str]) -> bool:
        return any((kw or "").lower() in text for kw in keywords)

    return any_hit(x_kws) and any_hit(y_kws) and any_hit(geo_kws) and any_hit(method_kws)


class OpenAlexVerifier:
    CACHE_NS = "openalex_novelty"

    def __init__(self, top_k: int = 50) -> None:
        self._top_k = top_k
        self._last_call: float = 0.0
        self._configure_pyalex_email()

    @staticmethod
    def _configure_pyalex_email() -> None:
        email = os.getenv("OPENALEX_EMAIL", "").strip()
        if not email:
            logger.warning("OpenAlex polite pool not enabled (set OPENALEX_EMAIL)")
            return
        try:
            from pyalex import config as pyalex_config  # type: ignore[import]

            pyalex_config.email = email
        except Exception as exc:
            logger.warning("Failed to set OpenAlex polite pool email: %s", exc)

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _RATE_LIMIT_INTERVAL:
            time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    def _compose_query(self, topic: Topic) -> str:
        x_terms = _expand_family_keywords(topic.exposure_X.family.value, _EXPOSURE_KEYWORDS)[:2]
        y_terms = _expand_family_keywords(topic.outcome_Y.family.value, _OUTCOME_KEYWORDS)[:2]
        method_terms = _expand_family_keywords(topic.identification.primary.value, _METHOD_KEYWORDS)[:1]
        geo_terms = _expand_geo_keywords(topic.spatial_scope.geography)[:1]
        return " ".join(x_terms[:1] + y_terms[:1] + geo_terms + method_terms)

    @staticmethod
    def _compose_narrower_query(topic: Topic, base_query: str) -> str:
        specific = (topic.exposure_X.specific_variable or "").strip()
        if not specific:
            specific = (topic.outcome_Y.specific_variable or "").strip()
        if not specific:
            return base_query
        return f"{base_query} {specific}"

    @staticmethod
    def _paper_to_dict(paper: dict) -> dict:
        host_venue = paper.get("host_venue") or {}
        venue_name = host_venue.get("display_name") or paper.get("venue", "")
        return {
            "id": paper.get("id", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "publication_year": paper.get("publication_year"),
            "cited_by_count": paper.get("cited_by_count", 0),
            "venue": venue_name,
        }

    def _search_openalex(self, query: str) -> list[dict]:
        from pyalex import Works  # type: ignore[import]

        self._rate_limit()
        return list(
            Works()
            .search(query)
            .sort(cited_by_count="desc")
            .get(per_page=self._top_k)
        )

    def verify_novelty_four_tuple(self, topic: Topic) -> NoveltyEvidence:
        cache_key = build_cache_key("novelty", topic.four_tuple_signature())
        cached = load_json_cache(self.CACHE_NS, cache_key, max_age_hours=_CACHE_HOURS)
        if isinstance(cached, dict):
            return NoveltyEvidence(**cached)

        base_query = self._compose_query(topic)
        queries_log = [base_query]

        try:
            papers = self._search_openalex(base_query)
        except Exception as exc:
            logger.warning("OpenAlex API unavailable: %s — returning fallback", exc)
            return NoveltyEvidence(
                total_hits=0,
                top_k_papers=[],
                four_tuple_match_count=None,
                queries_log=queries_log,
                was_fallback=True,
            )

        top_k = papers[: self._top_k]
        four_tuple_count = sum(1 for p in top_k if _papers_match_four_tuple(p, topic))

        if four_tuple_count == 0 and len(papers) >= 30:
            narrower_query = self._compose_narrower_query(topic, base_query)
            if narrower_query != base_query:
                queries_log.append(narrower_query)
                try:
                    narrower_papers = self._search_openalex(narrower_query)
                    if narrower_papers:
                        papers = narrower_papers
                        top_k = narrower_papers[: self._top_k]
                        four_tuple_count = sum(
                            1 for p in top_k if _papers_match_four_tuple(p, topic)
                        )
                except Exception as exc:
                    logger.warning("OpenAlex narrower query failed: %s", exc)

        evidence = NoveltyEvidence(
            total_hits=len(papers),
            top_k_papers=[self._paper_to_dict(p) for p in top_k],
            four_tuple_match_count=four_tuple_count,
            queries_log=queries_log,
        )
        save_json_cache(
            self.CACHE_NS,
            cache_key,
            {
                "total_hits": evidence.total_hits,
                "top_k_papers": evidence.top_k_papers,
                "four_tuple_match_count": evidence.four_tuple_match_count,
                "queries_log": evidence.queries_log,
                "was_fallback": evidence.was_fallback,
            },
        )
        return evidence
