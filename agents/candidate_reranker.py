"""Lightweight candidate reranking and display polish.

This module deliberately avoids additional network calls.  It takes the
deterministic candidate-factory pool and adds a small research-value layer:
domain fit, identification strength, novelty potential, and title quality.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_GEOAI_TERMS = {
    "geoai", "urban", "planning", "remote", "sensing", "satellite", "land",
    "mobility", "transport", "street", "network", "spatial", "geospatial",
    "city", "cities", "built", "environment",
}

_GEOAI_TECH = {
    "remote_sensing", "osmnx", "building_footprint", "gtfs", "viirs",
    "geospatial", "vision",
}

_STRONG_METHOD_TERMS = {
    "difference_in_differences", "event_study", "panel_fixed_effects",
    "spatial_discontinuity", "instrumental_variable", "synthetic_control",
}


def _tokens(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {tok for tok in normalized.split() if tok}


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _field_scan_text(field_scan_summary: dict | None) -> str:
    if not field_scan_summary:
        return ""
    strategy = field_scan_summary.get("search_strategy") or {}
    parts: list[str] = []
    parts.extend(strategy.get("query_pool") or [])
    parts.extend(strategy.get("methods") or [])
    parts.extend(strategy.get("primary_domains") or [])
    for topic in strategy.get("topics") or []:
        if isinstance(topic, dict):
            parts.append(str(topic.get("label", "")))
            parts.extend(topic.get("queries") or [])
    return " ".join(str(p) for p in parts)


def _domain_fit(card: dict, domain_tokens: set[str], scan_tokens: set[str]) -> float:
    tags = set(card.get("technology_tags") or [])
    source_blob = " ".join(
        str(card.get(k, ""))
        for k in ("exposure_source", "outcome_source", "exposure_label", "outcome_label")
    )
    card_tokens = _tokens(source_blob) | {t.lower() for t in tags}
    wanted = domain_tokens | scan_tokens
    overlap = len(card_tokens & wanted) / max(1, min(8, len(wanted)))
    score = 0.45 + min(overlap, 0.25)
    if tags & _GEOAI_TECH:
        score += 0.18
    exp = str(card.get("exposure_label") or "").lower()
    if ("geoai" in domain_tokens or "planning" in domain_tokens) and tags & {"remote_sensing", "viirs"}:
        score += 0.12
        if exp in {"nighttime_lights", "impervious_surface", "building_density", "tree_canopy", "land_cover"}:
            score += 0.12
        if exp in {"green_space", "park_access", "ndvi"}:
            score -= 0.08
    if ("urban" in domain_tokens or "planning" in domain_tokens) and tags & {"osmnx", "gtfs"}:
        score += 0.10
    return _clamp(score)


def _identification_strength(card: dict) -> float:
    method = str(card.get("method", "")).lower()
    if any(term in method for term in _STRONG_METHOD_TERMS):
        return 0.82
    if "fixed_effect" in method or "panel" in method:
        return 0.68
    if "cross_sectional" in method:
        return 0.42
    return 0.50


def _novelty_potential(card: dict, domain_tokens: set[str]) -> float:
    scores = card.get("scores") or {}
    base = float(scores.get("novelty", 0.6) or 0.6)
    tags = set(card.get("technology_tags") or [])
    if tags & {"remote_sensing", "viirs", "building_footprint"}:
        base += 0.12
    if tags & {"osmnx", "gtfs"}:
        base += 0.07
    if "geoai" in domain_tokens and tags & _GEOAI_TECH:
        base += 0.06
    return _clamp(base)


def _title_quality(card: dict) -> float:
    display = card.get("display") or {}
    title = display.get("display_title") or card.get("title", "")
    if not title:
        return 0.25
    score = 0.45
    if "?" in title:
        score += 0.15
    if len(title.split()) >= 8:
        score += 0.12
    if " and " not in title.lower():
        score += 0.08
    return _clamp(score)


def _readiness_penalty(card: dict) -> float:
    readiness = card.get("readiness") or card.get("shortlist_status")
    if readiness in {"ready", "ready_after_auto_fix"}:
        return 0.0
    if readiness in {"needs_review", "review"}:
        return 0.02
    return 0.25


def _polished_title(card: dict) -> str:
    display = card.get("display") or {}
    display_title = display.get("display_title")
    if display_title:
        return display_title
    exp = str(card.get("exposure_label", "")).replace("_", " ").title()
    out = str(card.get("outcome_label", "")).replace("_", " ").title()
    unit = str(card.get("unit_of_analysis", "units")).replace("_", " ").title()
    return f"How Does {exp} Relate to {out} at the {unit} Level?"


def _rerank_reason(card: dict, domain_fit: float, identification: float, novelty: float) -> str:
    tags = ", ".join(card.get("technology_tags") or []) or "standard public-data"
    method = str(card.get("method") or "unspecified method").replace("_", " ")
    readiness = card.get("readiness") or "review"
    return (
        f"Domain fit={domain_fit:.2f} via {tags}; identification={identification:.2f} "
        f"using {method}; novelty potential={novelty:.2f}; readiness={readiness}."
    )


def rerank_candidates(
    cards: list[dict[str, Any]],
    domain_input: str,
    field_scan_summary: dict | None = None,
) -> list[dict[str, Any]]:
    domain_tokens = _tokens(domain_input) | (_GEOAI_TERMS & _tokens(domain_input))
    scan_tokens = _tokens(_field_scan_text(field_scan_summary))
    ranked: list[dict[str, Any]] = []

    for original in cards:
        card = deepcopy(original)
        scores = dict(card.get("scores") or {})
        domain_fit = _domain_fit(card, domain_tokens, scan_tokens)
        identification = _identification_strength(card)
        novelty = _novelty_potential(card, domain_tokens)
        title_quality = _title_quality(card)
        original_overall = float(scores.get("overall", 0.0) or 0.0)
        final_score = (
            0.30 * original_overall
            + 0.30 * domain_fit
            + 0.20 * identification
            + 0.15 * novelty
            + 0.05 * title_quality
            - _readiness_penalty(card)
        )
        final_score = _clamp(final_score)
        signals = {
            "domain_fit_score": domain_fit,
            "identification_strength_score": identification,
            "novelty_potential_score": novelty,
            "title_quality_score": title_quality,
            "original_overall_score": round(original_overall, 3),
            "rerank_score": final_score,
            "rerank_reason": _rerank_reason(card, domain_fit, identification, novelty),
        }
        scores["overall"] = final_score
        scores["rerank"] = signals
        card["scores"] = scores
        card["rerank"] = signals
        polished_title = _polished_title(card)
        card["polished_title"] = polished_title
        display = dict(card.get("display") or {})
        display.setdefault("display_title", polished_title)
        display["rerank_reason"] = signals["rerank_reason"]
        card["display"] = display
        ranked.append(card)

    status_priority = {"blocked": 1}
    ranked.sort(
        key=lambda c: (
            status_priority.get(c.get("readiness") or c.get("shortlist_status"), 0),
            -float((c.get("rerank") or {}).get("rerank_score", 0.0)),
            c.get("candidate_id", ""),
        )
    )
    for idx, card in enumerate(ranked, start=1):
        card["rank"] = idx
    return ranked
