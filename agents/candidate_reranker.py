"""Lightweight candidate reranking and display polish.

This module deliberately avoids additional network calls.  It takes the
deterministic candidate-factory pool and adds a small research-value layer:
domain fit, empirical deepening value, identification strength, novelty
potential, and title quality.
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
    "geospatial", "vision", "streetview_cv",
}

_STRONG_METHOD_TERMS = {
    "difference_in_differences", "event_study", "panel_fixed_effects",
    "spatial_discontinuity", "instrumental_variable", "synthetic_control",
}

_TECH_TITLE_PREFIX = {
    "remote_sensing": "Satellite-Measured",
    "viirs": "Satellite-Measured",
    "building_footprint": "Building-Footprint-Derived",
    "osmnx": "Network-Derived",
    "gtfs": "Schedule-Derived",
    "mobility": "Mobility-Derived",
    "geospatial": "Spatially Resolved",
    "vision": "Computer-Vision-Measured",
}

_EXPOSURE_MEASUREMENT_GAIN = {
    "nighttime_lights": 0.92,
    "building_density": 0.90,
    "building_footprint": 0.90,
    "impervious_surface": 0.88,
    "tree_canopy": 0.86,
    "transit_access": 0.86,
    "destination_accessibility": 0.85,
    "street_network": 0.84,
    "walkability": 0.72,
    "park_access": 0.72,
    "green_space": 0.66,
    "streetview_built_form": 0.94,
    "greenery_visibility": 0.90,
    "sidewalk_presence": 0.88,
}

_MECHANISM_PAIRS = {
    ("street_network", "physical_inactivity"): 0.78,
    ("walkability", "physical_inactivity"): 0.78,
    ("transit_access", "physical_inactivity"): 0.76,
    ("destination_accessibility", "physical_inactivity"): 0.74,
    ("park_access", "physical_inactivity"): 0.72,
    ("nighttime_lights", "cardiovascular_disease"): 0.74,
    ("green_space", "poor_mental_health"): 0.68,
    ("tree_canopy", "poor_mental_health"): 0.68,
    ("impervious_surface", "asthma"): 0.62,
    ("tree_canopy", "asthma"): 0.60,
    ("green_space", "asthma"): 0.54,
    ("street_network", "asthma"): 0.48,
    ("streetview_built_form", "physical_inactivity"): 0.76,
    ("streetview_built_form", "obesity"): 0.70,
    ("streetview_built_form", "poor_mental_health"): 0.74,
}

_COMMON_EXPOSURES = {"green_space", "walkability"}

_DISPLAY_EXPOSURE_LABELS = {
    "street_network": "Street Connectivity",
    "nighttime_lights": "Nighttime Light Intensity",
    "impervious_surface": "Impervious Surface",
    "tree_canopy": "Tree Canopy",
    "transit_access": "Transit Access",
    "destination_accessibility": "Destination Accessibility",
    "building_density": "Building Density",
    "green_space": "Green Space",
    "streetview_built_form": "Street-Level Built Form",
    "greenery_visibility": "Street-View Greenery",
    "sidewalk_presence": "Sidewalk Presence",
}


def _tokens(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {tok for tok in normalized.split() if tok}


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _label(value: str) -> str:
    return str(value or "").replace("_", " ").strip()


def _display_exposure(value: str) -> str:
    value = str(value or "")
    return _DISPLAY_EXPOSURE_LABELS.get(value, _label(value).title())


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
    if tags & {"vision", "streetview_cv"}:
        base += 0.16
    if tags & {"osmnx", "gtfs"}:
        base += 0.07
    if "geoai" in domain_tokens and tags & _GEOAI_TECH:
        base += 0.06
    return _clamp(base)


def _measurement_gain(card: dict) -> float:
    exp = str(card.get("exposure_label") or "").lower()
    tags = set(card.get("technology_tags") or [])
    base = _EXPOSURE_MEASUREMENT_GAIN.get(exp, 0.48)
    if tags & {"remote_sensing", "viirs", "building_footprint"}:
        base += 0.04
    if tags & {"vision", "streetview_cv"}:
        base += 0.08
    if tags & {"osmnx", "gtfs", "mobility"}:
        base += 0.03
    if not tags:
        base -= 0.08
    return _clamp(base)


def _mechanism_gain(card: dict) -> float:
    exp = str(card.get("exposure_label") or "").lower()
    out = str(card.get("outcome_label") or "").lower()
    if (exp, out) in _MECHANISM_PAIRS:
        return _MECHANISM_PAIRS[(exp, out)]
    if out in {"physical_inactivity", "obesity"} and exp in {
        "street_network", "walkability", "transit_access",
        "destination_accessibility", "park_access",
    }:
        return 0.70
    if out in {"asthma", "cardiovascular_disease"} and exp in {
        "impervious_surface", "tree_canopy", "nighttime_lights", "building_density",
    }:
        return 0.62
    if out in {"poor_mental_health", "poor_physical_health"} and exp in {
        "green_space", "tree_canopy", "park_access", "nighttime_lights",
    }:
        return 0.62
    return 0.45


def _heterogeneity_gain(card: dict) -> float:
    unit = str(card.get("unit_of_analysis") or "").lower()
    if "block" in unit:
        return 0.76
    if "tract" in unit or "parcel" in unit or "street" in unit:
        return 0.68
    if "city" in unit or "county" in unit:
        return 0.50
    return 0.55


def _tech_lens_type(card: dict, measurement: float, mechanism: float, heterogeneity: float) -> str:
    if measurement >= mechanism and measurement >= heterogeneity:
        return "better_measurement_of_x"
    if mechanism >= heterogeneity:
        return "mechanism_probe"
    return "heterogeneity_probe"


def _tech_phrase(card: dict) -> str:
    source = str(card.get("exposure_source") or "").lower()
    if "viirs" in source or "nlcd" in source or "enviroatlas" in source:
        return "Satellite-Measured"
    if "osmnx" in source or "openstreetmap" in source:
        return "Network-Derived"
    if "gtfs" in source:
        return "Schedule-Derived"
    if "mapillary" in source or "street_view" in source or "streetview" in source:
        return "Computer-Vision-Measured"
    tags = list(card.get("technology_tags") or [])
    for tag in tags:
        if tag in _TECH_TITLE_PREFIX:
            return _TECH_TITLE_PREFIX[tag]
    return "Newly Measured"


def _empirical_deepening(card: dict, identification: float) -> dict[str, float | str]:
    measurement = _measurement_gain(card)
    mechanism = _mechanism_gain(card)
    heterogeneity = _heterogeneity_gain(card)
    exp = str(card.get("exposure_label") or "").lower()
    tags = set(card.get("technology_tags") or [])
    decoration_penalty = 0.0
    if exp in _COMMON_EXPOSURES:
        decoration_penalty += 0.08
    if not tags and measurement < 0.60:
        decoration_penalty += 0.06

    empirical_value = (
        0.45 * measurement
        + 0.25 * mechanism
        + 0.20 * heterogeneity
        + 0.10 * identification
        - decoration_penalty
    )
    lens_type = _tech_lens_type(card, measurement, mechanism, heterogeneity)
    exp_label = _label(card.get("exposure_label"))
    display_exp = _display_exposure(str(card.get("exposure_label") or "")).lower()
    out_label = _label(card.get("outcome_label"))
    source = card.get("exposure_source") or "public data"
    unit = _label(card.get("unit_of_analysis")) or "study unit"
    if lens_type == "better_measurement_of_x":
        claim = (
            f"Uses {source} to measure {exp_label} more precisely, making the "
            f"traditional {display_exp} -> {out_label} relationship testable at "
            f"the {unit} level."
        )
    elif lens_type == "mechanism_probe":
        claim = (
            f"Frames {exp_label} as a mechanism-rich exposure, helping explain "
            f"why the {display_exp} -> {out_label} association may appear."
        )
    else:
        claim = (
            f"Uses fine-grained spatial measurement to compare where the "
            f"{display_exp} -> {out_label} relationship is strongest."
        )
    return {
        "tech_lens_type": lens_type,
        "empirical_value_score": _clamp(empirical_value),
        "measurement_gain_score": measurement,
        "mechanism_gain_score": _clamp(mechanism),
        "heterogeneity_gain_score": _clamp(heterogeneity),
        "tech_decoration_penalty": _clamp(decoration_penalty),
        "empirical_deepening_claim": claim,
    }


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


def _polished_title(card: dict, lens: dict[str, float | str] | None = None) -> str:
    exp = _display_exposure(str(card.get("exposure_label") or ""))
    out = _label(card.get("outcome_label")).title()
    if lens and lens.get("empirical_value_score", 0) >= 0.62:
        return f"What Does {_tech_phrase(card)} {exp} Reveal About {out}?"
    display = card.get("display") or {}
    display_title = display.get("display_title")
    if display_title:
        return display_title
    unit = _label(card.get("unit_of_analysis") or "units").title()
    return f"How Does {exp} Relate to {out} at the {unit} Level?"


def _rerank_reason(
    card: dict,
    domain_fit: float,
    identification: float,
    novelty: float,
    empirical_value: float,
) -> str:
    tags = ", ".join(card.get("technology_tags") or []) or "standard public-data"
    method = str(card.get("method") or "unspecified method").replace("_", " ")
    readiness = card.get("readiness") or "review"
    return (
        f"Empirical deepening={empirical_value:.2f}; domain fit={domain_fit:.2f} "
        f"via {tags}; identification={identification:.2f} using {method}; "
        f"novelty potential={novelty:.2f}; readiness={readiness}."
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
        empirical_lens = _empirical_deepening(card, identification)
        empirical_value = float(empirical_lens["empirical_value_score"])
        original_overall = float(scores.get("overall", 0.0) or 0.0)
        final_score = (
            0.25 * original_overall
            + 0.20 * domain_fit
            + 0.15 * identification
            + 0.10 * novelty
            + 0.25 * empirical_value
            + 0.05 * title_quality
            - _readiness_penalty(card)
        )
        final_score = _clamp(final_score)
        signals = {
            "domain_fit_score": domain_fit,
            "identification_strength_score": identification,
            "novelty_potential_score": novelty,
            "empirical_value_score": empirical_value,
            "title_quality_score": title_quality,
            "original_overall_score": round(original_overall, 3),
            "rerank_score": final_score,
            "rerank_reason": _rerank_reason(
                card, domain_fit, identification, novelty, empirical_value
            ),
        }
        signals.update(empirical_lens)
        scores["overall"] = final_score
        scores["rerank"] = signals
        card["scores"] = scores
        card["rerank"] = signals
        card["tech_lens_type"] = empirical_lens["tech_lens_type"]
        card["empirical_deepening_claim"] = empirical_lens["empirical_deepening_claim"]
        card["empirical_value_score"] = empirical_value
        polished_title = _polished_title(card, empirical_lens)
        card["polished_title"] = polished_title
        display = dict(card.get("display") or {})
        display["display_title"] = polished_title
        display["empirical_deepening_claim"] = empirical_lens["empirical_deepening_claim"]
        display["tech_lens_type"] = empirical_lens["tech_lens_type"]
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
