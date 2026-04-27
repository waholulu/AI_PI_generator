from __future__ import annotations

# ── Novelty proxy table ────────────────────────────────────────────────────────
# Deterministic first-version novelty scores based on combination scarcity and
# methodological freshness.  Scores reflect how novel the (exposure, outcome,
# method) combination is relative to the published built-environment literature.
# Values are intentionally conservative to avoid over-claiming novelty.
#
# Scoring heuristic:
#   0.40-0.50  Saturated combination — many published studies, well-known pathway
#   0.55-0.65  Moderate overlap — active research area, some recent work
#   0.68-0.75  Emerging combination — limited published studies at this spatial grain
#   0.78-0.85  Novel combination — new data source or underexplored pathway
#
# Key rule: experimental/street-view combinations can score higher on novelty but
# are capped by automation_feasibility + 0.15 in the overall formula.

_NOVELTY_BY_EXPOSURE: dict[str, float] = {
    # Remote sensing / earth observation — newer in public-health literature
    "impervious_surface":       0.70,
    "land_cover":               0.68,
    "ndvi":                     0.68,
    "tree_canopy":              0.70,
    "nighttime_lights":         0.72,
    "building_footprint":       0.75,
    "building_density":         0.73,
    # Transit / active transport — growing but established body of work
    "transit_access":           0.65,
    "transit_frequency":        0.65,
    "bus_stop_density":         0.63,
    # Street network (OSMnx) — widely studied but data source is relatively new
    "street_connectivity":      0.63,
    "intersection_density":     0.63,
    "centrality":               0.70,   # graph centrality metrics are newer
    "network_centrality":       0.70,
    # Walkability indices — well-established
    "walkability":              0.48,
    "nwi":                      0.48,
    "pedestrian_infrastructure":0.52,
    # Green space / parks — established
    "park_access":              0.55,
    "green_space":              0.58,
    "enviroatlas":              0.65,
    # Environmental burden
    "air_quality":              0.58,
    "noise_exposure":           0.62,
    # Experimental / vision (high novelty but automation cost penalises score)
    "street_view_greenery":     0.82,
    "street_view_safety":       0.80,
    "visual_environment":       0.80,
}

_NOVELTY_BY_OUTCOME: dict[str, float] = {
    # Common outcomes — well-studied
    "obesity":                  0.45,
    "diabetes":                 0.48,
    "physical_inactivity":      0.50,
    "cardiovascular":           0.50,
    "hypertension":             0.50,
    # Intermediate outcomes — growing evidence
    "mental_health":            0.60,
    "depression":               0.62,
    "anxiety":                  0.62,
    "sleep":                    0.68,
    "well_being":               0.65,
    # Respiratory / environmental health
    "asthma":                   0.58,
    "respiratory":              0.58,
    "copd":                     0.60,
    # All-cause / mortality — established
    "mortality":                0.52,
}


def _compute_novelty(exposure_family: str, outcome_family: str, tags: set[str]) -> float:
    exp_key = exposure_family.lower().replace(" ", "_")
    out_key = outcome_family.lower().replace(" ", "_")

    exp_score = _NOVELTY_BY_EXPOSURE.get(exp_key)
    out_score = _NOVELTY_BY_OUTCOME.get(out_key)

    # If neither family is in the table, fall back to a moderate default
    if exp_score is None and out_score is None:
        base = 0.60
    elif exp_score is None:
        base = out_score
    elif out_score is None:
        base = exp_score
    else:
        # Geometric mean — high novelty on both dimensions amplifies; low on either pulls down
        base = (exp_score * out_score) ** 0.5

    # Small bonus for combining multiple novel technology tags
    novel_tags = tags & {"osmnx", "remote_sensing", "building_footprint", "gtfs", "viirs"}
    if len(novel_tags) >= 2:
        base = min(base + 0.05, 0.85)

    return round(base, 3)


def _status_to_score(status: str) -> float:
    if status == "pass":
        return 1.0
    if status == "warning":
        return 0.7
    return 0.2


def score_candidate(candidate: dict, gate_status: dict, repair_history: list[dict]) -> dict[str, float]:
    data_feasibility = _status_to_score(gate_status.get("overall", "fail"))

    risk = candidate.get("automation_risk", "medium")
    required_secrets = candidate.get("required_secrets", [])
    cloud_safe = bool(candidate.get("cloud_safe", True))
    if risk == "low" and not required_secrets and cloud_safe:
        automation_feasibility = 1.0
    elif risk == "high" or any("Google_Street_View" in s for s in required_secrets):
        automation_feasibility = 0.2
    else:
        automation_feasibility = 0.7

    # Guardrail: required_secrets cap automation feasibility (Step 3 policy)
    if required_secrets:
        automation_feasibility = min(automation_feasibility, 0.45)

    threats = candidate.get("key_threats", [])
    mitigations = candidate.get("mitigations", {})
    if threats:
        covered = len(set(threats) & set(mitigations.keys()))
        ratio = covered / len(threats)
    else:
        ratio = 0.0

    if len(threats) >= 4 and ratio >= 1.0:
        identification_quality = 0.9
    elif len(threats) >= 3 and ratio >= 0.8:
        identification_quality = 0.8
    else:
        identification_quality = 0.4

    tags = set(candidate.get("technology_tags", []))
    novelty = _compute_novelty(
        candidate.get("exposure_family", ""),
        candidate.get("outcome_family", ""),
        tags,
    )

    if "experimental" in tags:
        technology_innovation = 0.85
    elif tags & {"osmnx", "remote_sensing"}:
        technology_innovation = 0.75
    else:
        technology_innovation = 0.55

    technology_innovation = min(technology_innovation, automation_feasibility + 0.15)

    overall = (
        0.30 * data_feasibility
        + 0.25 * automation_feasibility
        + 0.20 * identification_quality
        + 0.15 * novelty
        + 0.10 * technology_innovation
    )

    # Guardrail: high-risk candidates are capped regardless of tech innovation (Step 3 policy)
    if risk == "high":
        overall = min(overall, 0.65)

    if any(h.get("result") == "blocked" for h in repair_history):
        overall = min(overall, 0.45)

    return {
        "data_feasibility": round(data_feasibility, 3),
        "automation_feasibility": round(automation_feasibility, 3),
        "identification_quality": round(identification_quality, 3),
        "novelty": round(novelty, 3),
        "technology_innovation": round(technology_innovation, 3),
        "overall": round(overall, 3),
    }


def rank_candidates(cards: list[dict]) -> list[dict]:
    ranked = sorted(
        cards,
        key=lambda c: (
            -float(c.get("scores", {}).get("overall", 0.0)),
            c.get("automation_risk", "medium") != "low",
            c.get("shortlist_status") != "ready",
        ),
    )
    for idx, card in enumerate(ranked, start=1):
        card["rank"] = idx
    return ranked
