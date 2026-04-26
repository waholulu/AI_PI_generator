from __future__ import annotations


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

    novelty = 0.60

    tags = set(candidate.get("technology_tags", []))
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
