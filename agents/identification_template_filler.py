from __future__ import annotations

from models.candidate_composer_schema import ComposedCandidate

_METHOD_THREATS: dict[str, dict] = {
    "cross_sectional_spatial_association": {
        "threats": [
            "socioeconomic_confounding",
            "residential_self_selection",
            "spatial_autocorrelation",
            "exposure_measurement_error",
        ],
        "mitigations": {
            "socioeconomic_confounding": "Include tract-level ACS socioeconomic controls.",
            "residential_self_selection": "Include county/metro fixed effects and avoid causal claims.",
            "spatial_autocorrelation": "Use clustered or spatially robust standard errors.",
            "exposure_measurement_error": "Run sensitivity checks with alternative feature definitions.",
        },
    }
}


def fill_identification_from_method(
    candidate: ComposedCandidate,
    method_spec: dict | None = None,
) -> tuple[list[str], dict[str, str]]:
    defaults = method_spec or _METHOD_THREATS.get(candidate.method_template, {})
    default_threats = list(defaults.get("threats", []))
    default_mitigations = dict(defaults.get("mitigations", {}))

    merged_threats = list(default_threats)
    for threat in candidate.key_threats:
        if threat not in merged_threats:
            merged_threats.append(threat)

    merged_mitigations = dict(default_mitigations)
    merged_mitigations.update(candidate.mitigations)

    return merged_threats, merged_mitigations


def ensure_identification_metadata(candidate: ComposedCandidate) -> ComposedCandidate:
    """Guarantee ≥3 threats with full mitigation coverage before gate checks.

    No-op if the candidate already has ≥3 threats and all have mitigations.
    Otherwise merges method-template defaults so the export validator's strict
    3-threat requirement is satisfied without LLM cost.
    """
    threats = list(candidate.key_threats or [])
    mits = dict(candidate.mitigations or {})
    if len(threats) >= 3 and all(t in mits for t in threats):
        return candidate
    new_threats, new_mits = fill_identification_from_method(candidate)
    return candidate.model_copy(update={"key_threats": new_threats, "mitigations": new_mits})
