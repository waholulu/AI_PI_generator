"""
Candidate normalizer: converts raw LLM seed dicts into canonical research designs.

Sits between LLM seed generation and evaluate_candidate() to:
  1. Map free-text exposure/outcome families to canonical enum values
  2. Resolve source names (including synonyms) to canonical source_ids via SourceRegistry
  3. Infer and fill exposure_source / outcome_source roles
  4. Auto-fill key_threats and mitigations from the method template
  5. Downgrade claim_strength to "associational" when the method is OLS / descriptive
  6. Ensure ACS (control) and TIGER_Lines (boundary) are always in declared_sources

The output dict is drop-in compatible with _dict_to_seed_candidate() and
build_research_plan_from_candidate() — no other code changes needed.
"""

from __future__ import annotations

import difflib
from typing import Any

from agents.logging_config import get_logger
from agents.source_registry import SourceRegistry

logger = get_logger(__name__)

# ── Synonym tables ─────────────────────────────────────────────────────────────

# Maps free-text exposure concepts → (canonical_family, preferred_source_id)
_EXPOSURE_SYNONYMS: dict[str, tuple[str, str]] = {
    # street / morphology
    "street network": ("street_network", "OSMnx_OpenStreetMap"),
    "street_network": ("street_network", "OSMnx_OpenStreetMap"),
    "urban morphology": ("street_network", "OSMnx_OpenStreetMap"),
    "street morphology": ("street_network", "OSMnx_OpenStreetMap"),
    "street design": ("street_network", "EPA_Smart_Location_Database"),
    "street connectivity": ("street_connectivity", "OSMnx_OpenStreetMap"),
    "intersection density": ("intersection_density", "OSMnx_OpenStreetMap"),
    "road network": ("street_network", "OSMnx_OpenStreetMap"),
    # walkability
    "walkability": ("walkability", "EPA_National_Walkability_Index"),
    "walk score": ("walkability", "EPA_National_Walkability_Index"),
    "pedestrian infrastructure": ("walkability", "EPA_National_Walkability_Index"),
    # built environment (broad)
    "built environment": ("density", "EPA_Smart_Location_Database"),
    "urban form": ("density", "EPA_Smart_Location_Database"),
    "land use mix": ("land_use_mix", "EPA_Smart_Location_Database"),
    "land_use_mix": ("land_use_mix", "EPA_Smart_Location_Database"),
    "density": ("density", "EPA_Smart_Location_Database"),
    "transit access": ("transit_access", "GTFS"),
    "transit_access": ("transit_access", "GTFS"),
    "destination accessibility": ("destination_accessibility", "EPA_National_Walkability_Index"),
    # green space
    "green space": ("green_space", "NLCD"),
    "greenness": ("green_space", "EPA_EnviroAtlas"),
    "tree canopy": ("tree_canopy", "EPA_EnviroAtlas"),
    "vegetation": ("green_space", "NLCD"),
    "park access": ("park_access", "EPA_EnviroAtlas"),
    "park_access": ("park_access", "EPA_EnviroAtlas"),
    "ndvi": ("green_space", "NLCD"),
    "land cover": ("green_space", "NLCD"),
    "impervious surface": ("impervious_surface", "NLCD"),
    # nighttime lights / vacancy
    "nighttime lights": ("nighttime_lights", "VIIRS"),
    "nighttime_lights": ("nighttime_lights", "VIIRS"),
    "ghost buildings": ("nighttime_lights", "VIIRS"),
    "vacancy": ("nighttime_lights", "VIIRS"),
    "vacant buildings": ("nighttime_lights", "VIIRS"),
    "building vacancy": ("nighttime_lights", "VIIRS"),
    # building footprints
    "building density": ("building_density", "Microsoft_Building_Footprints"),
    "building_density": ("building_density", "Microsoft_Building_Footprints"),
    "building footprints": ("building_density", "Microsoft_Building_Footprints"),
    "footprint coverage": ("building_density", "Microsoft_Building_Footprints"),
    # streetview
    "street view": ("streetview_built_form", "Mapillary_Street_Images"),
    "streetview": ("streetview_built_form", "Mapillary_Street_Images"),
    "greenery visibility": ("greenery_visibility", "Mapillary_Street_Images"),
    "sidewalk presence": ("sidewalk_presence", "Mapillary_Street_Images"),
}

# Maps free-text outcome concepts → (canonical_family, preferred_source_id)
_OUTCOME_SYNONYMS: dict[str, tuple[str, str]] = {
    # CDC PLACES outcomes
    "poor mental health": ("poor_mental_health", "CDC_PLACES"),
    "poor_mental_health": ("poor_mental_health", "CDC_PLACES"),
    "mental health": ("poor_mental_health", "CDC_PLACES"),
    "depression": ("poor_mental_health", "CDC_PLACES"),
    "anxiety": ("poor_mental_health", "CDC_PLACES"),
    "psychological distress": ("poor_mental_health", "CDC_PLACES"),
    "mental wellbeing": ("poor_mental_health", "CDC_PLACES"),
    "mental illness": ("poor_mental_health", "CDC_PLACES"),
    "obesity": ("obesity", "CDC_PLACES"),
    "overweight": ("obesity", "CDC_PLACES"),
    "bmi": ("obesity", "CDC_PLACES"),
    "diabetes": ("diabetes", "CDC_PLACES"),
    "type 2 diabetes": ("diabetes", "CDC_PLACES"),
    "physical inactivity": ("physical_inactivity", "CDC_PLACES"),
    "physical_inactivity": ("physical_inactivity", "CDC_PLACES"),
    "physical activity": ("physical_inactivity", "CDC_PLACES"),
    "sedentary behavior": ("physical_inactivity", "CDC_PLACES"),
    "asthma": ("asthma", "CDC_PLACES"),
    "respiratory": ("asthma", "CDC_PLACES"),
    "cardiovascular disease": ("cardiovascular_disease", "CDC_PLACES"),
    "cardiovascular_disease": ("cardiovascular_disease", "CDC_PLACES"),
    "heart disease": ("cardiovascular_disease", "CDC_PLACES"),
    "poor physical health": ("poor_physical_health", "CDC_PLACES"),
    "poor_physical_health": ("poor_physical_health", "CDC_PLACES"),
    "general health": ("poor_physical_health", "CDC_PLACES"),
    "self-rated health": ("poor_physical_health", "CDC_PLACES"),
    "hypertension": ("cardiovascular_disease", "CDC_PLACES"),
    "blood pressure": ("cardiovascular_disease", "CDC_PLACES"),
    "chronic kidney disease": ("poor_physical_health", "CDC_PLACES"),
    "ckd": ("poor_physical_health", "CDC_PLACES"),
    "kidney disease": ("poor_physical_health", "CDC_PLACES"),
}

# Methods that support quasi-causal claims; all others → associational
_QUASI_CAUSAL_METHODS = {
    "diff_in_diff",
    "diff_in_diff_event_study",
    "did",
    "regression_discontinuity",
    "rd",
    "instrumental_variable",
    "iv",
    "synthetic_control",
    "event_study",
    "interrupted_time_series",
    "its",
    "target_trial_emulation",
    "target_trial_emulation_iptw",
    "iptw",
    "target_trial_emulation_overlap_weighting",
    "overlap_weighting",
    "ow",
    "target_trial_emulation_tmle",
    "tmle",
    "target_trial_emulation_aipw",
    "aipw",
    "target_trial_emulation_matching",
    "propensity_score_matching",
    "propensity_score_weighting",
    "causal_forest",
    "causal_forest_heterogeneous_treatment_effects",
}

# Default threats/mitigations per method template
_METHOD_THREATS: dict[str, dict[str, str]] = {
    "diff_in_diff": {
        "parallel_trends": "test pre-treatment trend equality across treatment and control units",
        "confounding": "unit and time fixed effects absorb time-invariant and common-trend confounders",
        "spatial_autocorrelation": "cluster standard errors at the county level",
    },
    "diff_in_diff_event_study": {
        "parallel_trends_violation": "estimate pre-treatment event-study leads",
        "treatment_timing_heterogeneity": "use staggered-adoption robust estimators where applicable",
        "spillover": "exclude neighboring areas or model exposure buffers",
    },
    "regression_discontinuity": {
        "manipulation": "McCrary density test at the cutoff",
        "confounding": "local polynomial with optimal bandwidth; placebo cutoffs",
        "spillover": "donut RD excluding units within 0.5 SD of cutoff",
    },
    "interrupted_time_series": {
        "pre_existing_trends": "model and visualize pre-intervention trends",
        "seasonality": "include month or seasonal fixed effects where applicable",
        "autocorrelation": "use HAC or autoregressive error corrections",
    },
    "instrumental_variable": {
        "instrument_relevance": "first-stage F-statistic > 10",
        "exclusion_restriction": "argue exclusion via domain knowledge + overidentification test if available",
        "confounding": "control for ACS demographic and socioeconomic covariates",
    },
    "synthetic_control": {
        "donor_pool_contamination": "exclude units with known policy exposure",
        "inference": "permutation-based p-values across donor pool",
        "confounding": "match on pre-treatment outcome trajectories",
    },
    "target_trial_emulation_iptw": {
        "baseline_confounding": "estimate propensity scores using baseline-only covariates",
        "immortal_time_bias": "align eligibility, exposure assignment, and follow-up at time zero",
        "positivity_violation": "inspect propensity overlap and truncate extreme IPTW weights",
    },
    "target_trial_emulation_overlap_weighting": {
        "baseline_confounding": "estimate overlap weights from pre-exposure covariates",
        "limited_common_support": "report the overlap-population estimand and effective sample size",
        "immortal_time_bias": "define time zero before exposure and outcome follow-up",
    },
    "target_trial_emulation_tmle": {
        "nuisance_model_misspecification": "use cross-fitting for outcome and propensity nuisance models",
        "baseline_confounding": "adjust only for pre-exposure baseline covariates",
        "positivity_violation": "check propensity overlap and truncation sensitivity",
    },
    "target_trial_emulation_aipw": {
        "nuisance_model_misspecification": "estimate both treatment and outcome models and compare specifications",
        "baseline_confounding": "use baseline-only socioeconomic and demographic controls",
        "positivity_violation": "inspect support and trim unstable weights",
    },
    "target_trial_emulation_matching": {
        "baseline_confounding": "match on pre-exposure ACS and geography covariates",
        "poor_match_quality": "report standardized mean differences before and after matching",
        "limited_common_support": "drop unmatched units and describe the matched target population",
    },
    "causal_forest_heterogeneous_treatment_effects": {
        "baseline_confounding": "include rich baseline covariates and compare with weighted average effects",
        "positivity_violation": "restrict heterogeneity claims to regions with common support",
        "overfitting_heterogeneity": "use honest forests or cross-fitting and validate subgroup effects",
    },
    "default": {
        "confounding": "adjust for ACS race, income, education, and housing age controls",
        "spatial_autocorrelation": "use spatially-clustered standard errors or spatial-lag term",
        "ecological_bias": "state explicitly that inference is area-level, not individual-level",
    },
}

_METHOD_ALIASES = {
    "did": "diff_in_diff",
    "event_study": "diff_in_diff_event_study",
    "rd": "regression_discontinuity",
    "iv": "instrumental_variable",
    "its": "interrupted_time_series",
    "target_trial_emulation": "target_trial_emulation_iptw",
    "iptw": "target_trial_emulation_iptw",
    "overlap_weighting": "target_trial_emulation_overlap_weighting",
    "ow": "target_trial_emulation_overlap_weighting",
    "tmle": "target_trial_emulation_tmle",
    "aipw": "target_trial_emulation_aipw",
    "propensity_score_matching": "target_trial_emulation_matching",
    "propensity_score_weighting": "target_trial_emulation_iptw",
    "causal_forest": "causal_forest_heterogeneous_treatment_effects",
}

# Sources that are always added regardless of topic
_MANDATORY_CONTROL_SOURCES = ["ACS"]
_MANDATORY_BOUNDARY_SOURCES = ["TIGER_Lines"]


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw LLM seed dict into a canonical research design dict.

    Modifies a copy of *candidate* and returns it.  All original fields are
    preserved; normalized fields overwrite their raw equivalents.
    """
    out = dict(candidate)
    registry = SourceRegistry.load()

    out = _normalize_exposure(out, registry)
    out = _normalize_outcome(out, registry)
    out = _ensure_declared_sources(out)
    out = _fill_threats_and_mitigations(out)
    out = _normalize_claim_strength(out)

    return out


# ── Private helpers ────────────────────────────────────────────────────────────

def _resolve_source(name: str, registry: SourceRegistry) -> str | None:
    """Resolve a source name or alias to a canonical source_id.

    Priority: 1) exact registry resolve  2) fuzzy match on alias_to_id
    Returns None when no match is found.
    """
    if not name:
        return None
    canonical = registry.resolve(name)
    if canonical:
        return canonical

    # Fuzzy match against alias map
    key = name.strip().lower()
    best_ratio = 0.0
    best_id: str | None = None
    for alias, sid in registry.alias_to_id.items():
        ratio = difflib.SequenceMatcher(None, key, alias).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = sid

    if best_ratio >= 0.72:
        logger.debug("Fuzzy-resolved source '%s' → '%s' (score=%.2f)", name, best_id, best_ratio)
        return best_id
    return None


def _synonym_lookup(text: str, table: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """Return (family, source_id) for the best matching synonym, or None."""
    key = (text or "").strip().lower()
    if key in table:
        return table[key]

    # Substring scan: check if any synonym keyword is contained in the text
    best: tuple[str, str] | None = None
    best_len = 0
    for phrase, value in table.items():
        if phrase in key and len(phrase) > best_len:
            best = value
            best_len = len(phrase)
    return best


def _normalize_exposure(candidate: dict, registry: SourceRegistry) -> dict:
    """Set exposure_source from registry; update exposure_family if needed."""
    raw_source = str(candidate.get("exposure_source") or "").strip()
    raw_family = str(
        candidate.get("exposure_family") or candidate.get("exposure_specific") or ""
    ).strip()

    # 1. Try to resolve the declared source directly
    resolved = _resolve_source(raw_source, registry)
    if resolved:
        candidate["exposure_source"] = resolved
        return candidate

    # 2. Fall back to synonym lookup on the family/specific text
    hit = _synonym_lookup(raw_family.lower(), _EXPOSURE_SYNONYMS)
    if hit is None:
        # Try exposure_specific as well
        raw_specific = str(candidate.get("exposure_specific") or "").strip()
        hit = _synonym_lookup(raw_specific.lower(), _EXPOSURE_SYNONYMS)

    if hit:
        family_val, source_id = hit
        candidate["exposure_source"] = source_id
        # Only override exposure_family when the LLM gave something too vague
        if not candidate.get("exposure_family") or candidate["exposure_family"] in (
            "other", "built_environment", ""
        ):
            candidate["exposure_family"] = family_val
        logger.debug(
            "Normalized exposure '%s' → family='%s' source='%s'",
            raw_family, family_val, source_id,
        )
        return candidate

    # 3. Infer from the registry: if the family maps to a known source
    if raw_family:
        sources_for_family = registry.get_sources_by_variable_family(raw_family)
        if sources_for_family:
            candidate["exposure_source"] = sources_for_family[0]
            logger.debug("Inferred exposure source '%s' from family '%s'", sources_for_family[0], raw_family)
            return candidate

    logger.warning("Could not resolve exposure source for candidate '%s'", candidate.get("title"))
    return candidate


def _normalize_outcome(candidate: dict, registry: SourceRegistry) -> dict:
    """Set outcome_source from registry; update outcome_family if needed."""
    raw_source = str(candidate.get("outcome_source") or "").strip()
    raw_family = str(
        candidate.get("outcome_family") or candidate.get("outcome_specific") or ""
    ).strip()

    resolved = _resolve_source(raw_source, registry)
    if resolved:
        candidate["outcome_source"] = resolved
        return candidate

    hit = _synonym_lookup(raw_family.lower(), _OUTCOME_SYNONYMS)
    if hit is None:
        raw_specific = str(candidate.get("outcome_specific") or "").strip()
        hit = _synonym_lookup(raw_specific.lower(), _OUTCOME_SYNONYMS)

    if hit:
        family_val, source_id = hit
        candidate["outcome_source"] = source_id
        if not candidate.get("outcome_family") or candidate["outcome_family"] in ("other", "health", ""):
            candidate["outcome_family"] = family_val
        logger.debug(
            "Normalized outcome '%s' → family='%s' source='%s'",
            raw_family, family_val, source_id,
        )
        return candidate

    if raw_family:
        sources_for_family = registry.get_sources_by_variable_family(raw_family)
        if sources_for_family:
            candidate["outcome_source"] = sources_for_family[0]
            return candidate

    logger.warning("Could not resolve outcome source for candidate '%s'", candidate.get("title"))
    return candidate


def _ensure_declared_sources(candidate: dict) -> dict:
    """Make sure declared_sources contains exposure_source, outcome_source, ACS, TIGER_Lines."""
    declared = list(candidate.get("declared_sources") or [])
    seen = {str(s).strip() for s in declared}

    for required in [
        candidate.get("exposure_source"),
        candidate.get("outcome_source"),
        *_MANDATORY_CONTROL_SOURCES,
        *_MANDATORY_BOUNDARY_SOURCES,
    ]:
        if required and required not in seen:
            declared.append(required)
            seen.add(required)

    candidate["declared_sources"] = declared
    return candidate


def _fill_threats_and_mitigations(candidate: dict) -> dict:
    """Fill key_threats and mitigations from the method template when absent."""
    method = str(candidate.get("method") or candidate.get("method_template") or "").strip().lower()
    method = _METHOD_ALIASES.get(method, method)
    existing_threats: list[str] = list(candidate.get("key_threats") or [])
    existing_mitigations: dict[str, str] = dict(candidate.get("mitigations") or {})

    template = _METHOD_THREATS.get(method, _METHOD_THREATS["default"])

    # Always ensure the default threats are present
    default = _METHOD_THREATS["default"]
    merged_template = {**default, **template}

    for threat, mitigation in merged_template.items():
        if threat not in existing_threats:
            existing_threats.append(threat)
        if threat not in existing_mitigations:
            existing_mitigations[threat] = mitigation

    candidate["key_threats"] = existing_threats
    candidate["mitigations"] = existing_mitigations
    return candidate


def _normalize_claim_strength(candidate: dict) -> dict:
    """Set claim_strength based on the identification method."""
    method = str(candidate.get("method") or candidate.get("method_template") or "").strip().lower()
    method = _METHOD_ALIASES.get(method, method)
    current = str(candidate.get("claim_strength") or "").strip().lower()

    if method in _QUASI_CAUSAL_METHODS:
        # Only upgrade to quasi_causal; never override an explicit "causal" claim
        if current not in ("quasi_causal", "causal"):
            candidate["claim_strength"] = "quasi_causal"
    else:
        # Downgrade any over-claiming to associational
        if current in ("causal", ""):
            candidate["claim_strength"] = "associational"

    return candidate


# ── Executability classification ───────────────────────────────────────────────

# Reasons that make a candidate Blocked (cannot proceed without human repair)
# Public names are exported for use by hitl_helpers and other callers.
BLOCKER_REASONS = frozenset({
    "missing_exposure_role_source",
    "missing_outcome_role_source",
    "missing_machine_readable_source",
    "no_data_sources_declared",
    "missing_join_path",
})
_BLOCKER_REASONS = BLOCKER_REASONS  # backward-compat alias

# Reasons that indicate Needs review (can proceed but warrants human attention)
_REVIEW_REASONS = frozenset({
    "missing_identification_method",
    "missing_identification_threats",
    "partial_literature_overlap",
    "already_published_overlap",
    "short_rationale_too_brief",
    "research_question_too_brief",
    "geography_incompatible",
    "time_window_incompatible",
})

# Reasons that are purely informational (gray badge, never degrade status)
_INFO_REASONS = frozenset({
    "source_alias_resolved",
    "non_canonical_source_name",
    "canonicalize_source_name",
    "partial_registry_match",
    "novelty_evidence_limited",
    "experimental_source_requires_key",
    "url_not_reachable",
    "no_access_url",
})


def compute_executability_status(evaluation: dict[str, Any]) -> str:
    """Classify a candidate as 'ready', 'needs_review', or 'blocked'.

    Based on the evaluation dict produced by evaluate_candidate().
    """
    reasons: list[str] = list(evaluation.get("reasons") or [])
    overall = str(evaluation.get("overall_verdict") or "fail").lower()

    blocker_hits = [r for r in reasons if r in _BLOCKER_REASONS]
    if blocker_hits or overall == "fail":
        return "blocked"

    review_hits = [r for r in reasons if r in _REVIEW_REASONS]
    if review_hits or overall == "warning":
        return "needs_review"

    return "ready"
