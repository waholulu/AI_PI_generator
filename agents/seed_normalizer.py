from __future__ import annotations

import re
from enum import Enum
from typing import TypeVar

from agents.openalex_verifier import (
    _EXPOSURE_KEYWORDS,
    _METHOD_KEYWORDS,
    _OUTCOME_KEYWORDS,
)
from models.topic_schema import (
    ContributionPrimary,
    ExposureFamily,
    IdentificationPrimary,
    OutcomeFamily,
)

E = TypeVar("E", bound=Enum)


class SeedNormalizationError(ValueError):
    """Raised when a seed field cannot be normalized into a schema enum."""


def _norm_text(value: str) -> str:
    s = (value or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _norm_variants(value: str) -> set[str]:
    base = _norm_text(value)
    if not base:
        return set()
    variants = {base}
    if base.endswith("ies"):
        variants.add(base[:-3] + "y")
    if base.endswith("es"):
        variants.add(base[:-2])
    if base.endswith("s"):
        variants.add(base[:-1])
    return {v for v in variants if v}


def _try_match_enum_value(raw: str, enum_cls: type[E]) -> E | None:
    if not raw:
        return None
    for member in enum_cls:
        if raw == member.value:
            return member

    raw_variants = _norm_variants(raw)
    for member in enum_cls:
        if raw_variants & _norm_variants(member.value):
            return member
    return None


def _keyword_mapping_for_enum(enum_cls: type[E]) -> dict[str, list[str]]:
    if enum_cls is ExposureFamily:
        return {
            **_EXPOSURE_KEYWORDS,
            ExposureFamily.TRANSPORT_INFRA.value: _EXPOSURE_KEYWORDS.get(
                ExposureFamily.TRANSPORT_INFRA.value, []
            ) + ["gtfs", "gtfs_access", "transit_feed"],
            ExposureFamily.GREEN_BLUE_SPACE.value: _EXPOSURE_KEYWORDS.get(
                ExposureFamily.GREEN_BLUE_SPACE.value, []
            ) + ["park", "parks", "ndvi", "greenness"],
            ExposureFamily.BUILT_ENVIRONMENT.value: _EXPOSURE_KEYWORDS.get(
                ExposureFamily.BUILT_ENVIRONMENT.value, []
            ) + ["streetscape", "street_view"],
        }
    if enum_cls is OutcomeFamily:
        return _OUTCOME_KEYWORDS
    if enum_cls is ContributionPrimary:
        return {
            ContributionPrimary.NOVEL_CONTEXT.value: ["novel_context", "new context"],
            ContributionPrimary.NOVEL_METHOD.value: ["novel_method", "new method"],
            ContributionPrimary.NOVEL_DATA.value: ["novel_data", "new dataset"],
            ContributionPrimary.CAUSAL_REFINEMENT.value: ["causal_refinement", "identification improvement"],
            ContributionPrimary.POLICY_EVALUATION.value: ["policy_evaluation", "policy eval"],
            ContributionPrimary.THEORY_BUILDING.value: ["theory_building", "theory"],
            ContributionPrimary.REPLICATION.value: ["replication", "replicate"],
            ContributionPrimary.META_ANALYSIS.value: ["meta_analysis", "meta analysis"],
            ContributionPrimary.OTHER.value: ["other"],
        }
    return {}


def normalize_family(raw: str, families: type[E]) -> E:
    matched = _try_match_enum_value(raw, families)
    if matched is not None:
        return matched

    raw_norm = _norm_text(raw)
    if not raw_norm:
        raise SeedNormalizationError(f"empty family for {families.__name__}")

    mapping = _keyword_mapping_for_enum(families)
    if mapping:
        for enum_value, keywords in mapping.items():
            norm_keywords = {_norm_text(k) for k in keywords + [enum_value]}
            if any(k in raw_norm or raw_norm in k for k in norm_keywords if k):
                return families(enum_value)  # type: ignore[arg-type]

    raise SeedNormalizationError(
        f"cannot normalize family '{raw}' to {families.__name__}"
    )


_METHOD_ALIASES: dict[IdentificationPrimary, list[str]] = {
    IdentificationPrimary.DID: [
        "diff_in_diff",
        "difference_in_differences",
        "difference_in_difference",
        "did",
    ],
    IdentificationPrimary.RDD: [
        "regression_discontinuity",
        "rdd",
        "spatial_discontinuity",
    ],
    IdentificationPrimary.IV: [
        "instrumental_variable",
        "iv",
        "iv_instrumental_variables",
        "2sls",
        "two_stage_least_squares",
    ],
    IdentificationPrimary.PSM: [
        "propensity_score_matching",
        "matching",
        "psm",
        "propensity_score",
    ],
    IdentificationPrimary.FE: ["fixed_effects", "fe"],
    IdentificationPrimary.SYNTHETIC_CONTROL: ["synthetic_control"],
    IdentificationPrimary.EVENT_STUDY: ["event_study"],
    IdentificationPrimary.CAUSAL_FOREST: ["causal_forest"],
    IdentificationPrimary.OLS: ["ols_regression", "ols"],
    IdentificationPrimary.SPATIAL_REGRESSION: ["spatial_regression"],
    IdentificationPrimary.SURVIVAL: ["survival_analysis"],
    IdentificationPrimary.MACHINE_LEARNING: ["machine_learning", "ml"],
    IdentificationPrimary.DESCRIPTIVE: ["descriptive"],
    IdentificationPrimary.OTHER: ["other"],
}


def normalize_method(raw: str) -> IdentificationPrimary:
    matched = _try_match_enum_value(raw, IdentificationPrimary)
    if matched is not None:
        return matched

    raw_norm = _norm_text(raw)
    if not raw_norm:
        raise SeedNormalizationError("empty method")

    for method, aliases in _METHOD_ALIASES.items():
        alias_norm = {_norm_text(a) for a in aliases}
        if raw_norm in alias_norm:
            return method

    # Keyword dictionaries used by the OpenAlex verifier are also valid hints.
    for method_value, keywords in _METHOD_KEYWORDS.items():
        norm_keywords = {_norm_text(k) for k in keywords + [method_value]}
        if any(k in raw_norm or raw_norm in k for k in norm_keywords if k):
            return IdentificationPrimary(method_value)

    raise SeedNormalizationError(f"cannot normalize method '{raw}'")


def normalize_mitigations(raw, threats: list[str]) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {
            str(k): str(v)
            for k, v in raw.items()
            if str(k) in set(threats)
        }
    if isinstance(raw, list):
        return {
            str(threat): str(raw[idx])
            for idx, threat in enumerate(threats)
            if idx < len(raw)
        }
    if isinstance(raw, str) and threats:
        return {str(threats[0]): raw}
    return {}
