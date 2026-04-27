import pytest

from agents.seed_normalizer import (
    SeedNormalizationError,
    normalize_family,
    normalize_method,
)
from models.topic_schema import (
    ContributionPrimary,
    ExposureFamily,
    IdentificationPrimary,
    OutcomeFamily,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("air_quality", ExposureFamily.AIR_QUALITY),
        ("Air Quality", ExposureFamily.AIR_QUALITY),
        ("air-quality", ExposureFamily.AIR_QUALITY),
        ("air_pollution", ExposureFamily.AIR_QUALITY),
        ("walkability", ExposureFamily.BUILT_ENVIRONMENT),
        ("parks", ExposureFamily.GREEN_BLUE_SPACE),
        ("transport infrastructure", ExposureFamily.TRANSPORT_INFRA),
    ],
)
def test_normalize_exposure_family_variants(raw, expected):
    assert normalize_family(raw, ExposureFamily) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("health", OutcomeFamily.HEALTH),
        ("Health", OutcomeFamily.HEALTH),
        ("economic", OutcomeFamily.ECONOMIC),
        ("mobility", OutcomeFamily.MOBILITY),
        ("safety", OutcomeFamily.SAFETY),
        ("well-being", OutcomeFamily.WELLBEING),
    ],
)
def test_normalize_outcome_family_variants(raw, expected):
    assert normalize_family(raw, OutcomeFamily) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("iv", IdentificationPrimary.IV),
        ("iv_instrumental_variables", IdentificationPrimary.IV),
        ("2sls", IdentificationPrimary.IV),
        ("matching", IdentificationPrimary.PSM),
        ("psm", IdentificationPrimary.PSM),
        ("propensity_score", IdentificationPrimary.PSM),
        ("spatial_discontinuity", IdentificationPrimary.RDD),
        ("fixed_effects", IdentificationPrimary.FE),
        ("diff_in_diff", IdentificationPrimary.DID),
    ],
)
def test_normalize_method_variants(raw, expected):
    assert normalize_method(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("novel_context", ContributionPrimary.NOVEL_CONTEXT),
        ("new method", ContributionPrimary.NOVEL_METHOD),
        ("policy eval", ContributionPrimary.POLICY_EVALUATION),
    ],
)
def test_normalize_contribution_family(raw, expected):
    assert normalize_family(raw, ContributionPrimary) == expected


def test_normalize_family_raises_on_unknown_value():
    with pytest.raises(SeedNormalizationError):
        normalize_family("totally_unknown_family_xyz", ExposureFamily)


def test_normalize_method_raises_on_unknown_value():
    with pytest.raises(SeedNormalizationError):
        normalize_method("unsupported_method_xyz")
