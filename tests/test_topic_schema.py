"""Tests for models/topic_schema.py — Day 1 TDD."""

import hashlib
import pytest
from pydantic import ValidationError

from models.topic_schema import (
    ContributionPrimary,
    ExposureFamily,
    ExposureX,
    FinalStatus,
    HITLInterruption,
    IdentificationPrimary,
    IdentificationStrategy,
    OutcomeFamily,
    OutcomeY,
    SamplingMode,
    SeedCandidate,
    SpatialScope,
    TemporalScope,
    Topic,
    TopicMeta,
    Contribution,
    Frequency,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_topic(
    topic_id: str = "t001",
    exposure_family: ExposureFamily = ExposureFamily.BUILT_ENVIRONMENT,
    exposure_var: str = "street_connectivity",
    outcome_family: OutcomeFamily = OutcomeFamily.HEALTH,
    outcome_var: str = "obesity_prevalence",
    geography: str = "US cities",
    spatial_unit: str = "tract",
    method: IdentificationPrimary = IdentificationPrimary.FE,
) -> Topic:
    return Topic(
        meta=TopicMeta(topic_id=topic_id),
        exposure_X=ExposureX(
            family=exposure_family,
            specific_variable=exposure_var,
            spatial_unit=spatial_unit,
        ),
        outcome_Y=OutcomeY(
            family=outcome_family,
            specific_variable=outcome_var,
            spatial_unit=spatial_unit,
        ),
        spatial_scope=SpatialScope(
            geography=geography,
            spatial_unit=spatial_unit,
            sampling_mode=SamplingMode.PANEL,
        ),
        temporal_scope=TemporalScope(
            start_year=2010,
            end_year=2020,
            frequency=Frequency.ANNUAL,
        ),
        identification=IdentificationStrategy(
            primary=method,
            key_threats=["reverse_causality", "omitted_variable"],
            mitigations={
                "reverse_causality": "lagged_exposure",
                "omitted_variable": "city_fe",
            },
        ),
        contribution=Contribution(
            primary=ContributionPrimary.NOVEL_CONTEXT,
            statement="First causal estimate using US Census panel data.",
            gap_addressed="Prior studies rely on cross-sectional designs.",
        ),
        target_venues=["Nature Cities"],
        free_form_title="Street connectivity and obesity in US cities",
    )


# ── Test 1: basic construction ────────────────────────────────────────────────

def test_topic_construction():
    t = make_topic()
    assert t.meta.topic_id == "t001"
    assert t.exposure_X.family == ExposureFamily.BUILT_ENVIRONMENT
    assert t.outcome_Y.family == OutcomeFamily.HEALTH
    assert t.temporal_scope.start_year == 2010


# ── Test 2: four_tuple_signature is stable and deterministic ─────────────────

def test_four_tuple_signature_deterministic():
    t = make_topic()
    sig1 = t.four_tuple_signature()
    sig2 = t.four_tuple_signature()
    assert sig1 == sig2
    assert len(sig1) == 32  # MD5 hex


def test_four_tuple_signature_varies_with_method():
    t1 = make_topic(method=IdentificationPrimary.FE)
    t2 = make_topic(method=IdentificationPrimary.DID)
    assert t1.four_tuple_signature() != t2.four_tuple_signature()


def test_four_tuple_signature_varies_with_exposure():
    t1 = make_topic(exposure_family=ExposureFamily.BUILT_ENVIRONMENT)
    t2 = make_topic(exposure_family=ExposureFamily.AIR_QUALITY)
    assert t1.four_tuple_signature() != t2.four_tuple_signature()


def test_four_tuple_signature_same_for_different_topic_ids():
    t1 = make_topic(topic_id="t001")
    t2 = make_topic(topic_id="t002")
    # topic_id is not part of the 4-tuple
    assert t1.four_tuple_signature() == t2.four_tuple_signature()


# ── Test 3: TemporalScope validation ─────────────────────────────────────────

def test_temporal_scope_invalid_end_before_start():
    with pytest.raises(ValidationError):
        TemporalScope(start_year=2020, end_year=2010, frequency=Frequency.ANNUAL)


def test_temporal_scope_same_year_ok():
    ts = TemporalScope(start_year=2015, end_year=2015, frequency=Frequency.CROSS_SECTIONAL)
    assert ts.start_year == ts.end_year


# ── Test 4: to_legacy_dict ────────────────────────────────────────────────────

def test_to_legacy_dict_keys():
    t = make_topic()
    d = t.to_legacy_dict()
    for key in ["title", "abstract", "exposure_variable", "outcome_variable",
                "geography", "method", "contribution", "topic_id"]:
        assert key in d


def test_to_legacy_dict_uses_free_form_title():
    t = make_topic()
    d = t.to_legacy_dict()
    assert d["title"] == "Street connectivity and obesity in US cities"


def test_to_legacy_dict_fallback_title_when_empty():
    t = make_topic()
    t.free_form_title = ""
    d = t.to_legacy_dict()
    assert "street_connectivity" in d["title"]


# ── Test 5: SeedCandidate dataclass ──────────────────────────────────────────

def test_seed_candidate_construction():
    t = make_topic()
    sc = SeedCandidate(
        topic=t,
        declared_sources=["NHGIS", "CDC_PLACES"],
        declared_sources_rationale="NHGIS for exposure; CDC PLACES for outcome.",
    )
    assert sc.topic is t
    assert "NHGIS" in sc.declared_sources
    assert sc.declared_sources_rationale != ""


def test_seed_candidate_default_empty_sources():
    t = make_topic()
    sc = SeedCandidate(topic=t)
    assert sc.declared_sources == []
    assert sc.declared_sources_rationale == ""


# ── Test 6: FinalStatus enum ─────────────────────────────────────────────────

def test_final_status_values():
    assert FinalStatus.ACCEPTED == "ACCEPTED"
    assert FinalStatus.TENTATIVE == "TENTATIVE"
    assert FinalStatus.REJECTED == "REJECTED"
    assert FinalStatus.PENDING == "PENDING"


# ── Test 7: HITLInterruption exception ───────────────────────────────────────

def test_hitl_interruption_hard_blocker():
    exc = HITLInterruption(
        kind="hard_blocker_failed",
        message="G3 data_availability failed",
        failed_gates=["G3"],
        suggested_operations=[{"op": "change_geography", "params": {}}],
    )
    assert exc.kind == "hard_blocker_failed"
    assert "G3" in exc.failed_gates
    assert isinstance(exc, Exception)


def test_hitl_interruption_refinable_still_failing():
    exc = HITLInterruption(
        kind="refinable_still_failing_after_one_round",
        diff_from_original={"identification.primary": "fixed_effects → diff_in_diff"},
        suggested_next_operations=[{"op": "add_mitigations"}],
    )
    assert exc.kind == "refinable_still_failing_after_one_round"
    assert exc.diff_from_original != {}
    assert len(exc.suggested_next_operations) == 1


def test_hitl_interruption_defaults():
    exc = HITLInterruption(kind="hard_blocker_failed")
    assert exc.failed_gates == []
    assert exc.suggested_operations == []
    assert exc.diff_from_original == {}


# ── Test 8: enum completeness ─────────────────────────────────────────────────

def test_identification_primary_includes_key_methods():
    ids = {m.value for m in IdentificationPrimary}
    assert "diff_in_diff" in ids
    assert "regression_discontinuity" in ids
    assert "fixed_effects" in ids


def test_exposure_family_other_catchall():
    assert ExposureFamily.OTHER.value == "other"


def test_outcome_family_other_catchall():
    assert OutcomeFamily.OTHER.value == "other"
