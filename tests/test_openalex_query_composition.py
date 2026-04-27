from __future__ import annotations

from unittest.mock import patch

from agents.openalex_verifier import OpenAlexVerifier
from models.topic_schema import (
    Contribution,
    ContributionPrimary,
    ExposureFamily,
    ExposureX,
    Frequency,
    IdentificationPrimary,
    IdentificationStrategy,
    OutcomeFamily,
    OutcomeY,
    SamplingMode,
    SpatialScope,
    TemporalScope,
    Topic,
    TopicMeta,
)


def _topic(exposure: ExposureFamily, outcome: OutcomeFamily, method: IdentificationPrimary, geography: str) -> Topic:
    return Topic(
        meta=TopicMeta(topic_id="qtest"),
        exposure_X=ExposureX(
            family=exposure,
            specific_variable="x",
            spatial_unit="tract",
        ),
        outcome_Y=OutcomeY(
            family=outcome,
            specific_variable="y",
            spatial_unit="tract",
        ),
        spatial_scope=SpatialScope(
            geography=geography,
            spatial_unit="tract",
            sampling_mode=SamplingMode.PANEL,
        ),
        temporal_scope=TemporalScope(
            start_year=2010,
            end_year=2020,
            frequency=Frequency.ANNUAL,
        ),
        identification=IdentificationStrategy(
            primary=method,
            key_threats=["confounding"],
            mitigations={"confounding": "fixed effects"},
        ),
        contribution=Contribution(
            primary=ContributionPrimary.NOVEL_CONTEXT,
            statement="s",
        ),
    )


def test_compose_query_contains_method_and_geo_keywords_for_multiple_topics():
    verifier = OpenAlexVerifier()
    topics = [
        _topic(ExposureFamily.AIR_QUALITY, OutcomeFamily.HEALTH, IdentificationPrimary.IV, "metro:31080"),
        _topic(ExposureFamily.BUILT_ENVIRONMENT, OutcomeFamily.MOBILITY, IdentificationPrimary.DID, "state:CA"),
        _topic(ExposureFamily.GREEN_BLUE_SPACE, OutcomeFamily.WELLBEING, IdentificationPrimary.EVENT_STUDY, "conus"),
        _topic(ExposureFamily.NOISE, OutcomeFamily.SAFETY, IdentificationPrimary.PSM, "country:Japan"),
        _topic(ExposureFamily.DENSITY, OutcomeFamily.ECONOMIC, IdentificationPrimary.FE, "US"),
        _topic(ExposureFamily.TRANSPORT_INFRA, OutcomeFamily.EQUITY, IdentificationPrimary.RDD, "city:Boston"),
    ]

    for topic in topics:
        q = verifier._compose_query(topic).lower()
        assert any(term in q for term in ["diff", "regression", "instrumental", "matching", "event", "fixed"])
        assert any(term in q for term in ["metropolitan", "state", "united states", "us", "japan", "boston"])


def test_metro_identifier_not_used_raw_in_query():
    verifier = OpenAlexVerifier()
    q = verifier._compose_query(
        _topic(ExposureFamily.AIR_QUALITY, OutcomeFamily.HEALTH, IdentificationPrimary.IV, "metro:31080")
    ).lower()
    assert "metro:31080" not in q


@patch("agents.openalex_verifier.load_json_cache", return_value=None)
def test_api_failure_sets_match_count_none(mock_cache):
    verifier = OpenAlexVerifier()
    with patch.object(verifier, "_search_openalex", side_effect=RuntimeError("boom")):
        ev = verifier.verify_novelty_four_tuple(
            _topic(ExposureFamily.AIR_QUALITY, OutcomeFamily.HEALTH, IdentificationPrimary.IV, "US")
        )
    assert ev.was_fallback is True
    assert ev.four_tuple_match_count is None
