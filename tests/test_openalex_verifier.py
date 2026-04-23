"""Tests for agents/openalex_verifier.py — Day 3 TDD.

All tests mock pyalex to avoid live network calls.
Marked not live_openalex so they run in the standard mock suite.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.openalex_verifier import (
    NoveltyEvidence,
    OpenAlexVerifier,
    _expand_family_keywords,
    _papers_match_four_tuple,
    _EXPOSURE_KEYWORDS,
    _OUTCOME_KEYWORDS,
    _METHOD_KEYWORDS,
)
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


def make_topic(
    topic_id: str = "t_v",
    exposure: ExposureFamily = ExposureFamily.AIR_QUALITY,
    outcome: OutcomeFamily = OutcomeFamily.HEALTH,
    geography: str = "US cities",
    method: IdentificationPrimary = IdentificationPrimary.DID,
) -> Topic:
    return Topic(
        meta=TopicMeta(topic_id=topic_id),
        exposure_X=ExposureX(
            family=exposure, specific_variable="PM2.5", spatial_unit="tract"
        ),
        outcome_Y=OutcomeY(
            family=outcome, specific_variable="cardiovascular mortality", spatial_unit="tract"
        ),
        spatial_scope=SpatialScope(
            geography=geography, spatial_unit="tract", sampling_mode=SamplingMode.PANEL
        ),
        temporal_scope=TemporalScope(
            start_year=2010, end_year=2020, frequency=Frequency.ANNUAL
        ),
        identification=IdentificationStrategy(
            primary=method, key_threats=["confounding"], mitigations=["FE"]
        ),
        contribution=Contribution(
            primary=ContributionPrimary.CAUSAL_REFINEMENT,
            statement="Causal evidence on air quality and cardiovascular mortality.",
        ),
    )


# ── keyword expansion ─────────────────────────────────────────────────────────

def test_expand_exposure_keywords_returns_list():
    kws = _expand_family_keywords(ExposureFamily.AIR_QUALITY.value, _EXPOSURE_KEYWORDS)
    assert isinstance(kws, list)
    assert len(kws) > 0


def test_expand_unknown_family_returns_value():
    kws = _expand_family_keywords("totally_unknown", _EXPOSURE_KEYWORDS)
    assert kws == ["totally_unknown"]


def test_expand_method_keywords_did():
    kws = _expand_family_keywords(IdentificationPrimary.DID.value, _METHOD_KEYWORDS)
    assert any("difference" in k.lower() for k in kws)


# ── four-tuple matching ───────────────────────────────────────────────────────

def test_paper_matches_all_four_dimensions():
    t = make_topic(
        exposure=ExposureFamily.AIR_QUALITY,
        outcome=OutcomeFamily.HEALTH,
        geography="China",
        method=IdentificationPrimary.DID,
    )
    paper = {
        "title": "Air pollution and health outcomes in China",
        "abstract": "Using difference-in-differences we estimate the effect of PM2.5 on mortality.",
    }
    assert _papers_match_four_tuple(paper, t) is True


def test_paper_missing_geography_does_not_match():
    t = make_topic(
        exposure=ExposureFamily.AIR_QUALITY,
        outcome=OutcomeFamily.HEALTH,
        geography="Brazil",
        method=IdentificationPrimary.DID,
    )
    paper = {
        "title": "Air pollution and health outcomes in China",
        "abstract": "Using difference-in-differences we estimate the effect.",
    }
    assert _papers_match_four_tuple(paper, t) is False


def test_paper_missing_method_does_not_match():
    t = make_topic(method=IdentificationPrimary.DID)
    paper = {
        "title": "Air pollution and health in US cities",
        "abstract": "OLS regression on PM2.5 and cardiovascular disease.",
    }
    assert _papers_match_four_tuple(paper, t) is False


# ── verify_novelty_four_tuple (mocked pyalex) ─────────────────────────────────

@patch("agents.openalex_verifier.load_json_cache", return_value=None)
@patch("agents.openalex_verifier.save_json_cache")
def test_verify_returns_novelty_evidence(mock_save, mock_load):
    mock_works = MagicMock()
    mock_works.search.return_value = mock_works
    mock_works.sort.return_value = mock_works
    mock_works.get.return_value = [
        {
            "id": "W1",
            "title": "Air quality and health in US cities using difference-in-differences",
            "abstract": "diff-in-diff study of PM2.5 and mortality in US cities.",
            "publication_year": 2020,
            "cited_by_count": 50,
        }
    ]

    with patch.dict("sys.modules", {"pyalex": MagicMock(Works=lambda: mock_works)}):
        verifier = OpenAlexVerifier(top_k=50)
        t = make_topic()
        evidence = verifier.verify_novelty_four_tuple(t)

    assert isinstance(evidence, NoveltyEvidence)
    assert evidence.was_fallback is False
    assert evidence.total_hits == 1
    assert len(evidence.top_k_papers) == 1
    assert evidence.four_tuple_match_count >= 0


@patch("agents.openalex_verifier.load_json_cache", return_value=None)
@patch("agents.openalex_verifier.save_json_cache")
def test_verify_returns_fallback_on_api_error(mock_save, mock_load):
    with patch.dict("sys.modules", {"pyalex": None}):
        # Simulate ImportError by making pyalex import fail
        import sys
        original = sys.modules.get("pyalex")
        sys.modules["pyalex"] = None  # type: ignore[assignment]
        try:
            verifier = OpenAlexVerifier()
            t = make_topic()
            evidence = verifier.verify_novelty_four_tuple(t)
        finally:
            if original is None:
                sys.modules.pop("pyalex", None)
            else:
                sys.modules["pyalex"] = original

    assert evidence.was_fallback is True
    assert evidence.total_hits == 0
    assert evidence.four_tuple_match_count == 0


@patch("agents.openalex_verifier.load_json_cache")
@patch("agents.openalex_verifier.save_json_cache")
def test_verify_uses_cache_on_second_call(mock_save, mock_load):
    cached = {
        "total_hits": 5,
        "top_k_papers": [],
        "four_tuple_match_count": 1,
        "queries_log": ["test query"],
        "was_fallback": False,
    }
    mock_load.return_value = cached

    verifier = OpenAlexVerifier()
    t = make_topic()
    evidence = verifier.verify_novelty_four_tuple(t)

    assert evidence.total_hits == 5
    assert evidence.four_tuple_match_count == 1
    mock_save.assert_not_called()


# ── NoveltyEvidence dataclass ─────────────────────────────────────────────────

def test_novelty_evidence_defaults():
    ev = NoveltyEvidence(
        total_hits=10,
        top_k_papers=[],
        four_tuple_match_count=2,
        queries_log=["q1"],
    )
    assert ev.was_fallback is False
