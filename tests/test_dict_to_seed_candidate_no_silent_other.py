"""Tests for seed conversion to ensure no silent OTHER fallback."""

import pytest

from agents.ideation_agent_v2 import _dict_to_seed_candidate
from agents.seed_normalizer import SeedNormalizationError
from models.topic_schema import ExposureFamily, OutcomeFamily


def _seed_item(
    topic_id: str,
    exposure_family: str,
    outcome_family: str,
    method: str = "fixed_effects",
    geography: str = "US cities",
) -> dict:
    return {
        "topic_id": topic_id,
        "title": f"title-{topic_id}",
        "exposure_family": exposure_family,
        "exposure_specific": f"x-{topic_id}",
        "exposure_spatial_unit": "tract",
        "outcome_family": outcome_family,
        "outcome_specific": f"y-{topic_id}",
        "outcome_spatial_unit": "tract",
        "geography": geography,
        "spatial_unit": "tract",
        "sampling_mode": "panel",
        "start_year": 2010,
        "end_year": 2020,
        "frequency": "annual",
        "method": method,
        "key_threats": ["confounding"],
        "mitigations": {"confounding": "unit and time fixed effects"},
        "contribution_type": "novel_context",
        "contribution_statement": "statement",
        "gap_addressed": "gap",
        "declared_sources": ["NHGIS"],
        "target_venues": ["Nature Cities"],
    }


def test_dict_to_seed_candidate_no_silent_other_and_distinct_signatures():
    items = [
        _seed_item("seed_001", "air pollution", "health", method="fixed_effects", geography="US"),
        _seed_item("seed_002", "walkability", "mobility", method="diff_in_diff", geography="state:CA"),
        _seed_item("seed_003", "GTFS access", "economic", method="matching", geography="metro:31080"),
        _seed_item("seed_004", "NDVI", "wellbeing", method="event_study", geography="country:Japan"),
        _seed_item("seed_005", "streetscape", "safety", method="iv_instrumental_variables", geography="conus"),
    ]

    candidates = [
        _dict_to_seed_candidate(item, "Urban Health", idx)
        for idx, item in enumerate(items)
    ]
    topics = [c.topic for c in candidates]
    signatures = [t.four_tuple_signature() for t in topics]

    assert len(set(signatures)) == len(signatures)
    for topic in topics:
        assert topic.exposure_X.family != ExposureFamily.OTHER
        assert topic.outcome_Y.family != OutcomeFamily.OTHER


def test_dict_to_seed_candidate_raises_on_unmappable_family():
    with pytest.raises(SeedNormalizationError):
        _dict_to_seed_candidate(
            _seed_item("seed_006", "%%%unknown_family%%%", "health"),
            "Urban Health",
            0,
        )
