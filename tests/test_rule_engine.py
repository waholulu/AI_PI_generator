"""Tests for agents/rule_engine.py — Day 2 TDD.

Each gate has >= 3 boundary cases.  All tests use real YAML configs
(they must exist, verified by Day 1 tests).  No LLM calls.
"""

import pytest

from agents.rule_engine import GateResult, RuleEngine
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


# ── Shared fixture ────────────────────────────────────────────────────────────

def make_topic(
    topic_id: str = "t_test",
    x_spatial: str = "tract",
    y_spatial: str = "tract",
    geography: str = "US cities",
    start: int = 2010,
    end: int = 2020,
    method: IdentificationPrimary = IdentificationPrimary.FE,
    key_threats: list[str] | None = None,
    mitigations: list[str] | None = None,
) -> Topic:
    return Topic(
        meta=TopicMeta(topic_id=topic_id),
        exposure_X=ExposureX(
            family=ExposureFamily.BUILT_ENVIRONMENT,
            specific_variable="street_connectivity",
            spatial_unit=x_spatial,
        ),
        outcome_Y=OutcomeY(
            family=OutcomeFamily.HEALTH,
            specific_variable="obesity_prevalence",
            spatial_unit=y_spatial,
        ),
        spatial_scope=SpatialScope(
            geography=geography,
            spatial_unit=x_spatial,
            sampling_mode=SamplingMode.PANEL,
        ),
        temporal_scope=TemporalScope(
            start_year=start,
            end_year=end,
            frequency=Frequency.ANNUAL,
        ),
        identification=IdentificationStrategy(
            primary=method,
            key_threats=key_threats if key_threats is not None else ["reverse_causality"],
            mitigations=mitigations if mitigations is not None else ["lagged_exposure"],
        ),
        contribution=Contribution(
            primary=ContributionPrimary.NOVEL_CONTEXT,
            statement="First causal estimate.",
        ),
    )


@pytest.fixture(scope="module")
def engine():
    return RuleEngine()


# ── G2: scale_alignment ───────────────────────────────────────────────────────

class TestG2ScaleAlignment:
    def test_same_unit_passes(self, engine):
        t = make_topic(x_spatial="tract", y_spatial="tract")
        r = engine.check_G2_scale_alignment(t)
        assert r.passed is True
        assert r.gate_id == "G2"
        assert r.refinable is False

    def test_adjacent_units_pass(self, engine):
        # tract (rank 4) vs block_group (rank 3) → diff = 1 ≤ 4
        t = make_topic(x_spatial="tract", y_spatial="block_group")
        r = engine.check_G2_scale_alignment(t)
        assert r.passed is True

    def test_rank_diff_exactly_4_passes(self, engine):
        # tract (4) vs country (10) → diff = 6 → fail
        # tract (4) vs state (8) → diff = 4 → pass
        t = make_topic(x_spatial="tract", y_spatial="state")
        r = engine.check_G2_scale_alignment(t)
        assert r.passed is True
        assert r.details["rank_diff"] == 4

    def test_rank_diff_5_fails(self, engine):
        # block (2) vs metro_area (7) → diff = 5 → fail
        t = make_topic(x_spatial="block", y_spatial="metro_area")
        r = engine.check_G2_scale_alignment(t)
        assert r.passed is False
        assert r.refinable is False
        assert "rank_diff=5" in r.reason

    def test_unknown_unit_skips_gracefully(self, engine):
        t = make_topic(x_spatial="nonexistent_unit", y_spatial="tract")
        r = engine.check_G2_scale_alignment(t)
        assert r.passed is True
        assert "unknown_spatial_unit_skip" in r.reason

    def test_point_vs_country_fails(self, engine):
        # point (1) vs country (10) → diff = 9 → fail
        t = make_topic(x_spatial="point", y_spatial="country")
        r = engine.check_G2_scale_alignment(t)
        assert r.passed is False


# ── G3: data_availability ─────────────────────────────────────────────────────

class TestG3DataAvailability:
    def test_valid_source_and_years_passes(self, engine):
        t = make_topic(start=2010, end=2020, x_spatial="tract", y_spatial="tract")
        r = engine.check_G3_data_availability(t, ["NHGIS"])
        assert r.passed is True
        assert r.gate_id == "G3"

    def test_empty_declared_sources_fails(self, engine):
        t = make_topic()
        r = engine.check_G3_data_availability(t, [])
        assert r.passed is False
        assert "no_declared_sources" in r.reason

    def test_unknown_source_fails(self, engine):
        t = make_topic()
        r = engine.check_G3_data_availability(t, ["TOTALLY_UNKNOWN_DB_XYZ"])
        assert r.passed is False
        assert "no_sources_in_catalog" in r.reason

    def test_year_before_coverage_fails(self, engine):
        # NHGIS starts 1790; request 1700-1800 should fail on end > y_max only if year_max < 1800
        # Actually NHGIS coverage_year_max=2023, year_min=1790
        # Request start=1700 < 1790 → year_gap
        t = make_topic(start=1700, end=1800)
        r = engine.check_G3_data_availability(t, ["NHGIS"])
        assert r.passed is False
        assert any("year_gap" in issue for issue in r.details["coverage_issues"])

    def test_year_after_coverage_fails(self, engine):
        # NHGIS ends 2023; request 2025 → year_gap
        t = make_topic(start=2020, end=2025)
        r = engine.check_G3_data_availability(t, ["NHGIS"])
        assert r.passed is False
        assert any("year_gap" in issue for issue in r.details["coverage_issues"])

    def test_multiple_valid_sources_pass(self, engine):
        t = make_topic(start=2016, end=2020, x_spatial="tract", y_spatial="tract")
        r = engine.check_G3_data_availability(t, ["NHGIS", "CDC_PLACES"])
        assert r.passed is True

    def test_alias_lookup_works(self, engine):
        # "osm" is alias for OpenStreetMap; temporal scope should be covered
        t = make_topic(start=2010, end=2020, x_spatial="point", y_spatial="point")
        r = engine.check_G3_data_availability(t, ["osm"])
        assert r.passed is True


# ── G6: automation_feasibility ────────────────────────────────────────────────

class TestG6AutomationFeasibility:
    def test_available_source_and_method_passes(self, engine):
        t = make_topic(method=IdentificationPrimary.FE)
        r = engine.check_G6_automation_feasibility(t, ["NHGIS"])
        assert r.passed is True
        assert r.gate_id == "G6"
        assert r.refinable is False

    def test_planned_method_skill_fails(self, engine):
        # synthetic_control is status=planned in skill_registry.yaml
        t = make_topic(method=IdentificationPrimary.SYNTHETIC_CONTROL)
        r = engine.check_G6_automation_feasibility(t, ["NHGIS"])
        assert r.passed is False
        assert "synthetic_control" in r.details["missing_skills"]

    def test_multiple_available_skills_pass(self, engine):
        t = make_topic(method=IdentificationPrimary.DID)
        r = engine.check_G6_automation_feasibility(t, ["NHGIS", "EPA_AQS"])
        assert r.passed is True

    def test_no_sources_with_available_method_passes(self, engine):
        t = make_topic(method=IdentificationPrimary.OLS)
        r = engine.check_G6_automation_feasibility(t, [])
        assert r.passed is True

    def test_required_skills_collected_from_sources(self, engine):
        t = make_topic(method=IdentificationPrimary.OLS)
        r = engine.check_G6_automation_feasibility(t, ["NHGIS"])
        # NHGIS needs census_extract + spatial_join; both should be in required
        assert "census_extract" in r.details["required_skills"]
        assert "spatial_join" in r.details["required_skills"]


# ── G4: threat_coverage ───────────────────────────────────────────────────────

class TestG4ThreatCoverage:
    def test_full_coverage_passes(self, engine):
        t = make_topic(
            key_threats=["reverse_causality", "selection_bias"],
            mitigations=["lagged_exposure", "psm_matching"],
        )
        # lagged_exposure ~ reverse_causality? ratio may be low; use explicit matches
        t2 = make_topic(
            key_threats=["confounding"],
            mitigations=["confounding_adjustment"],
        )
        r = engine.check_G4_threat_coverage(t2)
        assert r.passed is True
        assert r.refinable is True

    def test_no_threats_skips(self, engine):
        t = make_topic(key_threats=[], mitigations=["lagged_exposure"])
        r = engine.check_G4_threat_coverage(t)
        assert r.passed is True
        assert "no_threats_declared_skip" in r.reason

    def test_zero_coverage_fails(self, engine):
        t = make_topic(
            key_threats=["attrition_bias", "measurement_error"],
            mitigations=[],
        )
        r = engine.check_G4_threat_coverage(t)
        assert r.passed is False
        assert r.details["covered"] == 0

    def test_partial_coverage_below_threshold_fails(self, engine):
        # 1 threat covered, 2 total → 50% < 80%
        t = make_topic(
            key_threats=["confounding", "selection_bias", "omitted_variable"],
            mitigations=["confounding_correction"],
        )
        r = engine.check_G4_threat_coverage(t)
        # coverage_ratio depends on fuzzy match; with exact match on first threat only
        # ratio = 1/3 ≈ 33% < 80%
        assert r.passed is False

    def test_gate_id_is_G4(self, engine):
        t = make_topic()
        r = engine.check_G4_threat_coverage(t)
        assert r.gate_id == "G4"


# ── run_hard_blockers convenience ─────────────────────────────────────────────

class TestRunHardBlockers:
    def test_returns_three_results(self, engine):
        t = make_topic()
        results = engine.run_hard_blockers(t, ["NHGIS"])
        assert len(results) == 3
        gate_ids = {r.gate_id for r in results}
        assert gate_ids == {"G2", "G3", "G6"}

    def test_all_pass_for_valid_topic(self, engine):
        t = make_topic(x_spatial="tract", y_spatial="tract",
                       start=2010, end=2020,
                       method=IdentificationPrimary.FE)
        results = engine.run_hard_blockers(t, ["NHGIS"])
        assert all(r.passed for r in results)

    def test_hard_blocker_result_has_refinable_false(self, engine):
        t = make_topic()
        results = engine.run_hard_blockers(t, ["NHGIS"])
        for r in results:
            assert r.refinable is False
