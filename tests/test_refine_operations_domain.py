from __future__ import annotations

from agents.reflection_loop import _apply_operations, _select_refine_operations
from agents.rule_engine import GateResult
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
    SeedCandidate,
    SpatialScope,
    TemporalScope,
    Topic,
    TopicMeta,
)


def _make_seed() -> SeedCandidate:
    topic = Topic(
        meta=TopicMeta(topic_id="seed_domain"),
        exposure_X=ExposureX(
            family=ExposureFamily.AIR_QUALITY,
            specific_variable="PM2.5",
            spatial_unit="tract",
        ),
        outcome_Y=OutcomeY(
            family=OutcomeFamily.HEALTH,
            specific_variable="mortality",
            spatial_unit="tract",
        ),
        spatial_scope=SpatialScope(
            geography="US cities",
            spatial_unit="tract",
            sampling_mode=SamplingMode.PANEL,
        ),
        temporal_scope=TemporalScope(
            start_year=2010,
            end_year=2020,
            frequency=Frequency.ANNUAL,
        ),
        identification=IdentificationStrategy(
            primary=IdentificationPrimary.FE,
            key_threats=["confounding"],
            mitigations={"confounding": "fixed effects"},
        ),
        contribution=Contribution(
            primary=ContributionPrimary.CAUSAL_REFINEMENT,
            statement="Initial contribution.",
        ),
    )
    return SeedCandidate(topic=topic, declared_sources=["NHGIS"])


def test_aggregate_X_up_only_changes_exposure_unit():
    seed = _make_seed()
    new_topic, _, _ = _apply_operations(
        seed.topic,
        [{"op": "aggregate_X_up", "params": {"target_unit": "county"}}],
        seed,
    )
    assert new_topic.exposure_X.spatial_unit == "county"
    assert new_topic.outcome_Y.spatial_unit == "tract"
    assert new_topic.spatial_scope.spatial_unit == "tract"


def test_free_form_applies_dot_path_modifications():
    seed = _make_seed()
    new_topic, _, slot_diff = _apply_operations(
        seed.topic,
        [
            {
                "op": "free_form",
                "params": {
                    "rationale": "Need shock",
                    "modified_fields": {"identification.requires_exogenous_shock": True},
                },
            }
        ],
        seed,
    )
    assert new_topic.identification.requires_exogenous_shock is True
    assert slot_diff["free_form"]["fields"] == ["identification.requires_exogenous_shock"]


def test_anti_oscillation_skips_inverse_pair():
    failed = [GateResult("G2", "scale_alignment", False, True, "bad")]
    # Simulate previous round used aggregate_X_up.
    ops = _select_refine_operations(failed, history=["aggregate_X_up"])
    assert all(op.get("op") != "disaggregate_Y_down" for op in ops)
