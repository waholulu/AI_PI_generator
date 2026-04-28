"""
Structured slot schema for Auto-PI topic representation (Module 1 upgrade).

Topic is the canonical five-dimensional representation replacing the legacy
free-form dict. All gate checks and reflection loop operate on Topic instances.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ────────────────────────────────────────────────────────────────────

class ExposureFamily(str, Enum):
    BUILT_ENVIRONMENT = "built_environment"
    TRANSPORT_INFRA = "transport_infra"
    GREEN_BLUE_SPACE = "green_blue_space"
    LAND_USE_MIX = "land_use_mix"
    DENSITY = "density"
    AIR_QUALITY = "air_quality"
    NOISE = "noise"
    HEAT_ISLAND = "heat_island"
    DIGITAL_INFRA = "digital_infra"
    GOVERNANCE = "governance"
    ECONOMIC_ACTIVITY = "economic_activity"
    SOCIAL_CAPITAL = "social_capital"
    OTHER = "other"


class OutcomeFamily(str, Enum):
    HEALTH = "health"
    MOBILITY = "mobility"
    WELLBEING = "wellbeing"
    ECONOMIC = "economic"
    ENVIRONMENT = "environment"
    SAFETY = "safety"
    EQUITY = "equity"
    COGNITION = "cognition"
    BEHAVIOR = "behavior"
    OTHER = "other"


class SamplingMode(str, Enum):
    CROSS_SECTIONAL = "cross_sectional"
    PANEL = "panel"
    LONGITUDINAL = "longitudinal"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    EXPERIMENTAL = "experimental"
    ECOLOGICAL = "ecological"


class Frequency(str, Enum):
    ANNUAL = "annual"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    SUB_DAILY = "sub_daily"
    DECADAL = "decadal"
    CROSS_SECTIONAL = "cross_sectional"
    IRREGULAR = "irregular"


class IdentificationPrimary(str, Enum):
    DID = "diff_in_diff"
    RDD = "regression_discontinuity"
    IV = "instrumental_variable"
    PSM = "propensity_score_matching"
    FE = "fixed_effects"
    SYNTHETIC_CONTROL = "synthetic_control"
    EVENT_STUDY = "event_study"
    CAUSAL_FOREST = "causal_forest"
    OLS = "ols_regression"
    SPATIAL_REGRESSION = "spatial_regression"
    SURVIVAL = "survival_analysis"
    MACHINE_LEARNING = "machine_learning"
    DESCRIPTIVE = "descriptive"
    OTHER = "other"


class ContributionPrimary(str, Enum):
    NOVEL_CONTEXT = "novel_context"
    NOVEL_METHOD = "novel_method"
    NOVEL_DATA = "novel_data"
    CAUSAL_REFINEMENT = "causal_refinement"
    POLICY_EVALUATION = "policy_evaluation"
    THEORY_BUILDING = "theory_building"
    REPLICATION = "replication"
    META_ANALYSIS = "meta_analysis"
    OTHER = "other"


class FinalStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    TENTATIVE = "TENTATIVE"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


# ── Sub-models ────────────────────────────────────────────────────────────────

class TopicMeta(BaseModel):
    topic_id: str
    seed_round: int = 0
    parent_topic_id: Optional[str] = None
    created_by: str = "ideation_v2"


class ExposureX(BaseModel):
    family: ExposureFamily
    specific_variable: str
    spatial_unit: str
    measurement_proxy: str = ""


class OutcomeY(BaseModel):
    family: OutcomeFamily
    specific_variable: str
    spatial_unit: str
    measurement_proxy: str = ""


class SpatialScope(BaseModel):
    geography: str
    spatial_unit: str
    sampling_mode: SamplingMode
    n_units_approx: Optional[int] = None


class TemporalScope(BaseModel):
    start_year: int
    end_year: int
    frequency: Frequency
    n_periods_approx: Optional[int] = None

    @field_validator("end_year")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        start = info.data.get("start_year")
        if start is not None and v < start:
            raise ValueError(f"end_year ({v}) must be >= start_year ({start})")
        return v


class IdentificationStrategy(BaseModel):
    primary: IdentificationPrimary
    key_threats: list[str] = Field(default_factory=list)
    mitigations: dict[str, str] = Field(default_factory=dict)
    requires_exogenous_shock: bool = False

    @field_validator("mitigations", mode="before")
    @classmethod
    def _coerce_mitigations(cls, value, info):
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        threats = info.data.get("key_threats", []) if info and info.data else []
        if isinstance(value, list):
            return {
                str(threat): str(value[idx])
                for idx, threat in enumerate(threats)
                if idx < len(value)
            }
        if isinstance(value, str) and threats:
            return {str(threats[0]): value}
        if value is None:
            return {}
        return value

    @model_validator(mode="after")
    def _validate_mitigations(self):
        for key in self.mitigations:
            if key not in self.key_threats:
                raise ValueError(
                    f"mitigation key '{key}' not in key_threats {self.key_threats}"
                )
        return self


class Contribution(BaseModel):
    primary: ContributionPrimary
    statement: str
    gap_addressed: str = ""


# ── Core model ────────────────────────────────────────────────────────────────

class Topic(BaseModel):
    meta: TopicMeta
    exposure_X: ExposureX
    outcome_Y: OutcomeY
    spatial_scope: SpatialScope
    temporal_scope: TemporalScope
    identification: IdentificationStrategy
    contribution: Contribution
    target_venues: list[str] = Field(default_factory=list, max_length=5)
    free_form_title: str = ""
    free_form_abstract: str = ""

    def four_tuple_signature(self) -> str:
        """Stable MD5 signature over (X_family, Y_family, geography, method).

        Used by reflection loop anti-oscillation check to detect cycling topics.
        """
        components = "|".join([
            self.exposure_X.family.value,
            self.outcome_Y.family.value,
            self.spatial_scope.geography.lower().strip(),
            self.identification.primary.value,
        ])
        return hashlib.md5(components.encode()).hexdigest()

    def to_legacy_dict(self) -> dict:
        """Convert to the flat dict format used by legacy downstream agents."""
        return {
            "title": self.free_form_title or (
                f"{self.exposure_X.specific_variable} → "
                f"{self.outcome_Y.specific_variable} in {self.spatial_scope.geography}"
            ),
            "abstract": self.free_form_abstract,
            "exposure_variable": self.exposure_X.specific_variable,
            "outcome_variable": self.outcome_Y.specific_variable,
            "geography": self.spatial_scope.geography,
            "method": self.identification.primary.value,
            "contribution": self.contribution.statement,
            "topic_id": self.meta.topic_id,
        }


# ── SeedCandidate ─────────────────────────────────────────────────────────────

@dataclass
class SeedCandidate:
    topic: Topic
    declared_sources: list[str] = field(default_factory=list)
    declared_sources_rationale: str = ""


# ── HITLInterruption ──────────────────────────────────────────────────────────

class HITLInterruption(Exception):
    """Raised by Level 1 path to surface gate failures or unresolved topics.

    kind options:
      "hard_blocker_failed"                 — G2/G3/G6 failed, no auto-refine
      "refinable_still_failing_after_one_round" — still TENTATIVE after max_rounds=1
    """

    def __init__(
        self,
        kind: str,
        message: str = "",
        failed_gates: Optional[list[str]] = None,
        suggested_operations: Optional[list[dict]] = None,
        diff_from_original: Optional[dict] = None,
        suggested_next_operations: Optional[list[dict]] = None,
    ) -> None:
        self.kind = kind
        self.message = message
        self.failed_gates = failed_gates or []
        self.suggested_operations = suggested_operations or []
        self.diff_from_original = diff_from_original or {}
        self.suggested_next_operations = suggested_next_operations or []
        super().__init__(message or kind)
