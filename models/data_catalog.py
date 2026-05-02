"""Pydantic schemas for the enhanced data-source catalog.

Each DataSourceProfile captures everything a downstream agent needs to know
about *how* to use a source: native spatial grain, aggregation recipes, join
keys, variable families with real column names, validation rules, and known
limitations.

These models are loaded by SourceRegistry.load() from
config/data_catalog/sources/*.yaml and stored alongside the flat dict used
by legacy callers.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ValidationRule(BaseModel):
    rule_id: str
    description: str
    severity: str = "warning"  # "warning" | "error"


class VariableSpec(BaseModel):
    name: str
    description: str = ""
    unit: str = ""
    dtype: str = "float"


class VariableFamilySpec(BaseModel):
    family_id: str = ""
    description: str = ""
    variables: list[VariableSpec] = Field(default_factory=list)
    aggregation_allowed_to: list[str] = Field(default_factory=list)
    join_keys: list[str] = Field(default_factory=list)


class TableSpec(BaseModel):
    table_id: str
    description: str = ""
    join_key: str = ""
    special_join_keys: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AggregationSpec(BaseModel):
    from_unit: str
    to_unit: str
    method: str
    weight_source: str = ""
    notes: str = ""


class JoinRecipe(BaseModel):
    recipe_id: str
    left_source: str
    right_source: str
    left_key: str
    right_key: str
    join_type: str = "left"
    notes: str = ""
    warnings: list[str] = Field(default_factory=list)


class TemporalCoverageSpec(BaseModel):
    coverage_year_min: int
    coverage_year_max: int
    release_lag_years: float = 0.0
    update_frequency: str = "unknown"
    cross_sectional_only: bool = False
    notes: str = ""


class GeographySpec(BaseModel):
    native_unit: str
    target_units_supported: list[str] = Field(default_factory=list)
    aggregation_required: bool = False
    default_aggregation: str = ""
    join_keys: list[str] = Field(default_factory=list)
    coverage: str = "nationwide"


class AcquisitionSpec(BaseModel):
    method: str = "download"
    url: str = ""
    documentation_url: str = ""
    expected_format: str = ""
    auth_required: bool = False
    cost_required: bool = False
    expected_files: list[str] = Field(default_factory=list)
    notes: str = ""


class DataSourceProfile(BaseModel):
    """Full semantic profile for a data source.

    Includes both the flat fields kept for backward compatibility and the
    nested structures used by the enhanced feasibility and spec-builder logic.
    """

    source_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    tier: str = "stable"
    cloud_safe: bool = True
    machine_readable: bool = True

    acquisition: AcquisitionSpec = Field(default_factory=AcquisitionSpec)
    geography: GeographySpec | None = None
    temporal_coverage: TemporalCoverageSpec | None = None

    tables: list[TableSpec] = Field(default_factory=list)
    variable_families: dict[str, Any] = Field(default_factory=dict)
    join_recipes: list[JoinRecipe] = Field(default_factory=list)
    validation_rules: list[ValidationRule] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)

    @property
    def native_unit(self) -> str:
        if self.geography:
            return self.geography.native_unit
        return ""

    @property
    def aggregation_required(self) -> bool:
        if self.geography:
            return self.geography.aggregation_required
        return False

    @property
    def default_aggregation(self) -> str:
        if self.geography:
            return self.geography.default_aggregation
        return ""

    def has_variable_mapping_for(self, family: str) -> bool:
        """Return True if the family has at least one concrete variable defined."""
        vf = self.variable_families.get(family)
        if not vf:
            return False
        if isinstance(vf, dict):
            variables = vf.get("variables", [])
            if not variables:
                return False
            # variables may be list[str] (old format) or list[dict/VariableSpec]
            return len(variables) > 0
        return False

    def get_join_recipes_for_target(self, target_unit: str) -> list[JoinRecipe]:
        """Return join recipes whose left_source is this source and target matches."""
        return [
            r for r in self.join_recipes
            if r.left_source == self.source_id
            and target_unit.replace("census_", "") in r.notes.lower()
            or (
                r.left_source == self.source_id
                and target_unit.replace("census_", "") in r.recipe_id.lower()
            )
        ]

    @classmethod
    def from_dict(cls, source_id: str, data: dict) -> "DataSourceProfile":
        """Construct from a raw YAML dict, handling both old and new formats."""
        payload = dict(data)
        payload.setdefault("source_id", source_id)

        # Normalize acquisition block
        if "acquisition" not in payload:
            payload["acquisition"] = {
                "method": payload.get("source_type", "download").replace("api_download", "api"),
                "url": payload.get("access_url", ""),
                "documentation_url": payload.get("documentation_url", ""),
                "expected_format": payload.get("expected_format", ""),
                "auth_required": payload.get("auth_required", False),
                "cost_required": payload.get("cost_required", False),
            }

        # Normalize geography block from flat fields
        if "geography" not in payload:
            spatial_units = payload.get("spatial_units", [])
            native = spatial_units[0] if spatial_units else "unknown"
            payload["geography"] = {
                "native_unit": native,
                "target_units_supported": spatial_units,
                "aggregation_required": False,
                "default_aggregation": "",
                "join_keys": ["GEOID"] if any(
                    u in {"tract", "county", "place", "zcta", "block_group"}
                    for u in spatial_units
                ) else [],
            }

        # Normalize temporal coverage from flat fields
        if "temporal_coverage" not in payload:
            yr_min = payload.get("coverage_year_min")
            yr_max = payload.get("coverage_year_max")
            if yr_min and yr_max:
                payload["temporal_coverage"] = {
                    "coverage_year_min": int(yr_min),
                    "coverage_year_max": int(yr_max),
                    "cross_sectional_only": payload.get("coverage_year_min") == payload.get("coverage_year_max"),
                }

        return cls.model_validate(payload)
