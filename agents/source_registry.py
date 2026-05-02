from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from models.data_catalog import DataSourceProfile
from models.research_plan_schema import DataSourceSpec


@dataclass
class SourceRegistry:
    """Lightweight accessor for canonical source metadata and aliases.

    Load priority:
      1. config/data_catalog/sources/*.yaml  (rich DataSourceProfile format)
      2. config/source_capabilities.yaml      (flat legacy format, fallback)

    Both formats are normalised into the same flat ``sources`` dict so that
    all legacy callers continue to work unchanged.  The enhanced profile is
    accessible via ``get_profile()``.
    """

    sources: dict[str, dict[str, Any]]
    alias_to_id: dict[str, str]
    _profiles: dict[str, DataSourceProfile] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path = "config/source_capabilities.yaml") -> "SourceRegistry":
        raw_sources: dict[str, dict[str, Any]] = {}
        profiles: dict[str, DataSourceProfile] = {}

        # ── 1. Load rich data catalog profiles (preferred) ───────────────────
        catalog_dir = Path("config/data_catalog/sources")
        if catalog_dir.exists():
            for yaml_file in sorted(catalog_dir.glob("*.yaml")):
                try:
                    data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                source_id = data.get("source_id") or yaml_file.stem
                profile = DataSourceProfile.from_dict(source_id, data)
                profiles[source_id] = profile
                # Build flat dict from profile for backward-compat callers
                raw_sources[source_id] = _profile_to_flat(profile, data)

        # ── 2. Fall back to source_capabilities.yaml for remaining sources ────
        fallback_path = Path(path)
        if fallback_path.exists():
            payload = yaml.safe_load(fallback_path.read_text(encoding="utf-8")) or {}
            for source_id, spec in (payload.get("sources", {}) or {}).items():
                if source_id not in raw_sources:
                    raw_sources[source_id] = spec or {}

        # ── 3. Build alias map ────────────────────────────────────────────────
        alias_to_id: dict[str, str] = {}
        for source_id, spec in raw_sources.items():
            alias_to_id[source_id.lower()] = source_id
            canonical = str(spec.get("canonical_name", "")).strip()
            if canonical:
                alias_to_id[canonical.lower()] = source_id
            for alias in spec.get("aliases", []) or []:
                alias_to_id[str(alias).strip().lower()] = source_id

        registry = cls(sources=raw_sources, alias_to_id=alias_to_id)
        registry._profiles = profiles
        return registry

    # ── Existing public API (unchanged) ──────────────────────────────────────

    def resolve(self, name_or_alias: str) -> str | None:
        return self.alias_to_id.get(name_or_alias.strip().lower())

    def get(self, name_or_alias: str) -> dict[str, Any] | None:
        source_id = self.resolve(name_or_alias)
        if not source_id:
            return None
        return self.sources.get(source_id)

    def list_variable_families(self, role: str | None = None) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        for source_id, spec in self.sources.items():
            roles = set(spec.get("roles", []) or [])
            if role and role not in roles:
                continue
            families = list((spec.get("variable_families") or {}).keys())
            output[source_id] = families
        return output

    def get_sources_by_role(self, role: str) -> list[str]:
        return [sid for sid, spec in self.sources.items() if role in (spec.get("roles", []) or [])]

    def get_sources_by_variable_family(self, family: str) -> list[str]:
        matches: list[str] = []
        for sid, spec in self.sources.items():
            families = spec.get("variable_families") or {}
            if family in families:
                matches.append(sid)
        return matches

    def is_cloud_safe(self, source_name: str) -> bool:
        spec = self.get(source_name) or {}
        if "cloud_safe" in spec:
            return bool(spec["cloud_safe"])
        return not bool(spec.get("cost_required") or spec.get("auth_required"))

    def requires_secret(self, source_name: str) -> bool:
        spec = self.get(source_name) or {}
        return bool(spec.get("auth_required") or spec.get("cost_required"))

    def get_machine_readable_sources(self) -> list[str]:
        return [sid for sid, spec in self.sources.items() if bool(spec.get("machine_readable"))]

    def get_source_tier(self, source_name: str) -> str:
        spec = self.get(source_name) or {}
        return str(spec.get("tier", "stable"))

    def enrich_data_source_from_registry(
        self,
        source_id: str,
        role: str | None = None,
        variable_family: str | None = None,
    ) -> DataSourceSpec:
        canonical_id = self.resolve(source_id) or source_id
        spec = self.sources.get(canonical_id, {})
        roles = list(spec.get("roles", []) or [])

        resolved_role = role
        if not resolved_role:
            if len(roles) == 1:
                resolved_role = roles[0]
            elif "control" in roles:
                resolved_role = "control"
            else:
                resolved_role = roles[0] if roles else "control"

        family_keys = list((spec.get("variable_families") or {}).keys())
        if variable_family and variable_family not in family_keys:
            family_keys.append(variable_family)

        join_keys: list[str] = []
        spatial_units = list(spec.get("spatial_units", []) or [])
        if any(u in {"tract", "county", "place", "zcta", "block_group"} for u in spatial_units):
            join_keys = ["GEOID"]

        source_type = str(spec.get("source_type", "unknown"))
        if source_type not in {"api", "download", "registry", "manual", "unknown"}:
            if source_type.endswith("_download"):
                source_type = "download"
            else:
                source_type = "unknown"

        return DataSourceSpec(
            name=canonical_id,
            role=resolved_role,  # type: ignore[arg-type]
            source_type=source_type,  # type: ignore[arg-type]
            access_url=str(spec.get("access_url", "")),
            documentation_url=str(spec.get("documentation_url", "")),
            expected_format=str(spec.get("expected_format", "")),
            machine_readable=bool(spec.get("machine_readable", False)),
            auth_required=bool(spec.get("auth_required", False)),
            cost_required=bool(spec.get("cost_required", False)),
            covers_variable_families=family_keys,
            spatial_units=spatial_units,
            technology_tags=list(spec.get("technology_tags", []) or []),
            join_keys=join_keys,
        )

    # ── New data-catalog-aware API ────────────────────────────────────────────

    def get_profile(self, name_or_alias: str) -> DataSourceProfile | None:
        """Return the rich DataSourceProfile, or None if only flat data is available."""
        source_id = self.resolve(name_or_alias)
        if not source_id:
            return None
        return self._profiles.get(source_id)

    def get_native_unit(self, name_or_alias: str) -> str:
        """Return the native spatial unit, or '' if unknown."""
        profile = self.get_profile(name_or_alias)
        if profile and profile.geography:
            return profile.geography.native_unit
        spec = self.get(name_or_alias) or {}
        units = spec.get("spatial_units") or []
        return units[0] if units else ""

    def get_target_units(self, name_or_alias: str) -> list[str]:
        """Return supported target spatial units."""
        profile = self.get_profile(name_or_alias)
        if profile and profile.geography:
            return list(profile.geography.target_units_supported)
        spec = self.get(name_or_alias) or {}
        # Fall back to aggregation_allowed_to inside variable_families
        for fam in (spec.get("variable_families") or {}).values():
            if isinstance(fam, dict):
                agg = fam.get("aggregation_allowed_to")
                if agg:
                    return list(agg)
        return list(spec.get("spatial_units") or [])

    def get_join_recipes(self, name_or_alias: str) -> list[dict]:
        """Return join recipes for this source (from data catalog profile)."""
        profile = self.get_profile(name_or_alias)
        if profile:
            return [r.model_dump() for r in profile.join_recipes]
        return []

    def has_variable_mapping(self, name_or_alias: str, family: str) -> bool:
        """Return True if the source has concrete variable names for the given family."""
        profile = self.get_profile(name_or_alias)
        if profile:
            return profile.has_variable_mapping_for(family)
        spec = self.get(name_or_alias) or {}
        vf = (spec.get("variable_families") or {}).get(family)
        if not vf:
            return False
        if isinstance(vf, dict):
            return bool(vf.get("variables"))
        return False

    def aggregation_required(self, name_or_alias: str) -> bool:
        """Return True if the source requires aggregation from native to analysis grain."""
        profile = self.get_profile(name_or_alias)
        if profile and profile.geography:
            return profile.geography.aggregation_required
        return False

    def get_default_aggregation_method(self, name_or_alias: str) -> str:
        """Return the recommended aggregation method, or ''."""
        profile = self.get_profile(name_or_alias)
        if profile and profile.geography:
            return profile.geography.default_aggregation
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _profile_to_flat(profile: DataSourceProfile, raw: dict) -> dict:
    """Merge DataSourceProfile back into a flat dict for backward-compat callers."""
    flat: dict[str, Any] = dict(raw)
    # Ensure all legacy fields are populated
    flat["canonical_name"] = profile.canonical_name
    flat["aliases"] = profile.aliases
    flat["roles"] = profile.roles
    flat["tier"] = profile.tier
    flat["cloud_safe"] = profile.cloud_safe
    flat["machine_readable"] = profile.machine_readable
    flat["auth_required"] = profile.acquisition.auth_required
    flat["cost_required"] = profile.acquisition.cost_required
    flat["source_type"] = profile.acquisition.method
    flat["expected_format"] = profile.acquisition.expected_format
    flat["access_url"] = profile.acquisition.url
    flat["documentation_url"] = profile.acquisition.documentation_url

    if profile.geography:
        # spatial_units only lists the native unit so that _check_aggregation
        # correctly detects when the target unit differs from the native grain.
        # aggregation_allowed_to inside variable_families guides the upgrade path.
        flat["spatial_units"] = [profile.geography.native_unit]

    if profile.temporal_coverage:
        flat["coverage_year_min"] = profile.temporal_coverage.coverage_year_min
        flat["coverage_year_max"] = profile.temporal_coverage.coverage_year_max

    # Keep variable_families in a format compatible with both old and new callers
    if profile.variable_families:
        flat["variable_families"] = profile.variable_families

    return flat
