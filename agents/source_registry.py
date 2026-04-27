from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from models.research_plan_schema import DataSourceSpec


@dataclass
class SourceRegistry:
    """Lightweight accessor for canonical source metadata and aliases."""

    sources: dict[str, dict[str, Any]]
    alias_to_id: dict[str, str]

    @classmethod
    def load(cls, path: str | Path = "config/source_capabilities.yaml") -> "SourceRegistry":
        p = Path(path)
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw_sources = payload.get("sources", {})

        alias_to_id: dict[str, str] = {}
        for source_id, spec in raw_sources.items():
            alias_to_id[source_id.lower()] = source_id
            canonical = str(spec.get("canonical_name", "")).strip()
            if canonical:
                alias_to_id[canonical.lower()] = source_id
            for alias in spec.get("aliases", []) or []:
                alias_to_id[str(alias).strip().lower()] = source_id

        return cls(sources=raw_sources, alias_to_id=alias_to_id)

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
