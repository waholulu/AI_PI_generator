from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
