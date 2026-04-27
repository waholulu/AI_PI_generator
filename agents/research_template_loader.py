from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agents.source_registry import SourceRegistry


class TemplateValidationError(ValueError):
    """Raised when a research template references unknown sources."""


def load_research_template(
    template_id: str,
    template_dir: str | Path = "config/research_templates",
) -> dict[str, Any]:
    path = Path(template_dir) / f"{template_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")

    template = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not template.get("template_id"):
        raise TemplateValidationError(f"Template missing template_id: {path}")
    return template


def validate_template_sources(
    template: dict[str, Any],
    registry: SourceRegistry | None = None,
) -> list[str]:
    registry = registry or SourceRegistry.load()
    missing: list[str] = []

    def _check_source(source_name: str) -> None:
        if registry.resolve(source_name) is None and source_name not in missing:
            missing.append(source_name)

    for exposure_spec in (template.get("allowed_exposure_families") or {}).values():
        for source in exposure_spec.get("preferred_sources", []) or []:
            _check_source(source)

    for outcome_spec in (template.get("allowed_outcome_families") or {}).values():
        for source in outcome_spec.get("preferred_sources", []) or []:
            _check_source(source)

    for source in template.get("default_controls", []) or []:
        _check_source(source)
    for source in template.get("default_boundary_source", []) or []:
        _check_source(source)

    return missing
