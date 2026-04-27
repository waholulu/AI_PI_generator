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


def validate_template_role_compatibility(
    template: dict[str, Any],
    registry: SourceRegistry | None = None,
) -> list[str]:
    """Check that every preferred source in the template carries the required role.

    Returns a list of error strings describing mismatches (empty = no problems).
    This is a config-layer check: call it once at startup or in CI, not per-candidate.

    Rules:
      allowed_exposure_families.*.preferred_sources  must each have "exposure" role
      allowed_outcome_families.*.preferred_sources   must each have "outcome" role
      default_controls                               must each have "control" role
      default_boundary_source                        must each have "boundary" role
    """
    registry = registry or SourceRegistry.load()
    errors: list[str] = []

    for family, spec in (template.get("allowed_exposure_families") or {}).items():
        for src in (spec.get("preferred_sources") or []):
            sid = registry.resolve(src)
            if not sid:
                errors.append(f"exposure_family={family}: source_not_in_registry:{src}")
                continue
            source_spec = registry.sources.get(sid, {})
            if "exposure" not in (source_spec.get("roles") or []):
                errors.append(
                    f"exposure_family={family}: source_missing_exposure_role:{src}"
                )

    for family, spec in (template.get("allowed_outcome_families") or {}).items():
        for src in (spec.get("preferred_sources") or []):
            sid = registry.resolve(src)
            if not sid:
                errors.append(f"outcome_family={family}: source_not_in_registry:{src}")
                continue
            source_spec = registry.sources.get(sid, {})
            if "outcome" not in (source_spec.get("roles") or []):
                errors.append(
                    f"outcome_family={family}: source_missing_outcome_role:{src}"
                )

    for src in (template.get("default_controls") or []):
        sid = registry.resolve(src)
        if not sid:
            errors.append(f"default_controls: source_not_in_registry:{src}")
            continue
        source_spec = registry.sources.get(sid, {})
        if "control" not in (source_spec.get("roles") or []):
            errors.append(f"default_controls: source_missing_control_role:{src}")

    for src in (template.get("default_boundary_source") or []):
        sid = registry.resolve(src)
        if not sid:
            errors.append(f"default_boundary_source: source_not_in_registry:{src}")
            continue
        source_spec = registry.sources.get(sid, {})
        if "boundary" not in (source_spec.get("roles") or []):
            errors.append(f"default_boundary_source: source_missing_boundary_role:{src}")

    return errors
