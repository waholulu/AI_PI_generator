"""Tests for validate_template_role_compatibility().

Verifies that the built_environment_health template passes the role check,
and that templates with misconfigured sources are correctly flagged.
"""
from __future__ import annotations

import pytest

from agents.research_template_loader import (
    load_research_template,
    validate_template_role_compatibility,
)
from agents.source_registry import SourceRegistry


# ── happy path ────────────────────────────────────────────────────────────────

def test_built_environment_health_passes_role_check() -> None:
    """All preferred sources in the production template must have the correct role."""
    template = load_research_template("built_environment_health")
    errors = validate_template_role_compatibility(template)
    assert errors == [], f"Role compatibility errors found:\n" + "\n".join(errors)


# ── exposure role mismatch ────────────────────────────────────────────────────

def test_outcome_source_as_exposure_flagged() -> None:
    """CDC_PLACES has only the 'outcome' role; using it as exposure must produce an error."""
    template = {
        "template_id": "test_bad_exposure",
        "allowed_exposure_families": {
            "bad_family": {
                "preferred_sources": ["CDC_PLACES"],
            }
        },
        "allowed_outcome_families": {},
    }
    errors = validate_template_role_compatibility(template)
    assert any("source_missing_exposure_role" in e for e in errors), (
        f"Expected missing_exposure_role error; got: {errors}"
    )
    assert any("bad_family" in e for e in errors)


# ── outcome role mismatch ─────────────────────────────────────────────────────

def test_exposure_source_as_outcome_flagged() -> None:
    """OSMnx has only 'exposure' role; using it as outcome must produce an error."""
    template = {
        "template_id": "test_bad_outcome",
        "allowed_exposure_families": {},
        "allowed_outcome_families": {
            "bad_outcome": {
                "preferred_sources": ["OSMnx_OpenStreetMap"],
            }
        },
    }
    errors = validate_template_role_compatibility(template)
    assert any("source_missing_outcome_role" in e for e in errors), (
        f"Expected missing_outcome_role error; got: {errors}"
    )


# ── default_controls mismatch ─────────────────────────────────────────────────

def test_outcome_source_as_control_flagged() -> None:
    """CDC_PLACES has no 'control' role; using it in default_controls must be flagged."""
    template = {
        "template_id": "test_bad_control",
        "allowed_exposure_families": {},
        "allowed_outcome_families": {},
        "default_controls": ["CDC_PLACES"],
    }
    errors = validate_template_role_compatibility(template)
    assert any("source_missing_control_role" in e for e in errors), (
        f"Expected missing_control_role error; got: {errors}"
    )


# ── default_boundary_source mismatch ─────────────────────────────────────────

def test_non_boundary_source_as_boundary_flagged() -> None:
    """ACS has 'control' but not 'boundary' role; using it as boundary must be flagged."""
    template = {
        "template_id": "test_bad_boundary",
        "allowed_exposure_families": {},
        "allowed_outcome_families": {},
        "default_boundary_source": ["ACS"],
    }
    errors = validate_template_role_compatibility(template)
    assert any("source_missing_boundary_role" in e for e in errors), (
        f"Expected missing_boundary_role error; got: {errors}"
    )


# ── unknown source ────────────────────────────────────────────────────────────

def test_unknown_source_flagged() -> None:
    """A source name that does not exist in the registry must produce an error."""
    template = {
        "template_id": "test_unknown",
        "allowed_exposure_families": {
            "some_family": {
                "preferred_sources": ["Nonexistent_Source_XYZ"],
            }
        },
        "allowed_outcome_families": {},
    }
    errors = validate_template_role_compatibility(template)
    assert any("source_not_in_registry" in e for e in errors), (
        f"Expected source_not_in_registry error; got: {errors}"
    )


# ── TIGER_Lines in default_boundary_source passes ────────────────────────────

def test_tiger_lines_as_boundary_passes() -> None:
    """TIGER_Lines has the 'boundary' role and must pass the boundary check."""
    template = {
        "template_id": "test_tiger_boundary",
        "allowed_exposure_families": {},
        "allowed_outcome_families": {},
        "default_boundary_source": ["TIGER_Lines"],
    }
    errors = validate_template_role_compatibility(template)
    boundary_errors = [e for e in errors if "boundary" in e]
    assert boundary_errors == [], f"Unexpected boundary errors: {boundary_errors}"


# ── empty template ────────────────────────────────────────────────────────────

def test_empty_template_passes() -> None:
    """A template with no families or defaults produces no errors."""
    template = {"template_id": "empty_template"}
    errors = validate_template_role_compatibility(template)
    assert errors == []
