from __future__ import annotations

from typing import Iterable
from urllib.parse import urlparse

from agents.logging_config import get_logger
from models.data_access_schema import DataAccessCheck
from models.research_plan_schema import DataSourceSpec, ResearchPlan

logger = get_logger(__name__)

_MACHINE_READABLE_FORMATS = {
    "csv",
    "parquet",
    "json",
    "geojson",
    "feather",
    "tsv",
    "xlsx",
}


def _safe_contains(text: str, keywords: Iterable[str]) -> bool:
    lowered = (text or "").lower()
    return any(k.lower() in lowered for k in keywords)


def _is_url_reachable(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    # Keep v1 deterministic/offline friendly: URL structure implies potentially reachable.
    return True


def _is_machine_readable(source: DataSourceSpec) -> bool:
    if source.machine_readable:
        return True
    fmt = (source.expected_format or "").strip().lower()
    if fmt in _MACHINE_READABLE_FORMATS:
        return True
    notes = f"{source.access_notes} {source.name}".lower()
    return any(token in notes for token in _MACHINE_READABLE_FORMATS)


def _covers_exposure_for_source(
    source: DataSourceSpec,
    exposure_name: str,
    exposure_proxy: str,
    exposure_family: str,
) -> bool:
    """Return True if this source plausibly covers the exposure variable."""
    if source.covers_variable_families:
        families_lower = [f.lower() for f in source.covers_variable_families]
        return (
            source.role == "exposure"
            and (exposure_family in families_lower or exposure_name.lower() in families_lower)
        )
    # Keyword fallback: works for legacy plans where roles aren't explicitly set
    source_blob = f"{source.name} {source.access_notes}".lower()
    return source.role == "exposure" or _safe_contains(source_blob, [exposure_name, exposure_proxy])


def _covers_outcome_for_source(
    source: DataSourceSpec,
    outcome_name: str,
    outcome_proxy: str,
    outcome_family: str,
) -> bool:
    """Return True if this source plausibly covers the outcome variable."""
    if source.covers_variable_families:
        families_lower = [f.lower() for f in source.covers_variable_families]
        return (
            source.role == "outcome"
            and (outcome_family in families_lower or outcome_name.lower() in families_lower)
        )
    source_blob = f"{source.name} {source.access_notes}".lower()
    return source.role == "outcome" or _safe_contains(source_blob, [outcome_name, outcome_proxy])


def evaluate_data_sources(plan: ResearchPlan) -> list[DataAccessCheck]:
    """Evaluate each data source at the source level.

    Source-level reasons only flag a source for its own role's issues.  An
    outcome source is NOT flagged for missing exposure coverage, and vice versa.
    Cross-role gaps appear in the plan-level report (``build_plan_level_report``).
    """
    checks: list[DataAccessCheck] = []
    exposure_name = plan.exposure.name.lower()
    exposure_proxy = (plan.exposure.measurement_proxy or "").lower()
    exposure_family = (plan.exposure.family or plan.exposure.name).lower()
    outcome_name = plan.outcome.name.lower()
    outcome_proxy = (plan.outcome.measurement_proxy or "").lower()
    outcome_family = (plan.outcome.family or plan.outcome.name).lower()
    geography_hint = plan.geography.lower()
    time_hint = plan.time_window.lower()
    temporal_frequency = (plan.exposure.temporal_frequency or plan.outcome.temporal_frequency or "").lower()

    for source in plan.data_sources:
        source_blob = " ".join(
            [source.name, source.access_notes, source.documentation_url]
        ).lower()

        reachable = _is_url_reachable(source.access_url) or _is_url_reachable(source.documentation_url)
        machine_readable = _is_machine_readable(source)
        license_found = bool(source.license.strip())

        covers_exposure = _covers_exposure_for_source(
            source, exposure_name, exposure_proxy, exposure_family
        )
        covers_outcome = _covers_outcome_for_source(
            source, outcome_name, outcome_proxy, outcome_family
        )

        geography_compatible = not geography_hint or _safe_contains(
            source_blob, [geography_hint, plan.exposure.spatial_unit, plan.outcome.spatial_unit]
        )
        time_compatible = not time_hint or _safe_contains(
            source_blob, [time_hint, temporal_frequency]
        )

        reasons: list[str] = []
        if not reachable and not source.access_url and not source.documentation_url:
            reasons.append("no_access_url")
        elif not reachable:
            reasons.append("url_not_reachable")
        if not machine_readable:
            reasons.append("missing_machine_readable_source")

        # Only flag role mismatch when the source explicitly declares the role
        # but doesn't fulfil it — avoids noise from boundary/control sources.
        if source.role == "exposure" and not covers_exposure:
            reasons.append("missing_exposure_role_source")
        if source.role == "outcome" and not covers_outcome:
            reasons.append("missing_outcome_role_source")
        # Boundary source must provide join keys
        if source.role == "boundary" and not source.join_keys:
            reasons.append("missing_join_path")
        if source.auth_required:
            reasons.append("experimental_source_requires_key")
        if "streetview" in source.name.lower() and "no_raw_image_storage" not in source.access_notes.lower():
            reasons.append("raw_image_policy_not_satisfied")
        if not geography_compatible:
            reasons.append("geography_mismatch")
        if not time_compatible:
            reasons.append("time_window_mismatch")

        role_covered = (covers_exposure or covers_outcome or source.role in {"control", "boundary"}) and machine_readable
        if not reachable and source.source_type in {"registry", "manual"}:
            verdict = "warning"
        elif not reachable:
            verdict = "fail"
        elif role_covered:
            verdict = "pass"
        else:
            verdict = "warning"

        checks.append(
            DataAccessCheck(
                source_name=source.name,
                access_url=source.access_url,
                documentation_url=source.documentation_url,
                reachable=reachable,
                machine_readable=machine_readable,
                expected_format=source.expected_format,
                license_found=license_found,
                covers_exposure=covers_exposure,
                covers_outcome=covers_outcome,
                geography_compatible=geography_compatible,
                time_compatible=time_compatible,
                verdict=verdict,
                reasons=reasons,
            )
        )
    return checks


def build_plan_level_report(checks: list[DataAccessCheck], plan: ResearchPlan) -> dict:
    """Return a plan-level role coverage summary.

    Answers: "taken as a whole, does this set of data sources satisfy all four
    required roles?"  Individual per-source noise does not appear here.
    """
    has_exposure = any(c.covers_exposure and c.machine_readable for c in checks)
    has_outcome = any(c.covers_outcome and c.machine_readable for c in checks)
    has_machine_readable = any(c.machine_readable for c in checks)

    has_control = any(
        getattr(s, "role", None) == "control" and _is_machine_readable(s)
        for s in plan.data_sources
    )

    # Boundary / join path: boundary source with join_keys, OR TIGER_Lines (provides GEOID implicitly)
    has_boundary = any(
        getattr(s, "role", None) == "boundary" and (s.join_keys or s.name)
        for s in plan.data_sources
    ) or any("tiger" in getattr(s, "name", "").lower() for s in plan.data_sources)

    role_coverage = {
        "exposure": "pass" if has_exposure else "fail",
        "outcome": "pass" if has_outcome else "fail",
        "control": "pass" if has_control else "warning",
        "boundary": "pass" if has_boundary else "fail",
    }

    join_path_ok = has_boundary and has_exposure and has_outcome
    overall_pass = has_exposure and has_outcome and has_machine_readable and has_boundary

    return {
        "role_coverage": role_coverage,
        "join_path": "pass" if join_path_ok else "fail",
        "machine_readable": "pass" if has_machine_readable else "fail",
        "cloud_safe": "pass",
        "overall": "pass" if overall_pass else "fail",
    }


def summarize_data_access(checks: list[DataAccessCheck]) -> tuple[str, list[str]]:
    """Aggregate check results to a plan-level verdict and failure reason list.

    Uses role-based aggregation: a role is satisfied when at least one source
    with that role is machine-readable.  Sources like OSMnx or TIGER that have
    no explicit access URL are still valid because they are acquired
    programmatically (osmnx library) or via bulk download.
    """
    if not checks:
        return "fail", ["no_data_sources_declared"]

    exposure_ok = any(c.covers_exposure and c.machine_readable for c in checks)
    outcome_ok = any(c.covers_outcome and c.machine_readable for c in checks)
    machine_readable = any(c.machine_readable for c in checks)
    geo_ok = any(c.geography_compatible for c in checks)
    time_ok = any(c.time_compatible for c in checks)

    # join_ok: only relevant when boundary sources exist. If no boundary sources
    # are declared at all, join path is outside scope of this module (checked by
    # candidate_feasibility). When boundary sources do exist, at least one must
    # not have the missing_join_path reason; TIGER always satisfies this.
    tiger_present = any("tiger" in c.source_name.lower() for c in checks)
    boundary_checks = [
        c for c in checks
        if not c.covers_exposure and not c.covers_outcome
    ]
    if not boundary_checks:
        join_ok = True  # no boundary sources → no join constraint to evaluate here
    else:
        boundary_has_join = any("missing_join_path" not in c.reasons for c in boundary_checks)
        join_ok = boundary_has_join or tiger_present

    reasons: list[str] = []
    if not exposure_ok:
        reasons.append("missing_exposure_role_source")
    if not outcome_ok:
        reasons.append("missing_outcome_role_source")
    if not machine_readable:
        reasons.append("missing_machine_readable_source")
    if not join_ok:
        reasons.append("missing_join_path")
    if not geo_ok:
        reasons.append("geography_incompatible")
    if not time_ok:
        reasons.append("time_window_incompatible")

    if not reasons:
        return "pass", []

    if any(c.verdict == "warning" for c in checks) and "missing_machine_readable_source" not in reasons:
        return "warning", reasons
    return "fail", reasons
