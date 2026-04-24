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
    fmt = (source.expected_format or "").strip().lower()
    if fmt in _MACHINE_READABLE_FORMATS:
        return True
    notes = f"{source.access_notes} {source.name}".lower()
    return any(token in notes for token in _MACHINE_READABLE_FORMATS)


def evaluate_data_sources(plan: ResearchPlan) -> list[DataAccessCheck]:
    checks: list[DataAccessCheck] = []
    exposure_name = plan.exposure.name.lower()
    outcome_name = plan.outcome.name.lower()
    geography_hint = plan.geography.lower()
    time_hint = plan.time_window.lower()
    temporal_frequency = plan.exposure.temporal_frequency.lower() or plan.outcome.temporal_frequency.lower()

    for source in plan.data_sources:
        source_blob = " ".join(
            [
                source.name,
                source.access_notes,
                source.documentation_url,
            ]
        ).lower()
        reachable = _is_url_reachable(source.access_url) or _is_url_reachable(source.documentation_url)
        machine_readable = _is_machine_readable(source)
        license_found = bool(source.license.strip())
        covers_exposure = _safe_contains(source_blob, [exposure_name, plan.exposure.measurement_proxy])
        covers_outcome = _safe_contains(source_blob, [outcome_name, plan.outcome.measurement_proxy])
        geography_compatible = not geography_hint or _safe_contains(
            source_blob, [geography_hint, plan.exposure.spatial_unit, plan.outcome.spatial_unit]
        )
        time_compatible = not time_hint or _safe_contains(source_blob, [time_hint, temporal_frequency])

        reasons: list[str] = []
        if not reachable and not source.access_url and not source.documentation_url:
            reasons.append("no_access_url")
        elif not reachable:
            reasons.append("url_not_reachable")
        if not machine_readable:
            reasons.append("format_unclear")
        if not covers_exposure:
            reasons.append("exposure_coverage_uncertain")
        if not covers_outcome:
            reasons.append("outcome_coverage_uncertain")
        if not geography_compatible:
            reasons.append("geography_mismatch")
        if not time_compatible:
            reasons.append("time_window_mismatch")

        if not reachable and source.source_type in {"registry", "manual"}:
            verdict = "warning"
        elif not reachable:
            verdict = "fail"
        elif (covers_exposure or covers_outcome) and machine_readable:
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


def summarize_data_access(checks: list[DataAccessCheck]) -> tuple[str, list[str]]:
    if not checks:
        return "fail", ["no_data_sources_declared"]

    exposure_reachable = any(c.reachable and c.covers_exposure for c in checks)
    outcome_reachable = any(c.reachable and c.covers_outcome for c in checks)
    machine_readable = any(c.machine_readable for c in checks)
    geo_ok = any(c.geography_compatible for c in checks)
    time_ok = any(c.time_compatible for c in checks)

    reasons: list[str] = []
    if not exposure_reachable:
        reasons.append("no_reachable_exposure_source")
    if not outcome_reachable:
        reasons.append("no_reachable_outcome_source")
    if not machine_readable:
        reasons.append("no_machine_readable_source")
    if not geo_ok:
        reasons.append("geography_incompatible")
    if not time_ok:
        reasons.append("time_window_incompatible")

    if not reasons:
        return "pass", []

    if all(r in {"no_reachable_exposure_source", "no_reachable_outcome_source"} for r in reasons):
        if any(c.verdict == "warning" for c in checks):
            return "warning", reasons
    if any(c.verdict == "warning" for c in checks) and "no_machine_readable_source" not in reasons:
        return "warning", reasons
    return "fail", reasons
