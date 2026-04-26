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


def evaluate_data_sources(plan: ResearchPlan) -> list[DataAccessCheck]:
    checks: list[DataAccessCheck] = []
    exposure_name = plan.exposure.name.lower()
    outcome_name = plan.outcome.name.lower()
    geography_hint = plan.geography.lower()
    time_hint = plan.time_window.lower()
    temporal_frequency = plan.exposure.temporal_frequency.lower() or plan.outcome.temporal_frequency.lower()

    exposure_family = (plan.exposure.family or plan.exposure.name).lower()
    outcome_family = (plan.outcome.family or plan.outcome.name).lower()

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
        if source.covers_variable_families:
            families_lower = [f.lower() for f in source.covers_variable_families]
            covers_exposure = source.role == "exposure" and (
                exposure_family in families_lower
                or exposure_name in families_lower
            )
            covers_outcome = source.role == "outcome" and (
                outcome_family in families_lower
                or outcome_name in families_lower
            )
        else:
            covers_exposure = (
                source.role == "exposure"
                or _safe_contains(source_blob, [exposure_name, plan.exposure.measurement_proxy])
            )
            covers_outcome = (
                source.role == "outcome"
                or _safe_contains(source_blob, [outcome_name, plan.outcome.measurement_proxy])
            )
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
            reasons.append("missing_machine_readable_source")
        if not covers_exposure:
            reasons.append("missing_exposure_role_source")
        if not covers_outcome:
            reasons.append("missing_outcome_role_source")
        if source.role == "boundary" and not source.join_keys:
            reasons.append("missing_join_path")
        if source.auth_required:
            reasons.append("experimental_source_requires_key")
        if "streetview" in source.name.lower() and "no_raw_image_storage" not in source.access_notes.lower():
            reasons.append("streetview_policy_not_satisfied")
        if not geography_compatible:
            reasons.append("geography_mismatch")
        if not time_compatible:
            reasons.append("time_window_mismatch")

        if not reachable and source.source_type in {"registry", "manual"}:
            verdict = "warning"
        elif not reachable:
            verdict = "fail"
        elif (covers_exposure or covers_outcome or source.role in {"control", "boundary"}) and machine_readable:
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
    join_ok = any("missing_join_path" not in c.reasons for c in checks)
    geo_ok = any(c.geography_compatible for c in checks)
    time_ok = any(c.time_compatible for c in checks)

    reasons: list[str] = []
    if not exposure_reachable:
        reasons.append("missing_exposure_role_source")
    if not outcome_reachable:
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

    if all(r in {"missing_exposure_role_source", "missing_outcome_role_source"} for r in reasons):
        if any(c.verdict == "warning" for c in checks):
            return "warning", reasons
    if any(c.verdict == "warning" for c in checks) and "missing_machine_readable_source" not in reasons:
        return "warning", reasons
    return "fail", reasons
