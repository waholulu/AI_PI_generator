"""Tests for eval_candidate_factory.py — threshold checks and golden candidates.

Runs without API keys (no live OpenAlex or LLM calls).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_candidate_factory import _check_thresholds, evaluate


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_eval(enable_experimental: bool = False, max_candidates: int = 40) -> dict:
    return evaluate(
        template_id="built_environment_health",
        domain="Built environment and health outcomes",
        max_candidates=max_candidates,
        enable_experimental=enable_experimental,
    )


# ---------------------------------------------------------------------------
# Threshold tests (v1 thresholds)
# ---------------------------------------------------------------------------

def test_candidate_count_at_least_20():
    result = _run_eval()
    assert result["candidate_count"] >= 20, (
        f"candidate_count {result['candidate_count']} < 20"
    )


def test_score_completion_rate():
    result = _run_eval()
    assert result["score_completion_rate"] >= 0.95, (
        f"score_completion_rate {result['score_completion_rate']:.2%} < 95%"
    )


def test_implementation_spec_completion_rate():
    result = _run_eval()
    assert result["implementation_spec_completion_rate"] >= 0.85, (
        f"implementation_spec_completion_rate "
        f"{result['implementation_spec_completion_rate']:.2%} < 85%"
    )


def test_development_pack_ready_rate():
    result = _run_eval()
    assert result["development_pack_ready_rate"] >= 0.80, (
        f"development_pack_ready_rate {result['development_pack_ready_rate']:.2%} < 80%"
    )


def test_low_or_medium_automation_risk_rate():
    result = _run_eval()
    assert result["low_or_medium_automation_risk_rate"] >= 0.70, (
        f"low_or_medium_automation_risk_rate "
        f"{result['low_or_medium_automation_risk_rate']:.2%} < 70%"
    )


def test_no_experimental_candidates_when_disabled():
    result = _run_eval(enable_experimental=False)
    assert result["experimental_candidate_count"] == 0, (
        f"experimental_candidate_count {result['experimental_candidate_count']} != 0 "
        "when experimental disabled"
    )


def test_top5_candidate_ids_populated():
    result = _run_eval()
    assert len(result["top_5_candidate_ids"]) >= 1


def test_result_has_all_required_keys():
    result = _run_eval()
    required_keys = {
        "template_id",
        "candidate_count",
        "candidate_pass_rate",
        "score_completion_rate",
        "implementation_spec_completion_rate",
        "development_pack_ready_rate",
        "low_or_medium_automation_risk_rate",
        "experimental_candidate_count",
        "ready_candidate_count",
        "blocked_candidate_count",
        "top_5_candidate_ids",
    }
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# _check_thresholds helper
# ---------------------------------------------------------------------------

def test_check_thresholds_passes_on_good_result():
    good = {
        "candidate_count": 25,
        "score_completion_rate": 1.0,
        "implementation_spec_completion_rate": 0.92,
        "development_pack_ready_rate": 0.85,
        "low_or_medium_automation_risk_rate": 0.88,
        "experimental_candidate_count": 0,
        "enable_experimental": False,
    }
    assert _check_thresholds(good) == []


def test_check_thresholds_catches_low_candidate_count():
    bad = {
        "candidate_count": 5,
        "score_completion_rate": 1.0,
        "implementation_spec_completion_rate": 0.92,
        "development_pack_ready_rate": 0.85,
        "low_or_medium_automation_risk_rate": 0.88,
        "experimental_candidate_count": 0,
        "enable_experimental": False,
    }
    failures = _check_thresholds(bad)
    assert any("candidate_count" in f for f in failures)


def test_check_thresholds_catches_experimental_leak():
    leaky = {
        "candidate_count": 25,
        "score_completion_rate": 1.0,
        "implementation_spec_completion_rate": 0.92,
        "development_pack_ready_rate": 0.85,
        "low_or_medium_automation_risk_rate": 0.88,
        "experimental_candidate_count": 3,
        "enable_experimental": False,
    }
    failures = _check_thresholds(leaky)
    assert any("experimental" in f for f in failures)


# ---------------------------------------------------------------------------
# Golden candidate checks
# ---------------------------------------------------------------------------

_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden_candidates" / "built_environment_health"

_GOLDEN_IDS = [
    "osmnx_connectivity_physical_inactivity",
    "osmnx_intersection_density_obesity",
    "epa_walkability_physical_inactivity",
    "nlcd_green_space_mental_health",
    "nlcd_impervious_asthma",
    "enviroatlas_tree_canopy_asthma",
    "gtfs_transit_access_physical_inactivity",
    "microsoft_building_density_physical_inactivity",
]


@pytest.mark.parametrize("golden_name", _GOLDEN_IDS)
def test_golden_candidate_file_exists(golden_name: str):
    path = _GOLDEN_DIR / f"{golden_name}.json"
    assert path.exists(), f"Golden candidate file missing: {path}"


@pytest.mark.parametrize("golden_name", _GOLDEN_IDS)
def test_golden_candidate_required_fields(golden_name: str):
    path = _GOLDEN_DIR / f"{golden_name}.json"
    if not path.exists():
        pytest.skip(f"File not found: {path}")
    candidate = json.loads(path.read_text(encoding="utf-8"))

    required = [
        "candidate_id", "template_id", "exposure_family", "exposure_source",
        "outcome_family", "outcome_source", "unit_of_analysis",
        "method_template", "key_threats", "mitigations",
        "technology_tags", "automation_risk", "cloud_safe",
    ]
    for field in required:
        assert field in candidate, f"Missing field '{field}' in {golden_name}"


@pytest.mark.parametrize("golden_name", _GOLDEN_IDS)
def test_golden_candidate_source_role_check(golden_name: str):
    from agents.source_registry import SourceRegistry

    path = _GOLDEN_DIR / f"{golden_name}.json"
    if not path.exists():
        pytest.skip(f"File not found: {path}")
    candidate = json.loads(path.read_text(encoding="utf-8"))

    registry = SourceRegistry.load()
    exp_sid = registry.resolve(candidate["exposure_source"])
    out_sid = registry.resolve(candidate["outcome_source"])

    assert exp_sid is not None, f"Exposure source {candidate['exposure_source']!r} not in registry"
    assert out_sid is not None, f"Outcome source {candidate['outcome_source']!r} not in registry"

    exp_spec = registry.sources.get(exp_sid, {})
    out_spec = registry.sources.get(out_sid, {})

    assert "exposure" in exp_spec.get("roles", []), (
        f"Exposure source {candidate['exposure_source']!r} missing 'exposure' role"
    )
    assert "outcome" in out_spec.get("roles", []), (
        f"Outcome source {candidate['outcome_source']!r} missing 'outcome' role"
    )


@pytest.mark.parametrize("golden_name", _GOLDEN_IDS)
def test_golden_candidate_machine_readable(golden_name: str):
    from agents.source_registry import SourceRegistry

    path = _GOLDEN_DIR / f"{golden_name}.json"
    if not path.exists():
        pytest.skip(f"File not found: {path}")
    candidate = json.loads(path.read_text(encoding="utf-8"))

    registry = SourceRegistry.load()
    for src_name in [candidate["exposure_source"], candidate["outcome_source"]]:
        sid = registry.resolve(src_name)
        if sid:
            spec = registry.sources.get(sid, {})
            assert spec.get("machine_readable", False), (
                f"Source {src_name!r} is not machine-readable"
            )


@pytest.mark.parametrize("golden_name", _GOLDEN_IDS)
def test_golden_candidate_threat_mitigation_coverage(golden_name: str):
    path = _GOLDEN_DIR / f"{golden_name}.json"
    if not path.exists():
        pytest.skip(f"File not found: {path}")
    candidate = json.loads(path.read_text(encoding="utf-8"))

    threats = set(candidate.get("key_threats", []))
    mitigations = set(candidate.get("mitigations", {}).keys())

    assert len(threats) >= 3, f"Too few threats in {golden_name}: {threats}"
    coverage = len(threats & mitigations) / len(threats) if threats else 0
    assert coverage >= 0.8, (
        f"Threat mitigation coverage {coverage:.0%} < 80% in {golden_name}"
    )


@pytest.mark.parametrize("golden_name", _GOLDEN_IDS)
def test_golden_candidate_automation_risk_not_high(golden_name: str):
    """Golden candidates should all be low or medium automation risk."""
    path = _GOLDEN_DIR / f"{golden_name}.json"
    if not path.exists():
        pytest.skip(f"File not found: {path}")
    candidate = json.loads(path.read_text(encoding="utf-8"))
    assert candidate["automation_risk"] in {"low", "medium"}, (
        f"Golden candidate {golden_name} has high automation risk"
    )


@pytest.mark.parametrize("golden_name", _GOLDEN_IDS)
def test_golden_candidate_cloud_safe(golden_name: str):
    path = _GOLDEN_DIR / f"{golden_name}.json"
    if not path.exists():
        pytest.skip(f"File not found: {path}")
    candidate = json.loads(path.read_text(encoding="utf-8"))
    assert candidate["cloud_safe"] is True, f"Golden candidate {golden_name} not cloud_safe"
