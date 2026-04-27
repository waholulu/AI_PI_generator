from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.candidate_export_validator import validate_candidate_export_contract
from agents.candidate_feasibility import precheck_candidate
from agents.candidate_repair import repair_candidate
from agents.source_registry import SourceRegistry
from models.candidate_composer_schema import ComposedCandidate


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "bad_candidates"


def _load_candidate(name: str) -> ComposedCandidate:
    data = json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return ComposedCandidate(**data)


@pytest.mark.parametrize(
    "fixture,expected_reason",
    [
        ("missing_exposure_role_source.json", "missing_exposure_role_source"),
        ("missing_outcome_role_source.json", "missing_outcome_role_source"),
        ("missing_identification_threats.json", "missing_identification_threats"),
    ],
)
def test_precheck_identifies_canonical_reasons(fixture: str, expected_reason: str) -> None:
    candidate = _load_candidate(fixture)
    gate = precheck_candidate(candidate)
    assert expected_reason in gate["reasons"]
    assert "partial_registry_match" not in gate["reasons"]


def test_export_contract_blocks_non_machine_readable_core_source(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _load_candidate("missing_machine_readable_source.json")

    original = SourceRegistry.load()
    patched_sources = dict(original.sources)
    patched_sources["FAKE_MANUAL"] = {
        "roles": ["control"],
        "machine_readable": False,
        "source_type": "manual",
    }
    patched_alias = dict(original.alias_to_id)
    patched_alias["fakemanual"] = "FAKE_MANUAL"

    monkeypatch.setattr(
        SourceRegistry,
        "load",
        classmethod(lambda cls, path="config/source_capabilities.yaml": SourceRegistry(patched_sources, patched_alias)),
    )

    candidate = candidate.model_copy(update={"join_plan": {"controls": ["FakeManual"], "boundary_source": ["TIGER_Lines"]}})

    gate = precheck_candidate(candidate)
    assert "missing_machine_readable_source" in gate["reasons"]

    repaired, repaired_gate, _ = repair_candidate(candidate, gate)
    final_gate = validate_candidate_export_contract(repaired, repaired_gate, no_paid_api=True)
    assert final_gate["shortlist_status"] in {"review", "blocked"}
    assert final_gate["claude_code_ready"] is False


def test_source_alias_fixture_repairs_without_fake_partial_flag() -> None:
    candidate = _load_candidate("partial_registry_match.json")

    gate = precheck_candidate(candidate)
    repaired, repaired_gate, _ = repair_candidate(candidate, gate)
    final_gate = validate_candidate_export_contract(repaired, repaired_gate, no_paid_api=True)

    assert "partial_registry_match" not in final_gate.get("reasons", [])
    assert final_gate["shortlist_status"] in {"ready", "review", "blocked"}
