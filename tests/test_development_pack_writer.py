import json

import yaml

from agents import settings
from agents.development_pack_writer import write_development_pack

_CANDIDATE_BASE = {
    "candidate_id": "beh_001",
    "template_id": "built_environment_health_v1",
    "exposure_family": "street_network",
    "exposure_source": "OSMnx_OpenStreetMap",
    "exposure_variables": ["intersection_density", "edge_density"],
    "outcome_family": "physical_inactivity",
    "outcome_source": "CDC_PLACES",
    "outcome_variables": ["physical_inactivity"],
    "unit_of_analysis": "census_tract",
    "join_plan": {"join_key": "GEOID", "boundary_source": ["TIGER_Lines"], "controls": ["ACS"]},
    "method_template": "cross_sectional_spatial_association",
    "key_threats": ["confounding"],
    "mitigations": {"confounding": "controls"},
    "technology_tags": ["osmnx"],
    "automation_risk": "low",
}

_CANDIDATE_SLD = {
    **_CANDIDATE_BASE,
    "candidate_id": "beh_sld_001",
    "exposure_family": "density",
    "exposure_source": "EPA_Smart_Location_Database",
    "exposure_variables": ["D1B", "D3B"],
    "technology_tags": [],
}


def test_development_pack_writer_outputs_expected_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-pack-1"

    pack_dir = write_development_pack(run_id, _CANDIDATE_BASE)

    expected = {
        "README.md",
        "implementation_spec.json",
        "data_contract.yaml",
        "feature_plan.yaml",
        "analysis_plan.yaml",
        "acceptance_tests.md",
        "claude_task_prompt.md",
    }
    files = {p.name for p in pack_dir.glob("*") if p.is_file()}
    assert expected.issubset(files)

    spec_path = settings.run_root(run_id) / "output" / "candidates" / "beh_001" / "implementation_spec.json"
    assert spec_path.exists()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    assert payload["candidate_id"] == "beh_001"


def test_development_pack_writes_data_source_notes(monkeypatch, tmp_path) -> None:
    """data_source_notes.md is written and contains source-aware content."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-pack-notes"

    pack_dir = write_development_pack(run_id, _CANDIDATE_SLD)
    notes_path = pack_dir / "data_source_notes.md"

    assert notes_path.exists(), "data_source_notes.md must be written"
    assert notes_path.stat().st_size > 0

    content = notes_path.read_text(encoding="utf-8")
    assert "EPA_Smart_Location_Database" in content
    assert "census_block_group" in content
    assert "census_tract" in content


def test_development_pack_writes_data_lineage_plan(monkeypatch, tmp_path) -> None:
    """data_lineage_plan.yaml is written and contains grain-conversion steps."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-pack-lineage"

    pack_dir = write_development_pack(run_id, _CANDIDATE_SLD)
    lineage_path = pack_dir / "data_lineage_plan.yaml"

    assert lineage_path.exists(), "data_lineage_plan.yaml must be written"
    assert lineage_path.stat().st_size > 0

    content = lineage_path.read_text(encoding="utf-8")
    assert "analysis_unit" in content
    assert "lineage_steps" in content


def test_data_contract_is_source_aware(monkeypatch, tmp_path) -> None:
    """data_contract.yaml for EPA SLD candidate includes native_grain and aggregation info."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-pack-contract"

    pack_dir = write_development_pack(run_id, _CANDIDATE_SLD)
    contract_path = pack_dir / "data_contract.yaml"

    assert contract_path.exists()
    content = contract_path.read_text(encoding="utf-8")

    # Must include native_grain and target_grain for source-aware contract
    assert "native_grain" in content
    assert "target_grain" in content
