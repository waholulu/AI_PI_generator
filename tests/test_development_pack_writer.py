import json

from agents import settings
from agents.development_pack_writer import write_development_pack


def test_development_pack_writer_outputs_expected_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    run_id = "run-pack-1"

    candidate = {
        "candidate_id": "beh_001",
        "template_id": "built_environment_health_v1",
        "exposure_family": "street_network",
        "exposure_source": "OSMnx_OpenStreetMap",
        "exposure_variables": ["intersection_density", "edge_density"],
        "outcome_family": "physical_inactivity",
        "outcome_source": "CDC_PLACES",
        "outcome_variables": ["physical_inactivity"],
        "unit_of_analysis": "census_tract",
        "join_plan": {"join_key": "GEOID"},
        "method_template": "cross_sectional_spatial_association",
        "key_threats": ["confounding"],
        "mitigations": {"confounding": "controls"},
        "technology_tags": ["osmnx"],
        "automation_risk": "low",
    }

    pack_dir = write_development_pack(run_id, candidate)

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
