from pathlib import Path

from agents import settings


def test_run_scope_changes_output_config_data_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    token = settings.activate_run_scope("run-123")
    try:
        assert settings.output_dir() == tmp_path / "runs" / "run-123" / "output"
        assert settings.config_dir() == tmp_path / "runs" / "run-123" / "config"
        assert settings.data_dir() == tmp_path / "runs" / "run-123" / "data"
        assert settings.field_scan_path().endswith("runs/run-123/output/field_scan.json")
        assert settings.research_plan_path().endswith("runs/run-123/config/research_plan.json")
    finally:
        settings.deactivate_run_scope(token)

    # Back to global root after scope reset.
    assert settings.output_dir() == tmp_path / "output"
    assert settings.config_dir() == tmp_path / "config"
    assert settings.data_dir() == tmp_path / "data"


def test_memory_dir_stays_global_under_run_scope(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    token = settings.activate_run_scope("run-abc")
    try:
        assert settings.memory_dir() == tmp_path / "memory"
        assert settings.idea_memory_csv_path().endswith("/memory/idea_memory.csv")
    finally:
        settings.deactivate_run_scope(token)
