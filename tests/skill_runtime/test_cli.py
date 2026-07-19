from __future__ import annotations

import json

from autopi_engine.cli import main


def test_cli_init_outputs_json(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    rc = main(["init", "--domain", "Built environment and health", "--run-id", "cli_run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "cli_run"
    assert payload["pending_action"]["kind"] == "search_plan"


def test_cli_status_unknown_run_returns_error(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))

    rc = main(["status", "--run-id", "missing"])

    assert rc == 2
    assert "unknown run_id" in capsys.readouterr().err
