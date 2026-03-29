"""
Smoke tests for the main.py CLI entrypoint.

These tests use monkeypatching to avoid real LLM / graph calls and to simulate
user input, so they run completely offline.
"""
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _import_main():
    """Import main freshly each test to avoid state leakage."""
    if "main" in sys.modules:
        del sys.modules["main"]
    import main as m
    return m


def test_main_exits_on_empty_domain(capsys: pytest.CaptureFixture[str]) -> None:
    """main() must exit with code 1 when the user enters an empty domain."""
    m = _import_main()

    with patch("builtins.input", return_value=""):
        with pytest.raises(SystemExit) as exc_info:
            m.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "未输入" in captured.out or "exit" in captured.out.lower() or exc_info.value.code == 1


def test_main_runs_with_valid_domain(capsys: pytest.CaptureFixture[str]) -> None:
    """main() must run without raising when given a valid domain and a mocked graph."""
    m = _import_main()

    fake_graph = MagicMock()
    # stream() must yield at least one event dict
    fake_graph.stream = MagicMock(return_value=[{"ideation": {"execution_status": "harvesting"}}])

    with patch("builtins.input", return_value="GeoAI and Urban Planning"):
        with patch("agents.orchestrator.build_orchestrator", return_value=fake_graph):
            # Should not raise
            m.main()

    captured = capsys.readouterr()
    assert "Auto-PI" in captured.out


def test_main_handles_graph_exception_gracefully(capsys: pytest.CaptureFixture[str]) -> None:
    """main() must catch graph-level exceptions and print a message rather than crashing."""
    m = _import_main()

    fake_graph = MagicMock()
    fake_graph.stream = MagicMock(side_effect=RuntimeError("Simulated graph failure"))

    with patch("builtins.input", return_value="GeoAI and Urban Planning"):
        with patch("agents.orchestrator.build_orchestrator", return_value=fake_graph):
            # main() has a try/except; it should NOT re-raise
            m.main()

    captured = capsys.readouterr()
    assert "异常" in captured.out or "error" in captured.out.lower() or "Simulated" in captured.out
