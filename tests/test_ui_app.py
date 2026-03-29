"""
Import-level smoke tests for ui/app.py.

Streamlit runs all top-level code at import time (set_page_config, session_state
accesses, widget calls, etc.).  This test patches the Streamlit namespace so that
importing app.py does not raise and does not require a running Streamlit server or
any external API.
"""
import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch


def _build_streamlit_stub() -> types.ModuleType:
    """Return a minimal mock of the `streamlit` module."""
    st = types.ModuleType("streamlit")

    # --- session_state: a plain namespace that supports attribute get/set ---
    st.session_state = MagicMock()
    st.session_state.__contains__ = MagicMock(return_value=False)

    # --- UI primitives that execute at import time in app.py ---
    for attr in (
        "set_page_config",
        "title",
        "markdown",
        "header",
        "subheader",
        "text_input",
        "button",
        "columns",
        "info",
        "warning",
        "success",
        "spinner",
        "json",
        "write",
        "rerun",
    ):
        setattr(st, attr, MagicMock(return_value=MagicMock()))

    # columns() must return a context-manager-compatible pair
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    st.columns = MagicMock(return_value=(col, col))

    return st


def test_ui_app_imports_without_error() -> None:
    """
    Importing ui/app.py with a stubbed `streamlit` must succeed without raising
    any exception and without calling any real external service.
    """
    # Patch streamlit before import so app.py uses the stub
    stub_st = _build_streamlit_stub()

    # Also stub build_orchestrator so no LLM clients are initialised at import time
    fake_graph = MagicMock()
    fake_graph.get_state = MagicMock(return_value=MagicMock(values={}, next=[]))
    fake_orchestrator = MagicMock(return_value=fake_graph)

    # Remove previously imported ui.app so we get a fresh import
    for key in list(sys.modules.keys()):
        if "ui.app" in key or key == "app":
            del sys.modules[key]

    with patch.dict(sys.modules, {"streamlit": stub_st}):
        with patch("agents.orchestrator.build_orchestrator", fake_orchestrator):
            import ui.app  # noqa: F401 – import for side-effect only

    # If we reach here the import did not crash
    assert True


def test_ui_app_calls_set_page_config() -> None:
    """set_page_config must be called at module level in app.py."""
    stub_st = _build_streamlit_stub()
    fake_graph = MagicMock()
    fake_graph.get_state = MagicMock(return_value=MagicMock(values={}, next=[]))

    for key in list(sys.modules.keys()):
        if "ui.app" in key or key == "app":
            del sys.modules[key]

    with patch.dict(sys.modules, {"streamlit": stub_st}):
        with patch("agents.orchestrator.build_orchestrator", MagicMock(return_value=fake_graph)):
            import ui.app  # noqa: F401

    stub_st.set_page_config.assert_called_once()
