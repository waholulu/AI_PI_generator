"""Tests for ideation_node() failure modes.

The old V2→V0 emergency fallback chain is no longer reachable: ideation_node()
routes exclusively to the Candidate Factory (or legacy V0 in explicit legacy
mode).  This file verifies the new failure semantics.
"""
from unittest.mock import patch

from agents.ideation_agent import ideation_node


def test_ideation_node_returns_factory_failure_on_compose_error(tmp_path, monkeypatch) -> None:
    """When compose_candidates() returns empty, factory returns execution_status=failed."""
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("LEGACY_IDEATION", raising=False)

    with patch("agents.candidate_factory_ideation.compose_candidates", return_value=[]):
        out = ideation_node({"domain_input": "GeoAI test", "execution_status": "starting"})

    assert out.get("execution_status") == "failed"
    assert "ideation:no_candidates_from_factory" in out.get("degraded_nodes", [])


def test_ideation_node_legacy_explicit_routes_to_v0(monkeypatch) -> None:
    """Explicit legacy mode routes to IdeationAgentV0, not Candidate Factory."""
    monkeypatch.setenv("LEGACY_IDEATION", "1")

    called = []

    class _DummyV0:
        def run(self, state):
            called.append("v0")
            return {"execution_status": "harvesting"}

    with patch("agents._legacy.ideation_agent_v0.IdeationAgentV0", return_value=_DummyV0()):
        out = ideation_node({"domain_input": "test", "legacy_ideation": True})

    assert called == ["v0"]
    assert out["execution_status"] == "harvesting"
