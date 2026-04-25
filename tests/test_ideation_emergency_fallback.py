import json

from agents import settings
from agents.ideation_agent import ideation_node


class _BrokenV2:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, state):
        raise RuntimeError("v2 failed")


class _BrokenV0:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("v0 init failed")


def test_ideation_emergency_fallback_when_v2_and_v0_fail(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("LEGACY_IDEATION", "0")

    import agents.ideation_agent_v2 as ideation_v2_mod
    import agents._legacy.ideation_agent_v0 as ideation_v0_mod

    monkeypatch.setattr(ideation_v2_mod, "IdeationAgentV2", _BrokenV2)
    monkeypatch.setattr(ideation_v0_mod, "IdeationAgentV0", _BrokenV0)

    out = ideation_node({"domain_input": "GeoAI fallback", "execution_status": "starting"})

    assert out["execution_status"] == "harvesting"
    assert "ideation" in out.get("degraded_nodes", [])

    screening_path = settings.output_dir() / "topic_screening.json"
    payload = json.loads(screening_path.read_text(encoding="utf-8"))
    assert payload["fallback_reason"] == "llm_unavailable_and_legacy_fallback_failed"
    assert payload["candidates"]
