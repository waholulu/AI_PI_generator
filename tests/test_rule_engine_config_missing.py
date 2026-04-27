from __future__ import annotations

from pathlib import Path

import pytest

from agents.rule_engine import RuleEngine
from tests.test_rule_engine import make_topic


def test_g2_blocks_when_spatial_config_missing(monkeypatch):
    missing = Path("/tmp/does-not-exist-spatial-units.yaml")
    monkeypatch.setattr("agents.rule_engine.spatial_units_path", lambda: str(missing))

    engine = RuleEngine()
    result = engine.check_G2_scale_alignment(make_topic())

    assert result.passed is False
    assert result.refinable is False
    assert "config_unavailable_blocking" in result.reason
