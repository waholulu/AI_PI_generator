"""
Unit tests for agents/keyword_planner.py.

All LLM calls are mocked so this file runs without API keys.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.keyword_planner import KeywordPlan, KeywordPlanner, TopicQueryGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(topics_data: list[dict]) -> KeywordPlan:
    """Build a KeywordPlan from raw topic dicts."""
    topics = [TopicQueryGroup(**t) for t in topics_data]
    return KeywordPlan(primary_domains=["Urban Science"], methods=["causal inference"], topics=topics)


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

def test_fallback_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When OPENALEX_QUERY_REWRITE_ENABLED=false the planner must fall back immediately."""
    monkeypatch.setenv("OPENALEX_QUERY_REWRITE_ENABLED", "false")
    planner = KeywordPlanner()
    result = planner.plan("some broad domain")

    assert result["used_fallback"] is True
    assert result["query_pool"] == ["some broad domain"]
    assert result["topics"][0]["queries"] == ["some broad domain"]


def test_fallback_when_llm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the LLM call itself raises, the planner propagates RuntimeError."""
    monkeypatch.setenv("OPENALEX_QUERY_REWRITE_ENABLED", "true")

    planner = KeywordPlanner.__new__(KeywordPlanner)
    planner._enabled = True
    planner._max_queries = 10
    # Simulate a broken LLM
    broken_llm = MagicMock()
    broken_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("API error")
    planner._llm = broken_llm

    with pytest.raises(RuntimeError, match="KeywordPlanner failed"):
        planner.plan("GeoAI and health")


# ---------------------------------------------------------------------------
# Happy-path: LLM returns a valid plan
# ---------------------------------------------------------------------------

def test_plan_returns_deduplicated_query_pool() -> None:
    """Duplicate queries across topic groups must be removed from the pool."""
    plan = _make_plan([
        {"label": "Health", "queries": ["built environment health", "walkability cardiovascular"]},
        {"label": "Health duplicate", "queries": ["built environment health", "air quality mortality"]},
    ])

    planner = KeywordPlanner.__new__(KeywordPlanner)
    planner._enabled = True
    planner._max_queries = 10
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = plan
    planner._llm = mock_llm

    result = planner.plan("public health and built environment")

    pool = result["query_pool"]
    assert len(pool) == len(set(q.lower() for q in pool)), "Pool contains duplicates"
    assert "built environment health" in pool
    assert "walkability cardiovascular" in pool
    assert "air quality mortality" in pool
    assert pool.count("built environment health") == 1


def test_plan_respects_max_queries() -> None:
    """query_pool must not exceed OPENALEX_QUERY_REWRITE_MAX_QUERIES."""
    plan = _make_plan([
        {
            "label": f"Topic {i}",
            "queries": [f"query topic{i} alpha", f"query topic{i} beta", f"query topic{i} gamma"],
        }
        for i in range(10)
    ])

    planner = KeywordPlanner.__new__(KeywordPlanner)
    planner._enabled = True
    planner._max_queries = 5
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = plan
    planner._llm = mock_llm

    result = planner.plan("very broad input")

    assert len(result["query_pool"]) <= 5


def test_plan_structure_keys() -> None:
    """plan() must always return a dict with all required keys."""
    plan = _make_plan([
        {"label": "Urban models", "queries": ["urban foundation models city", "LLM urban planning"]}
    ])

    planner = KeywordPlanner.__new__(KeywordPlanner)
    planner._enabled = True
    planner._max_queries = 10
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = plan
    planner._llm = mock_llm

    result = planner.plan("urban science")

    for key in ("query_pool", "topics", "primary_domains", "methods", "used_fallback"):
        assert key in result, f"Missing key: {key}"

    assert isinstance(result["query_pool"], list)
    assert isinstance(result["topics"], list)
    assert result["used_fallback"] is False


def test_plan_with_extra_context() -> None:
    """Extra context is forwarded to the LLM prompt without errors."""
    plan = _make_plan([
        {"label": "Spatial networks", "queries": ["spatial network urban mobility"]}
    ])

    planner = KeywordPlanner.__new__(KeywordPlanner)
    planner._enabled = True
    planner._max_queries = 10
    mock_llm = MagicMock()
    mock_structured = mock_llm.with_structured_output.return_value
    mock_structured.invoke.return_value = plan
    planner._llm = mock_llm

    result = planner.plan("urban systems", extra_context="Selected topic: spatial analysis")

    assert result["used_fallback"] is False
    # Verify LLM was actually called with a prompt containing the extra context
    call_args = mock_structured.invoke.call_args[0][0]
    assert "spatial analysis" in call_args
