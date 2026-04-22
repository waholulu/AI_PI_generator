"""Tests for agents/budget_tracker.py — Day 3 TDD."""

import threading
import pytest

from agents.budget_tracker import BudgetExceededError, BudgetTracker


def make_tracker(per_run=1.50, per_topic=0.10):
    return BudgetTracker(per_run_budget_usd=per_run, per_topic_budget_usd=per_topic)


# ── snapshot / initial state ──────────────────────────────────────────────────

def test_initial_snapshot():
    t = make_tracker()
    snap = t.snapshot()
    assert snap["per_run_spent_usd"] == 0.0
    assert snap["ratio"] == 0.0
    assert snap["new_topics_disabled"] is False


def test_snapshot_keys():
    t = make_tracker()
    snap = t.snapshot()
    for key in ["per_run_spent_usd", "per_run_budget", "ratio", "new_topics_disabled"]:
        assert key in snap


# ── record_call ───────────────────────────────────────────────────────────────

def test_record_call_updates_run_spent():
    t = make_tracker(per_run=1.50)
    t.record_call("t1", "gemini", 100, 50, 0.01)
    assert abs(t.snapshot()["per_run_spent_usd"] - 0.01) < 1e-9


def test_record_call_updates_per_topic():
    t = make_tracker(per_topic=0.10)
    t.record_call("topic_a", "gemini", 100, 50, 0.03)
    t.record_call("topic_a", "gemini", 100, 50, 0.02)
    # total for topic_a = 0.05
    assert abs(t._state.per_topic_spent.get("topic_a", 0) - 0.05) < 1e-9


def test_record_call_separate_topics():
    t = make_tracker()
    t.record_call("t1", "m", 100, 50, 0.04)
    t.record_call("t2", "m", 100, 50, 0.06)
    assert abs(t._state.per_topic_spent.get("t1", 0) - 0.04) < 1e-9
    assert abs(t._state.per_topic_spent.get("t2", 0) - 0.06) < 1e-9


# ── can_start_new_topic ───────────────────────────────────────────────────────

def test_can_start_below_90_percent():
    t = make_tracker(per_run=1.00)
    t.record_call("t1", "m", 1, 1, 0.89)
    assert t.can_start_new_topic() is True


def test_cannot_start_at_90_percent():
    t = make_tracker(per_run=1.00)
    t.record_call("t1", "m", 1, 1, 0.90)
    assert t.can_start_new_topic() is False


def test_cannot_start_above_90_percent():
    t = make_tracker(per_run=1.00)
    t.record_call("t1", "m", 1, 1, 0.95)
    assert t.can_start_new_topic() is False


# ── check_can_proceed ─────────────────────────────────────────────────────────

def test_check_can_proceed_ok():
    t = make_tracker(per_run=1.50, per_topic=0.10)
    t.record_call("t1", "m", 1, 1, 0.05)
    t.check_can_proceed("t1")  # should not raise


def test_check_can_proceed_raises_on_per_topic_exceeded():
    t = make_tracker(per_run=10.0, per_topic=0.10)
    t.record_call("t1", "m", 1, 1, 0.10)  # exactly at limit
    with pytest.raises(BudgetExceededError) as exc_info:
        t.check_can_proceed("t1")
    assert exc_info.value.scope == "per_topic"


def test_check_can_proceed_raises_on_per_run_exceeded():
    t = make_tracker(per_run=1.00, per_topic=10.0)
    t.record_call("t1", "m", 1, 1, 1.00)  # exactly at run limit
    with pytest.raises(BudgetExceededError) as exc_info:
        t.check_can_proceed("t1")
    assert exc_info.value.scope == "per_run"


def test_budget_exceeded_error_has_scope_and_amounts():
    err = BudgetExceededError("per_run", 1.50, 1.50)
    assert err.scope == "per_run"
    assert err.spent == 1.50
    assert err.limit == 1.50


# ── thread safety ─────────────────────────────────────────────────────────────

def test_concurrent_record_calls_are_thread_safe():
    t = make_tracker(per_run=100.0, per_topic=100.0)
    errors = []

    def do_calls():
        try:
            for _ in range(50):
                t.record_call("shared_topic", "m", 1, 1, 0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=do_calls) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    expected = 4 * 50 * 0.001
    assert abs(t.snapshot()["per_run_spent_usd"] - expected) < 1e-6


# ── ui_callback ───────────────────────────────────────────────────────────────

def test_ui_callback_called_on_record():
    received = []

    def cb(snap):
        received.append(snap)

    t = BudgetTracker(per_run_budget_usd=1.50, per_topic_budget_usd=0.10, ui_callback=cb)
    t.record_call("t1", "m", 1, 1, 0.01)
    assert len(received) == 1
    assert "per_run_spent_usd" in received[0]
