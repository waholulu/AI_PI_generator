"""
Dual-layer (per-run + per-topic) LLM cost guard for Module 1 reflection loop.

Thresholds are loaded from config/reflection_config.yaml cost_guardrails section.
All state mutations are protected by a threading.Lock for concurrent topic processing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import yaml

from agents.logging_config import get_logger
from agents.settings import reflection_config_path

logger = get_logger(__name__)


class BudgetExceededError(Exception):
    """Raised when per-run or per-topic budget is exhausted."""

    def __init__(self, scope: str, spent: float, limit: float) -> None:
        self.scope = scope
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"Budget exceeded [{scope}]: spent ${spent:.4f} / limit ${limit:.4f}"
        )


@dataclass
class BudgetState:
    per_run_budget: float
    per_topic_budget: float
    warn_ratio: float
    disable_new_ratio: float
    kill_ratio: float
    per_run_spent: float = 0.0
    per_topic_spent: dict = field(default_factory=dict)  # topic_id → float

    @property
    def run_ratio(self) -> float:
        return self.per_run_spent / self.per_run_budget if self.per_run_budget else 0.0

    def topic_ratio(self, topic_id: str) -> float:
        spent = self.per_topic_spent.get(topic_id, 0.0)
        return spent / self.per_topic_budget if self.per_topic_budget else 0.0


class BudgetTracker:
    """Thread-safe dual-layer cost tracker."""

    def __init__(
        self,
        per_run_budget_usd: Optional[float] = None,
        per_topic_budget_usd: Optional[float] = None,
        ui_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        cfg = self._load_config()
        cg = cfg.get("cost_guardrails", {})

        self._lock = threading.Lock()
        self._state = BudgetState(
            per_run_budget=per_run_budget_usd or cg.get("per_run_budget_usd", 1.50),
            per_topic_budget=per_topic_budget_usd or cg.get("per_topic_budget_usd", 0.10),
            warn_ratio=cg.get("emit_warning_at_ratio", 0.70),
            disable_new_ratio=cg.get("disable_new_at_ratio", 0.90),
            kill_ratio=cg.get("kill_switch_at_ratio", 1.00),
        )
        self._ui_callback = ui_callback

    @staticmethod
    def _load_config() -> dict:
        try:
            with open(reflection_config_path()) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("reflection_config.yaml unavailable: %s", e)
            return {}

    def record_call(
        self,
        topic_id: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        with self._lock:
            self._state.per_run_spent += cost_usd
            self._state.per_topic_spent[topic_id] = (
                self._state.per_topic_spent.get(topic_id, 0.0) + cost_usd
            )

        run_ratio = self._state.run_ratio
        if run_ratio >= self._state.warn_ratio:
            logger.warning(
                "Budget warning: per-run %.0f%% consumed ($%.4f / $%.2f)",
                run_ratio * 100,
                self._state.per_run_spent,
                self._state.per_run_budget,
            )

        snap = self.snapshot()
        if self._ui_callback:
            try:
                self._ui_callback(snap)
            except Exception:
                pass

    def check_can_proceed(self, topic_id: str) -> None:
        """Raise BudgetExceededError if per-topic or per-run limit is hit."""
        with self._lock:
            topic_spent = self._state.per_topic_spent.get(topic_id, 0.0)
            topic_limit = self._state.per_topic_budget
            if topic_spent >= topic_limit * self._state.kill_ratio:
                raise BudgetExceededError("per_topic", topic_spent, topic_limit)

            run_spent = self._state.per_run_spent
            run_limit = self._state.per_run_budget
            if run_spent >= run_limit * self._state.kill_ratio:
                raise BudgetExceededError("per_run", run_spent, run_limit)

    def can_start_new_topic(self) -> bool:
        """Return False when per-run budget is >= 90% consumed."""
        return self._state.run_ratio < self._state.disable_new_ratio

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "per_run_spent_usd": self._state.per_run_spent,
                "per_run_budget": self._state.per_run_budget,
                "ratio": self._state.run_ratio,
                "new_topics_disabled": not self.can_start_new_topic(),
            }
