from __future__ import annotations

import json
import os
from typing import Any

from agents import settings
from agents.data_accessibility import evaluate_data_sources, summarize_data_access
from agents.logging_config import get_logger
from agents.orchestrator import ResearchState
from models.research_plan_schema import ResearchPlan

logger = get_logger(__name__)


class DataAccessAgent:
    def __init__(self) -> None:
        self.report_path = settings.data_access_report_path()

    def run(self, state: ResearchState) -> ResearchState:
        plan_path = state.get("current_plan_path", settings.research_plan_path())
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_raw = json.load(f)
            plan = ResearchPlan.model_validate(plan_raw)
        except Exception as exc:
            logger.warning("Failed to load ResearchPlan for data access report: %s", exc)
            report = {
                "run_id": state.get("run_id", "unknown"),
                "project_title": "",
                "overall_verdict": "fail",
                "checks": [],
                "reasons": ["invalid_or_missing_research_plan"],
                "recommended_next_data_steps": [
                    "Confirm exposure/outcome definitions in research_plan.json.",
                    "Declare at least one accessible data source URL.",
                ],
            }
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return {
                "execution_status": "completed",
                "raw_data_manifest_path": self.report_path,
            }

        checks = evaluate_data_sources(plan)
        overall_verdict, reasons = summarize_data_access(checks)
        report = {
            "run_id": plan.run_id,
            "project_title": plan.project_title,
            "overall_verdict": overall_verdict,
            "checks": [c.model_dump() for c in checks],
            "reasons": reasons,
            "recommended_next_data_steps": self._next_steps(overall_verdict, reasons),
        }

        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return {
            "execution_status": "completed",
            "raw_data_manifest_path": self.report_path,
        }

    @staticmethod
    def _next_steps(overall_verdict: str, reasons: list[str]) -> list[str]:
        if overall_verdict == "pass":
            return [
                "Prioritize downloading source metadata and schema docs.",
                "Prototype exposure/outcome extraction with one sample dataset.",
            ]
        steps = []
        if "missing_exposure_role_source" in reasons:
            steps.append("Add at least one reachable exposure source URL.")
        if "missing_outcome_role_source" in reasons:
            steps.append("Add at least one reachable outcome source URL.")
        if "missing_machine_readable_source" in reasons:
            steps.append("Prefer CSV/JSON/Parquet data endpoints over PDFs.")
        if not steps:
            steps.append("Review source notes and align scope with geography/time window.")
        return steps


def data_access_node(state: ResearchState) -> ResearchState:
    return DataAccessAgent().run(state)
