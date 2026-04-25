import json
import os

from agents import settings
from agents.data_access_agent import DataAccessAgent
from agents.logging_config import get_logger
from agents.orchestrator import ResearchState

logger = get_logger(__name__)


class DataFetcherAgent:
    """Compatibility wrapper: emit data accessibility report instead of mock parquet."""

    def __init__(self):
        self.report_path = settings.data_access_report_path()

    def run(self, state: ResearchState) -> ResearchState:
        logger.info("--- Module 4: Data Accessibility Report ---")
        result = DataAccessAgent().run(state)

        run_index_path = settings.run_index_path()
        os.makedirs(os.path.dirname(run_index_path), exist_ok=True)
        with open(run_index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "plan": state.get("current_plan_path", settings.research_plan_path()),
                    "literature": settings.literature_index_path(),
                    "draft": settings.draft_path(),
                    "data": self.report_path,
                    "context": state.get("research_context_path", settings.research_context_path()),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "execution_status": "completed",
            "raw_data_manifest_path": result.get("raw_data_manifest_path", self.report_path),
        }


def data_fetcher_node(state: ResearchState) -> ResearchState:
    return DataFetcherAgent().run(state)
