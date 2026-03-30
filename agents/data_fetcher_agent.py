import json
import os
from typing import List

from langchain_anthropic import ChatAnthropic

from agents import settings
from agents.logging_config import get_logger
from agents.orchestrator import ResearchState

logger = get_logger(__name__)


class DataFetcherAgent:
    def __init__(self):
        self.output_dir = str(settings.raw_data_dir())
        self.manifest_path = settings.raw_manifest_path()

        try:
            self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        except Exception:
            self.llm = None

    def execute_mock_fetch(self) -> dict:
        """Since E2B Sandbox has been removed, this simulates the output."""
        logger.info("Executing mock data fetch...")
        return {"status": "success", "file": "simulated_data.parquet", "epsg": 4326}

    def fetch_data_source(self, source: dict) -> dict:
        name = source.get("name") or source.get("source", "Unknown Source")
        api_endpoint = source.get("api_endpoint", "")
        data_type = source.get("data_type", "")
        logger.info("Fetching %s from %s...", name, api_endpoint)

        result = self.execute_mock_fetch()

        return {
            "source_name": name,
            "data_type": data_type,
            "filename": result["file"],
            "epsg_code": result["epsg"],
            "status": result["status"],
        }

    def run(self, state: ResearchState) -> ResearchState:
        logger.info("--- Module 4: Data Fetcher & Circuit Breaker ---")

        plan_path = state.get("current_plan_path", settings.research_plan_path())
        context_path = state.get("research_context_path", settings.research_context_path())
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
        except Exception:
            plan = {"data_sources": [{"name": "Mock Test Data", "api_endpoint": "http://example.com"}]}

        outcomes: List[str] = []
        exposures: List[str] = []
        if os.path.exists(context_path):
            try:
                with open(context_path, "r", encoding="utf-8") as f:
                    context = json.load(f)
                plan_ess = context.get("plan_essentials", {}) if isinstance(context, dict) else {}
                raw_outcomes = plan_ess.get("outcomes", [])
                raw_exposures = plan_ess.get("exposures", [])
                outcomes = [
                    (o.get("variable") or o.get("name"))
                    if isinstance(o, dict)
                    else str(o)
                    for o in raw_outcomes
                ]
                exposures = [
                    (e.get("variable") or e.get("name"))
                    if isinstance(e, dict)
                    else str(e)
                    for e in raw_exposures
                ]
                outcomes = [v for v in outcomes if isinstance(v, str) and v]
                exposures = [v for v in exposures if isinstance(v, str) and v]
            except Exception as e:
                logger.warning("Failed to load research context for data fetcher: %s", e)

        data_sources = plan.get("data_sources", [])
        manifest = []

        for source in data_sources:
            entry = self.fetch_data_source(source)
            entry["covers_outcomes"] = outcomes
            entry["covers_exposures"] = exposures
            manifest.append(entry)

            dummy_file = os.path.join(self.output_dir, entry["filename"])
            with open(dummy_file, "w") as f:
                f.write("mock parquet data")

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump({"datasets": manifest}, f, indent=2)

        logger.info("Data fetching complete. Manifest written to %s", self.manifest_path)
        logger.info("[SYSTEM HALT] Open Source Data Collection Complete.")

        run_index_path = settings.run_index_path()
        with open(run_index_path, "w", encoding="utf-8") as f:
            json.dump({
                "plan": plan_path,
                "literature": settings.literature_index_path(),
                "draft": settings.draft_path(),
                "data": self.manifest_path,
                "context": context_path,
            }, f, indent=2)

        return {
            "execution_status": "completed",
            "raw_data_manifest_path": self.manifest_path,
        }


def data_fetcher_node(state: ResearchState) -> ResearchState:
    agent = DataFetcherAgent()
    return agent.run(state)
