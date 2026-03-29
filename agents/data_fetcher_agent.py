import json
import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from agents.orchestrator import ResearchState

class DataFetcherAgent:
    def __init__(self):
        self.output_dir = "data/raw"
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.output_dir, "manifest.json")
        
        # Will fail gracefully if API key missing
        try:
            self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        except Exception:
            self.llm = None

    def execute_mock_fetch(self) -> dict:
        """
        Since E2B Sandbox has been removed, this simulates the output.
        """
        print("Executing mock data fetch...")
        return {"status": "success", "file": "simulated_data.parquet", "epsg": 4326}

    def fetch_data_source(self, source: dict) -> dict:
        name = source.get("name") or source.get("source", "Unknown Source")
        api_endpoint = source.get("api_endpoint", "")
        data_type = source.get("data_type", "")
        print(f"Fetching {name} from {api_endpoint}...")
        
        # 1. Execute Mock Fetch
        result = self.execute_mock_fetch()
        
        # 2. Create manifest entry
        return {
            "source_name": name,
            "data_type": data_type,
            "filename": result["file"],
            "epsg_code": result["epsg"],
            "status": result["status"],
        }

    def run(self, state: ResearchState) -> ResearchState:
        print("--- Module 4: Data Fetcher & Circuit Breaker ---")
        
        plan_path = state.get("current_plan_path", "config/research_plan.json")
        context_path = state.get("research_context_path", "output/research_context.json")
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
        except Exception:
            plan = {"data_sources": [{"name": "Mock Test Data", "api_endpoint": "http://example.com"}]}

        # Load essentials from shared research context to annotate manifest
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
                print(f"Warning: Failed to load research context for data fetcher: {e}")

        data_sources = plan.get("data_sources", [])
        manifest = []
        
        # Fetch loop
        for source in data_sources:
            entry = self.fetch_data_source(source)
            entry["covers_outcomes"] = outcomes
            entry["covers_exposures"] = exposures
            manifest.append(entry)
            
            # Write dummy parquet file for testing
            dummy_file = os.path.join(self.output_dir, entry["filename"])
            with open(dummy_file, "w") as f:
                f.write("mock parquet data")
        
        # Write manifest
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump({"datasets": manifest}, f, indent=2)

        print(f"Data fetching complete. Manifest written to {self.manifest_path}")
        
        # Halting Sequence
        print("="*60)
        print("[SYSTEM HALT] Open Source Data Collection Complete.")
        print("System shutting down. No analysis will be performed beyond this point.")
        print("="*60)
        
        # Write Run Index
        with open("output/run_index.json", "w", encoding="utf-8") as f:
            json.dump({
                "plan": plan_path,
                "literature": "data/literature/index.json",
                "draft": "output/Draft_v1.md",
                "data": self.manifest_path,
                "context": context_path,
            }, f, indent=2)

        # Force exit
        # sys.exit(0) # Commented out in modular testing so we can verify state
        return {
            "execution_status": "completed",
            "raw_data_manifest_path": self.manifest_path
        }

def data_fetcher_node(state: ResearchState) -> ResearchState:
    agent = DataFetcherAgent()
    return agent.run(state)
