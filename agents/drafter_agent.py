import json
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agents import settings
from agents.logging_config import get_logger
from agents.orchestrator import ResearchState
from models.research_plan_schema import ResearchPlan

logger = get_logger(__name__)


class DrafterAgent:
    def __init__(self):
        self.output_md = settings.draft_path()
        model = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
        try:
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
        except Exception as exc:
            logger.warning("LLM unavailable for drafter; fallback memo will be used: %s", exc)
            self.llm = None

    def load_prompt(self):
        prompt_path = settings.research_memo_prompt_path()
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Prompt file not found at %s, using fallback.", prompt_path)
            return (
                "Write a Research Memo with exactly 8 sections: "
                "1.Proposed Title 2.Research Question 3.Why This Matters "
                "4.Data and Measurement 5.Empirical Strategy 6.Related Literature "
                "7.Main Risks 8.Recommended Next Steps. "
                "Only cite papers from evidence cards; if evidence is weak, state: literature evidence is limited."
            )

    def load_literature_evidence(self, index_path: str) -> tuple[str, list[dict]]:
        if not os.path.exists(index_path):
            return "No literature evidence provided.", []
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                evidence = json.load(f)
            if not isinstance(evidence, list):
                evidence = []
            return json.dumps(evidence, indent=2, ensure_ascii=False), evidence
        except Exception:
            return "Failed to parse literature evidence.", []

    def load_data_access_report(self) -> str:
        report_path = settings.data_access_report_path()
        if not os.path.exists(report_path):
            return "No data access report provided."
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            return json.dumps(report, indent=2, ensure_ascii=False)
        except Exception:
            return "Failed to parse data access report."

    def _fallback_memo(self, plan: ResearchPlan, evidence: list[dict]) -> str:
        limited = "literature evidence is limited" if not evidence else ""
        title = plan.project_title or "Research Memo"
        rq = plan.research_question or "Research question not specified."
        data_sources = ", ".join(ds.name for ds in plan.data_sources) or "No data sources listed."
        method = plan.identification.primary_method or "Method not specified."
        hypotheses = "\n".join(f"- {h}" for h in (plan.hypotheses or [])) or "- Hypotheses pending."
        risks = "\n".join(f"- {r}" for r in (plan.feasibility.main_risks or [])) or "- Data availability uncertainty."
        lit_points = []
        for card in evidence[:3]:
            lit_points.append(f"- {card.get('title', 'Untitled')} ({card.get('year', 'n/a')})")
        lit_text = "\n".join(lit_points) if lit_points else "- No evidence cards available."
        if limited:
            lit_text = f"{lit_text}\n- {limited}"
        return (
            "# Research Memo\n\n"
            "## 1. Proposed Title\n"
            f"{title}\n\n"
            "## 2. Research Question\n"
            f"{rq}\n\n"
            "## 3. Why This Matters\n"
            f"{plan.short_rationale}\n\n"
            "## 4. Data and Measurement\n"
            f"- Exposure: {plan.exposure.name}\n- Outcome: {plan.outcome.name}\n- Data sources: {data_sources}\n\n"
            "## 5. Empirical Strategy\n"
            f"- Primary method: {method}\n- Hypotheses:\n{hypotheses}\n\n"
            "## 6. Related Literature\n"
            f"{lit_text}\n\n"
            "## 7. Main Risks\n"
            f"{risks}\n\n"
            "## 8. Recommended Next Steps\n"
            "- Confirm data source variable mappings.\n"
            "- Run pilot extraction for one geography/time slice.\n"
            "- Refine identification threats and mitigations before full execution.\n"
        )

    def run(self, state: ResearchState) -> ResearchState:
        logger.info("--- Module 3: Research Memo Drafter ---")

        plan_path = state.get("current_plan_path", settings.research_plan_path())
        index_path = state.get("literature_inventory_path", settings.literature_index_path())
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_obj = ResearchPlan.model_validate(json.load(f))
        plan = json.dumps(plan_obj.model_dump(), indent=2, ensure_ascii=False)

        evidence, evidence_list = self.load_literature_evidence(index_path)
        data_access_report = self.load_data_access_report()
        system_prompt = self.load_prompt()

        logger.info("Synthesizing Research Memo...")

        template = (
            "{system}\n\n"
            "PLAN:\n{plan}\n\n"
            "EVIDENCE CARDS:\n{evidence}\n\n"
            "DATA ACCESS REPORT:\n{data_access_report}\n\n"
            "Constraints:\n"
            "- Only cite papers present in EVIDENCE CARDS.\n"
            "- Do not invent DOI, authors, or venues.\n"
            "- If evidence is sparse, explicitly state: literature evidence is limited.\n"
        )

        prompt = PromptTemplate.from_template(template)
        prompt_inputs = {
            "system": system_prompt,
            "plan": plan,
            "evidence": evidence,
            "data_access_report": data_access_report,
        }

        if self.llm is not None:
            try:
                chain = prompt | self.llm
                result = chain.invoke(prompt_inputs)
                draft_content = getattr(result, "content", str(result))
                if isinstance(draft_content, list):
                    draft_content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in draft_content
                    )
            except Exception as exc:
                logger.warning("LLM drafting failed; writing deterministic fallback memo: %s", exc)
                draft_content = self._fallback_memo(plan_obj, evidence_list)
        else:
            draft_content = self._fallback_memo(plan_obj, evidence_list)

        os.makedirs(os.path.dirname(self.output_md), exist_ok=True)
        with open(self.output_md, "w", encoding="utf-8") as f:
            f.write(draft_content)

        logger.info("Draft saved to %s", self.output_md)
        return {
            "execution_status": "fetching",
            "draft_content_path": self.output_md,
        }


def drafter_node(state: ResearchState) -> ResearchState:
    agent = DrafterAgent()
    return agent.run(state)
