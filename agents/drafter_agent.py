import json
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agents import settings
from agents.logging_config import get_logger
from agents.orchestrator import ResearchState

logger = get_logger(__name__)


class DrafterAgent:
    def __init__(self):
        self.output_md = settings.draft_path()
        try:
            model = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
        except Exception as e:
            logger.warning("Gemini API Key missing. Using mocked generation. Error: %s", e)
            self.llm = None

    def load_prompt(self):
        prompt_path = settings.academic_drafter_prompt_path()
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Prompt file not found at %s, using fallback.", prompt_path)
            return "You are an expert academic writer. Write a structured research draft."

    def load_literature_evidence(self, index_path: str) -> str:
        if not os.path.exists(index_path):
            return "No literature evidence provided."
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                evidence = json.load(f)
            return json.dumps(evidence, indent=2)
        except Exception:
            return "Failed to parse literature evidence."

    def load_research_context(self, context_path: str) -> str:
        if not os.path.exists(context_path):
            return ""
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                ctx = json.load(f)
            # Extract the most useful parts for drafting
            parts = []
            selected = ctx.get("selected_topic", {})
            if selected.get("title"):
                parts.append(f"Selected Topic: {selected['title']}")
            if selected.get("quantitative_specs"):
                parts.append(f"Quantitative Specs: {json.dumps(selected['quantitative_specs'], indent=2)}")
            if selected.get("publishability"):
                parts.append(f"Target Journals: {selected['publishability']}")
            plan_ess = ctx.get("plan_essentials", {})
            if plan_ess.get("research_questions"):
                parts.append(f"Research Questions: {json.dumps(plan_ess['research_questions'])}")
            if plan_ess.get("hypotheses"):
                parts.append(f"Hypotheses: {json.dumps(plan_ess['hypotheses'])}")
            if plan_ess.get("methodology"):
                parts.append(f"Methodology: {json.dumps(plan_ess['methodology'], indent=2)}")
            return "\n\n".join(parts)
        except Exception:
            return ""

    def run(self, state: ResearchState) -> ResearchState:
        logger.info("--- Module 3: Academic Drafter ---")

        plan_path = state.get("current_plan_path", settings.research_plan_path())
        index_path = state.get("literature_inventory_path", settings.literature_index_path())
        context_path = state.get("research_context_path", settings.research_context_path())

        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = f.read()
        except Exception as e:
            logger.error("Failed to load plan for drafting: %s", e)
            plan = "Draft fallback plan"

        evidence = self.load_literature_evidence(index_path)
        research_context = self.load_research_context(context_path)
        system_prompt = self.load_prompt()

        logger.info("Synthesizing Draft_v1.md via Gemini Pro...")

        try:
            if not self.llm:
                raise ValueError("LLM not initialized")

            template = "{system}\n\nPLAN:\n{plan}\n\n"
            if research_context:
                template += "RESEARCH CONTEXT:\n{research_context}\n\n"
            template += "EVIDENCE CARDS:\n{evidence}"

            prompt = PromptTemplate.from_template(template)
            prompt_inputs = {
                "system": system_prompt,
                "plan": plan,
                "evidence": evidence,
                "research_context": research_context,
            }

            if isinstance(self.llm, ChatGoogleGenerativeAI):
                chain = prompt | self.llm
                result = chain.invoke(prompt_inputs)
            else:
                rendered = prompt.format(**prompt_inputs)
                payload = {"prompt": rendered}
                result = self.llm.invoke(payload)

            draft_content = getattr(result, "content", str(result))
        except Exception as e:
            logger.warning("Draft generation failed (%s)", e)
            draft_content = "> This draft is AI-assisted.\n\n# Fallback Draft\nFailed to reach API."

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
