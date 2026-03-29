import json
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.orchestrator import ResearchState

class DrafterAgent:
    def __init__(self):
        self.output_md = "output/Draft_v1.md"
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
        except Exception as e:
            print(f"Warning: Gemini API Key missing. Using mocked generation. Error: {e}")
            self.llm = None

    def load_prompt(self):
        prompt_path = "prompts/academic_drafter.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: Prompt file not found at {prompt_path}, using fallback.")
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

    def run(self, state: ResearchState) -> ResearchState:
        print("--- Module 3: Academic Drafter ---")
        
        # Gather inputs
        plan_path = state.get("current_plan_path", "config/research_plan.json")
        index_path = state.get("literature_inventory_path", "data/literature/index.json")
        
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = f.read()
        except Exception as e:
            print(f"Failed to load plan for drafting: {e}")
            plan = "Draft fallback plan"

        evidence = self.load_literature_evidence(index_path)
        system_prompt = self.load_prompt()
        
        print("Synthesizing Draft_v1.md via Gemini Pro...")

        try:
            if not self.llm:
                raise ValueError("LLM not initialized")

            # Prepare prompt content
            prompt = PromptTemplate.from_template(
                "{system}\n\nPLAN:\n{plan}\n\nEVIDENCE CARDS:\n{evidence}"
            )
            prompt_inputs = {
                "system": system_prompt,
                "plan": plan,
                "evidence": evidence,
            }

            # When running with the real LangChain chat model, we use the
            # standard Runnable composition. In tests, the LLM is replaced
            # with a light-weight fake object that only exposes `invoke`.
            if isinstance(self.llm, ChatGoogleGenerativeAI):
                chain = prompt | self.llm
                result = chain.invoke(prompt_inputs)
            else:
                rendered = prompt.format(**prompt_inputs)
                # Test fakes expect a dict, but they ignore its contents.
                payload = {"prompt": rendered}
                result = self.llm.invoke(payload)

            draft_content = getattr(result, "content", str(result))
        except Exception as e:
            print(f"Warning: draft generation failed ({e})")
            draft_content = "> This draft is AI-assisted.\n\n# Fallback Draft\nFailed to reach API."
            
        os.makedirs("output", exist_ok=True)
        with open(self.output_md, "w", encoding="utf-8") as f:
            f.write(draft_content)

        print(f"Draft saved to {self.output_md}")
        return {
            "execution_status": "fetching",
            "draft_content_path": self.output_md
        }

def drafter_node(state: ResearchState) -> ResearchState:
    agent = DrafterAgent()
    return agent.run(state)
