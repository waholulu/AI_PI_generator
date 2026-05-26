import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root so model/provider settings are used.
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

def check_llm_provider() -> bool:
    print("=== Testing configured LLM provider ===")
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        print("FAIL: No LLM API key is set in .env")
        return False

    try:
        from agents.llm import create_chat_model, get_model_name

        models_to_test = [get_model_name("fast"), get_model_name("pro")]
        all_passed = True
        for role, model in zip(("fast", "pro"), models_to_test):
            print(f"Testing {provider} model: {model}...")
            llm = create_chat_model(role, model=model, temperature=0.0)
            response = llm.invoke("Hi, please reply with 'OK' and nothing else.")
            content_str = str(response.content)
            if "OK" in content_str.upper():
                print(f"PASS: Model {model} is working!")
            else:
                print(f"WARN: Model {model} responded, but output was unexpected: {content_str}")
                all_passed = False
        return all_passed
    except ImportError:
        print("FAIL: Required LangChain provider package is not installed.")
        return False
    except Exception as e:
        print(f"FAIL: LLM API test failed: {e}")
        return False


@pytest.mark.live_llm
def test_llm_provider():
    assert check_llm_provider()


if __name__ == "__main__":

    llm_ok = check_llm_provider()
    
    print("\n=== Final Report ===")
    if llm_ok:
        print("PASS: LLM API key is working perfectly!")
    else:
        print("WARN: Please check the errors above.")

