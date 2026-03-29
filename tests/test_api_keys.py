import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so model names (GEMINI_FAST_MODEL, GEMINI_PRO_MODEL) are used
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

def test_gemini():
    print("=== Testing Gemini API Key ===")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY / GOOGLE_API_KEY is not set in .env")
        return False
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Model names from .env (loaded above)
        fast_model = os.getenv("GEMINI_FAST_MODEL", "gemini-3.1-flash-lite-preview")
        pro_model = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
        models_to_test = [fast_model, pro_model]
        all_passed = True
        for model in models_to_test:
            print(f"Testing model: {model}...")
            llm = ChatGoogleGenerativeAI(model=model, temperature=0.0)
            response = llm.invoke("Hi, please reply with 'OK' and nothing else.")
            content_str = str(response.content)
            if "OK" in content_str.upper():
                print(f"✅ Model {model} is working!")
            else:
                print(f"⚠️ Model {model} responded, but output was unexpected: {content_str}")
                all_passed = False
        return all_passed
    except ImportError:
        print("❌ langchain_google_genai is not installed.")
        return False
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False


if __name__ == "__main__":

    gemini_ok = test_gemini()
    
    print("\n=== Final Report ===")
    if gemini_ok:
        print("🎉 Gemini API Key is working perfectly!")
    else:
        print("⚠️ Please check the errors above.")

