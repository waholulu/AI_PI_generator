"""Central LLM factory for Auto-PI."""

from __future__ import annotations

import os
import json
from typing import Any


DEFAULT_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_GEMINI_FAST_MODEL = "gemini-2.0-flash-lite"
DEFAULT_GEMINI_PRO_MODEL = "gemini-2.5-pro"


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()


def get_model_name(role: str = "fast") -> str:
    role = role.strip().lower()
    if role in {"pro", "critique", "refine", "draft"}:
        generic_env = "LLM_PRO_MODEL"
        gemini_env = "GEMINI_PRO_MODEL"
        gemini_default = DEFAULT_GEMINI_PRO_MODEL
    else:
        generic_env = "LLM_FAST_MODEL"
        gemini_env = "GEMINI_FAST_MODEL"
        gemini_default = DEFAULT_GEMINI_FAST_MODEL

    provider = provider_name()
    if provider == "google":
        return os.getenv(generic_env) or os.getenv(gemini_env, gemini_default)
    if provider == "deepseek":
        return (
            os.getenv(generic_env)
            or os.getenv("DEEPSEEK_MODEL")
            or DEFAULT_DEEPSEEK_MODEL
        )
    return os.getenv(generic_env, DEFAULT_DEEPSEEK_MODEL)


def has_llm_credentials() -> bool:
    provider = provider_name()
    if provider == "google":
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if provider == "deepseek":
        return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY"))
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def create_chat_model(
    role: str = "fast",
    *,
    temperature: float = 0.1,
    model: str | None = None,
) -> Any:
    provider = provider_name()
    model_name = model or get_model_name(role)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    if provider in {"deepseek", "openai-compatible", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        api_key = (
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY or LLM_API_KEY is required for DeepSeek")
        base_url = os.getenv("LLM_BASE_URL") or os.getenv(
            "DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL
        )
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)

    raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}")


def with_structured_output(llm: Any, schema: Any) -> Any:
    """Return a structured-output wrapper compatible with the active provider."""
    if provider_name() in {"deepseek", "openai-compatible", "openai_compatible"}:
        try:
            return llm.with_structured_output(schema, method="function_calling")
        except TypeError:
            return llm.with_structured_output(schema)
    return llm.with_structured_output(schema)


def _is_openai_compatible_llm(llm: Any) -> bool:
    return llm.__class__.__module__.startswith("langchain_openai")


def _extract_json_text(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    return raw.strip()


def invoke_structured(llm: Any, schema: Any, prompt: Any) -> Any:
    """Invoke an LLM and validate the response against a Pydantic schema.

    DeepSeek/OpenAI-compatible endpoints may reject LangChain's structured
    response_format and tool_choice payloads. For real ChatOpenAI instances we
    ask for plain JSON and validate locally; for tests and non-OpenAI providers
    we preserve the provider-native structured-output path.
    """
    if _is_openai_compatible_llm(llm):
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True)
        json_prompt = (
            f"{prompt}\n\n"
            "Return only valid JSON, with no markdown fences or commentary. "
            "The JSON must conform to this schema:\n"
            f"{schema_json}"
        )
        response = llm.invoke(json_prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        data = json.loads(_extract_json_text(str(content)))
        return schema.model_validate(data)

    return with_structured_output(llm, schema).invoke(prompt)
