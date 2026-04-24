import re

from models.topic_schema import (
    ContributionPrimary,
    ExposureFamily,
    Frequency,
    IdentificationPrimary,
    OutcomeFamily,
    SamplingMode,
)


def _load(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _enum_values(enum_cls) -> set[str]:
    return {m.value for m in enum_cls}


def _extract_required_values(prompt_text: str, section_name: str) -> set[str]:
    pattern = rf"<one of {section_name}>:\s*([^\n]+)"
    m = re.search(pattern, prompt_text)
    if not m:
        return set()
    raw = m.group(1)
    return {part.strip() for part in raw.split("|") if part.strip()}


def test_ideation_seed_prompt_enum_alignment():
    text = _load("/workspace/prompts/ideation_seed.txt")
    assert _extract_required_values(text, "ExposureFamily") <= _enum_values(ExposureFamily)
    assert _extract_required_values(text, "OutcomeFamily") <= _enum_values(OutcomeFamily)
    assert _extract_required_values(text, "SamplingMode") <= _enum_values(SamplingMode)
    assert _extract_required_values(text, "Frequency") <= _enum_values(Frequency)
    assert _extract_required_values(text, "IdentificationPrimary") <= _enum_values(IdentificationPrimary)
    assert _extract_required_values(text, "ContributionPrimary") <= _enum_values(ContributionPrimary)


def test_reflection_refine_prompt_enum_alignment():
    text = _load("/workspace/prompts/reflection_refine.txt")
    assert _extract_required_values(text, "ExposureFamily") <= _enum_values(ExposureFamily)
    assert _extract_required_values(text, "OutcomeFamily") <= _enum_values(OutcomeFamily)
    assert _extract_required_values(text, "SamplingMode") <= _enum_values(SamplingMode)
    assert _extract_required_values(text, "Frequency") <= _enum_values(Frequency)
    assert _extract_required_values(text, "IdentificationPrimary") <= _enum_values(IdentificationPrimary)
    assert _extract_required_values(text, "ContributionPrimary") <= _enum_values(ContributionPrimary)


def test_reflection_critique_includes_openalex_top_papers_placeholder():
    text = _load("/workspace/prompts/reflection_critique.txt")
    assert "{oa_top_papers_formatted}" in text
