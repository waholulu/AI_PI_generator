from __future__ import annotations

import pytest

from autopi_engine.contracts import NoveltyAssessment, SearchPlan


def test_search_plan_rejects_empty_queries() -> None:
    with pytest.raises(ValueError):
        SearchPlan(purpose="field scan", queries=["", "   "])


def test_novelty_assessment_contract_accepts_supported_verdicts() -> None:
    assessment = NoveltyAssessment(
        verdict="partially_overlapping",
        summary="Prior studies are close but not direct duplicates.",
    )
    assert assessment.recommended_action == "proceed"
