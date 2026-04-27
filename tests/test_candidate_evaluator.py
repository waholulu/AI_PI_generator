from agents.candidate_evaluator import evaluate_candidate
from models.research_plan_schema import (
    DataSourceSpec,
    FeasibilitySpec,
    IdentificationSpec,
    ResearchPlan,
    VariableSpec,
)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        run_id="r1",
        project_title="Air pollution and mortality",
        research_question="How does PM2.5 affect mortality?",
        short_rationale="Policy-relevant and measurable question with public data.",
        geography="US",
        time_window="2010-2020",
        exposure=VariableSpec(name="PM2.5", measurement_proxy="satellite PM2.5"),
        outcome=VariableSpec(name="Mortality", measurement_proxy="death rate"),
        identification=IdentificationSpec(primary_method="fixed_effects", key_threats=["confounding"]),
        data_sources=[
            DataSourceSpec(
                name="US Census Bureau ACS",
                source_type="api",
                access_url="https://data.census.gov",
                expected_format="csv",
                access_notes="US county panel with annual updates and demographics",
            ),
            DataSourceSpec(
                name="EPA AirData",
                source_type="download",
                access_url="https://www.epa.gov/outdoor-air-quality-data",
                expected_format="csv",
                access_notes="PM2.5 exposure data for US counties annual",
            ),
        ],
        literature_queries=["pm2.5 mortality fixed effects", "air pollution mortality US", "pm2.5 panel data"],
        feasibility=FeasibilitySpec(overall_verdict="warning"),
    )


def test_candidate_evaluator_returns_structured_evaluation() -> None:
    candidate = {"title": "Air pollution and mortality", "rank": 1, "topic_id": "c1"}
    evaluation = evaluate_candidate(candidate, _plan(), llm=None)
    assert evaluation.candidate_id == "c1"
    assert evaluation.title == "Air pollution and mortality"
    assert evaluation.overall_verdict in {"pass", "warning", "fail"}


def test_candidate_evaluator_no_data_sources_cannot_pass() -> None:
    plan = _plan().model_copy(update={"data_sources": []})
    evaluation = evaluate_candidate({"title": "x", "rank": 1}, plan, llm=None)
    assert evaluation.overall_verdict != "pass"


def test_candidate_evaluator_missing_identification_threats_warns() -> None:
    plan = _plan().model_copy(
        update={"identification": IdentificationSpec(primary_method="fixed_effects", key_threats=[])}
    )
    evaluation = evaluate_candidate({"title": "x", "rank": 1}, plan, llm=None)
    assert evaluation.identification_verdict == "warning"

