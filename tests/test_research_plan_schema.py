from models.research_plan_schema import (
    DataSourceSpec,
    FeasibilitySpec,
    IdentificationSpec,
    ResearchPlan,
    VariableSpec,
)


def test_research_plan_schema_parses_minimal_valid_payload() -> None:
    plan = ResearchPlan(
        run_id="run-1",
        project_title="PM2.5 and mortality",
        research_question="How does PM2.5 exposure affect mortality?",
        short_rationale="Air pollution remains a major public health burden.",
        exposure=VariableSpec(name="PM2.5"),
        outcome=VariableSpec(name="Mortality"),
        identification=IdentificationSpec(primary_method="fixed_effects"),
        data_sources=[DataSourceSpec(name="US Census Bureau ACS", source_type="api")],
        literature_queries=["pm2.5 mortality fixed effects", "air pollution mortality US", "pm2.5 public data"],
        feasibility=FeasibilitySpec(overall_verdict="warning"),
    )
    assert plan.project_title == "PM2.5 and mortality"
    assert plan.exposure.name == "PM2.5"
