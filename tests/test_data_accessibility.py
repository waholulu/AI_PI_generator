from agents.data_accessibility import evaluate_data_sources, summarize_data_access
from models.research_plan_schema import (
    DataSourceSpec,
    FeasibilitySpec,
    IdentificationSpec,
    ResearchPlan,
    VariableSpec,
)


def _plan_with_sources(sources: list[DataSourceSpec]) -> ResearchPlan:
    return ResearchPlan(
        run_id="run-a",
        project_title="Air pollution and mortality",
        research_question="How does PM2.5 affect mortality?",
        short_rationale="Publicly available environmental and health datasets can be linked.",
        geography="US",
        time_window="2010-2020",
        exposure=VariableSpec(name="PM2.5", measurement_proxy="pm2.5"),
        outcome=VariableSpec(name="Mortality", measurement_proxy="mortality"),
        identification=IdentificationSpec(primary_method="fixed_effects", key_threats=["confounding"]),
        data_sources=sources,
        literature_queries=["a", "b", "c"],
        feasibility=FeasibilitySpec(overall_verdict="warning"),
    )


def test_data_accessibility_mock_reachable_url_returns_pass_or_warning() -> None:
    plan = _plan_with_sources(
        [
            DataSourceSpec(
                name="EPA PM2.5",
                access_url="https://example.org/pm25.csv",
                expected_format="csv",
                access_notes="pm2.5 annual county US 2010-2020",
            ),
            DataSourceSpec(
                name="CDC Mortality",
                access_url="https://example.org/mortality.csv",
                expected_format="csv",
                access_notes="mortality annual county US 2010-2020",
            ),
        ]
    )
    checks = evaluate_data_sources(plan)
    verdict, _ = summarize_data_access(checks)
    assert verdict in {"pass", "warning"}


def test_data_accessibility_empty_sources_returns_fail() -> None:
    plan = _plan_with_sources([])
    checks = evaluate_data_sources(plan)
    verdict, reasons = summarize_data_access(checks)
    assert verdict == "fail"
    assert "no_data_sources_declared" in reasons


def test_data_accessibility_registry_only_no_url_returns_warning_or_fail() -> None:
    plan = _plan_with_sources(
        [
            DataSourceSpec(
                name="US Census Bureau ACS",
                source_type="registry",
                access_url="",
                expected_format="csv",
                access_notes="demographics US county",
            )
        ]
    )
    checks = evaluate_data_sources(plan)
    assert checks[0].verdict in {"warning", "fail"}
