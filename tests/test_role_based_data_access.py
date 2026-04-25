from agents.data_accessibility import evaluate_data_sources, summarize_data_access
from models.research_plan_schema import (
    DataSourceSpec,
    FeasibilitySpec,
    IdentificationSpec,
    ResearchPlan,
    VariableSpec,
)


def test_role_based_data_access_passes_for_exposure_outcome_boundary() -> None:
    plan = ResearchPlan(
        run_id="r1",
        project_title="X to Y",
        research_question="q",
        short_rationale="r",
        exposure=VariableSpec(name="street_connectivity"),
        outcome=VariableSpec(name="physical_inactivity"),
        identification=IdentificationSpec(primary_method="cross_sectional"),
        feasibility=FeasibilitySpec(),
        data_sources=[
            DataSourceSpec(
                name="OSMnx_OpenStreetMap",
                role="exposure",
                source_type="api",
                access_url="https://example.org/osm",
                machine_readable=True,
                covers_variable_families=["street_connectivity"],
            ),
            DataSourceSpec(
                name="CDC_PLACES",
                role="outcome",
                source_type="download",
                access_url="https://example.org/places.csv",
                machine_readable=True,
                covers_variable_families=["physical_inactivity"],
            ),
            DataSourceSpec(
                name="TIGER_Lines",
                role="boundary",
                source_type="download",
                access_url="https://example.org/tiger.zip",
                machine_readable=True,
                join_keys=["GEOID"],
            ),
        ],
    )

    checks = evaluate_data_sources(plan)
    verdict, reasons = summarize_data_access(checks)

    assert verdict == "pass"
    assert reasons == []
