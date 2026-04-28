import json
import os

from agents.ideation_agent_v2 import IdeationAgentV2
from models.research_plan_schema import ResearchPlan
from models.topic_schema import (
    Contribution,
    ContributionPrimary,
    ExposureFamily,
    ExposureX,
    Frequency,
    IdentificationPrimary,
    IdentificationStrategy,
    OutcomeFamily,
    OutcomeY,
    SamplingMode,
    SeedCandidate,
    SpatialScope,
    TemporalScope,
    Topic,
    TopicMeta,
)
from agents.literature_agent import LiteratureHarvester
from agents.drafter_agent import DrafterAgent
from agents.data_access_agent import DataAccessAgent


def _seed() -> SeedCandidate:
    topic = Topic(
        meta=TopicMeta(topic_id="seed-1"),
        exposure_X=ExposureX(family=ExposureFamily.AIR_QUALITY, specific_variable="PM2.5", spatial_unit="county"),
        outcome_Y=OutcomeY(family=OutcomeFamily.HEALTH, specific_variable="Mortality", spatial_unit="county"),
        spatial_scope=SpatialScope(geography="US", spatial_unit="county", sampling_mode=SamplingMode.PANEL),
        temporal_scope=TemporalScope(start_year=2010, end_year=2020, frequency=Frequency.ANNUAL),
        identification=IdentificationStrategy(
            primary=IdentificationPrimary.FE,
            key_threats=["confounding"],
            mitigations={"confounding": "unit and year FE"},
        ),
        contribution=Contribution(primary=ContributionPrimary.CAUSAL_REFINEMENT, statement="Policy-relevant estimate"),
        free_form_title="PM2.5 and Mortality in US Counties",
    )
    return SeedCandidate(topic=topic, declared_sources=["US Census Bureau ACS", "EPA AirData"])


def test_simple_ideation_outputs_contract_valid_research_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPI_IDEATION_MODE", "simple")

    agent = IdeationAgentV2()
    agent._generate_seeds = lambda *args, **kwargs: [_seed()]  # type: ignore[assignment]

    result = agent.run_level2({"domain_input": "Urban health"})
    assert "current_plan_path" in result
    plan_path = result["current_plan_path"]
    assert os.path.exists(plan_path)

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = ResearchPlan.model_validate(json.load(f))

    assert plan.project_title
    assert plan.research_question
    assert plan.exposure.name
    assert plan.outcome.name
    assert plan.identification.primary_method
    assert len(plan.literature_queries) >= 3
    assert len(plan.data_sources) >= 1
    assert plan.feasibility.overall_verdict in {"pass", "warning", "fail"}


def test_module_contracts_use_research_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPI_IDEATION_MODE", "simple")
    monkeypatch.setenv("ARXIV_SEARCH_ENABLED", "false")

    agent = IdeationAgentV2()
    agent._generate_seeds = lambda *args, **kwargs: [_seed()]  # type: ignore[assignment]
    result = agent.run_level2({"domain_input": "Urban health"})
    plan_path = result["current_plan_path"]

    harvester = LiteratureHarvester()
    lit_out = harvester.run({"current_plan_path": plan_path})
    assert lit_out["execution_status"] == "drafting"

    drafter = DrafterAgent()
    draft_out = drafter.run(
        {
            "current_plan_path": plan_path,
            "literature_inventory_path": lit_out.get("literature_inventory_path", ""),
        }
    )
    assert draft_out["execution_status"] == "fetching"

    data_out = DataAccessAgent().run({"current_plan_path": plan_path})
    assert data_out["execution_status"] == "completed"
