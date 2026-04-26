from agents.research_plan_builder import build_research_plan_from_candidate


def test_research_plan_sources_enriched_from_registry() -> None:
    candidate = {
        "candidate_id": "beh_001",
        "title": "Street network and inactivity",
        "research_question": "rq",
        "exposure_variable": "street_network",
        "outcome_variable": "physical_inactivity",
        "exposure_family": "street_network",
        "outcome_family": "physical_inactivity",
        "exposure_source": "OSMnx",
        "outcome_source": "CDC Places",
        "join_plan": {"controls": ["ACS"], "boundary_source": ["TIGER_Lines"]},
        "method": "cross_sectional_spatial_association",
        "key_threats": ["a", "b", "c"],
        "mitigations": {"a": "m1", "b": "m2", "c": "m3"},
    }

    plan = build_research_plan_from_candidate(candidate, evaluation=None, run_id="r1")
    by_name = {s.name: s for s in plan.data_sources}

    assert by_name["OSMnx_OpenStreetMap"].role == "exposure"
    assert by_name["CDC_PLACES"].role == "outcome"
    assert by_name["ACS"].role == "control"
    assert by_name["TIGER_Lines"].role == "boundary"

    assert by_name["OSMnx_OpenStreetMap"].machine_readable is True
    assert by_name["CDC_PLACES"].machine_readable is True
    assert by_name["ACS"].machine_readable is True
    assert by_name["TIGER_Lines"].machine_readable is True

    assert plan.exposure.family
    assert plan.outcome.family
