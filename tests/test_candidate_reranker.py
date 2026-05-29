from agents.candidate_reranker import rerank_candidates


def _card(candidate_id: str, tags: list[str], overall: float, readiness: str = "ready") -> dict:
    return {
        "candidate_id": candidate_id,
        "title": candidate_id,
        "display": {"display_title": f"How Does {candidate_id} Change Urban Outcomes?"},
        "exposure_label": candidate_id,
        "outcome_label": "physical_inactivity",
        "exposure_source": "source",
        "outcome_source": "CDC_PLACES",
        "unit_of_analysis": "census_tract",
        "method": "cross_sectional_spatial_association",
        "technology_tags": tags,
        "readiness": readiness,
        "shortlist_status": "ready" if readiness == "ready" else "review",
        "scores": {"overall": overall, "novelty": 0.55},
    }


def test_geoai_domain_boosts_geospatial_candidates() -> None:
    generic = _card("generic_walkability", [], 0.95)
    geoai = _card("satellite_land_use", ["remote_sensing"], 0.82)

    ranked = rerank_candidates(
        [generic, geoai],
        domain_input="GeoAI and Urban Planning",
        field_scan_summary={
            "search_strategy": {
                "methods": ["remote sensing", "spatial analysis"],
                "query_pool": ["urban land use classification satellite imagery"],
            }
        },
    )

    assert ranked[0]["candidate_id"] == "satellite_land_use"
    assert ranked[0]["rerank"]["domain_fit_score"] > ranked[1]["rerank"]["domain_fit_score"]
    assert ranked[0]["scores"]["overall"] == ranked[0]["rerank"]["rerank_score"]


def test_rerank_preserves_blocked_candidates_at_bottom() -> None:
    ready = _card("ready_candidate", [], 0.5)
    blocked = _card("blocked_candidate", ["remote_sensing"], 0.99, readiness="blocked")

    ranked = rerank_candidates([blocked, ready], domain_input="GeoAI")

    assert ranked[0]["candidate_id"] == "ready_candidate"
    assert ranked[-1]["candidate_id"] == "blocked_candidate"


def test_empirical_deepening_rewards_measurement_gain_over_generic_tech() -> None:
    common = _card("green_space", ["remote_sensing"], 0.88, readiness="needs_review")
    common["exposure_source"] = "NLCD"
    common["outcome_label"] = "asthma"

    richer = _card("nighttime_lights", ["remote_sensing"], 0.82, readiness="needs_review")
    richer["exposure_source"] = "VIIRS"
    richer["outcome_label"] = "cardiovascular_disease"

    ranked = rerank_candidates(
        [common, richer],
        domain_input="new technologies that deepen traditional empirical X to Y research",
    )

    assert ranked[0]["candidate_id"] == "nighttime_lights"
    assert ranked[0]["tech_lens_type"] == "better_measurement_of_x"
    assert "traditional nighttime light intensity -> cardiovascular disease" in ranked[0]["empirical_deepening_claim"]
    assert ranked[0]["rerank"]["empirical_value_score"] > ranked[1]["rerank"]["empirical_value_score"]
    assert ranked[0]["polished_title"].startswith("What Does Satellite-Measured")
