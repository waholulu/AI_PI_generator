from agents.rule_engine import RuleEngine


def test_g3_role_based_subchecks_pass_for_osmnx_places_stack() -> None:
    engine = RuleEngine()
    result = engine.check_G3_role_based_data_availability(
        exposure_family="street_connectivity",
        outcome_family="physical_inactivity",
        declared_sources=["OSMnx_OpenStreetMap", "CDC_PLACES", "TIGER_Lines"],
    )

    assert result.gate_id == "G3"
    assert result.details["subchecks"]["source_exists"] == "pass"
    assert result.details["subchecks"]["role_coverage"] == "pass"
    assert result.details["subchecks"]["machine_readable"] == "pass"
