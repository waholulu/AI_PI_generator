from agents.research_template_loader import load_research_template, validate_template_sources


def test_template_loads() -> None:
    template = load_research_template("built_environment_health")

    assert template["template_id"] == "built_environment_health_v1"
    assert "allowed_exposure_families" in template


def test_template_sources_exist_in_registry() -> None:
    template = load_research_template("built_environment_health")
    missing = validate_template_sources(template)

    assert missing == []
