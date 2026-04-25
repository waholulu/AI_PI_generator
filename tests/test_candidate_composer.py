from agents.candidate_composer import compose_candidates
from models.candidate_composer_schema import ComposeRequest


def test_candidate_composer_generates_candidates() -> None:
    req = ComposeRequest(
        template_id="built_environment_health",
        domain_input="Built environment and health",
        max_candidates=20,
        enable_experimental=False,
    )
    candidates = compose_candidates(req)

    assert len(candidates) >= 6
    assert all(c.exposure_source for c in candidates)
    assert all(c.outcome_source for c in candidates)
    assert all(c.method_template for c in candidates)


def test_candidate_composer_respects_experimental_toggle() -> None:
    req = ComposeRequest(
        template_id="built_environment_health",
        domain_input="Built environment and health",
        max_candidates=40,
        enable_experimental=False,
    )
    candidates = compose_candidates(req)

    assert all("streetview" not in c.exposure_family for c in candidates)
