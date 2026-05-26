"""Tests for the llm_training_research v1 template, sources, composer wiring,
training feasibility gates, and the LLM-flavoured development pack writer.

These tests run without API keys (no live OpenAlex / no live LLM).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agents.candidate_composer import compose_candidates
from agents.candidate_feasibility import precheck_candidate
from agents.development_pack_writer import write_development_pack
from agents.development_pack_writer_llm import _expand_matrix, _strategy_specific_axes
from agents.research_template_loader import (
    load_research_template,
    validate_template_role_compatibility,
    validate_template_sources,
)
from agents.source_registry import SourceRegistry
from agents.training_feasibility import (
    check_gpu_runtime_capacity,
    check_license_in_whitelist,
    check_train_eval_leakage_plan,
    evaluate_training_candidate,
)
from models.candidate_composer_schema import ComposeRequest, ComposedCandidate


TEMPLATE_ID = "llm_training_research"


# ── Template & registry wiring ────────────────────────────────────────────────


def test_template_loads_and_declares_training_kind():
    t = load_research_template(TEMPLATE_ID)
    assert t["template_id"] == "llm_training_research_v1"
    assert t["kind"] == "training_research"
    assert t["default_unit_of_analysis"] == "training_run"


def test_template_sources_all_in_registry():
    t = load_research_template(TEMPLATE_ID)
    reg = SourceRegistry.load()
    assert validate_template_sources(t, reg) == []


def test_template_role_compatibility_clean():
    t = load_research_template(TEMPLATE_ID)
    reg = SourceRegistry.load()
    errors = validate_template_role_compatibility(t, reg)
    assert errors == [], f"role compatibility errors: {errors}"


def test_template_declares_three_v1_strategies_only():
    t = load_research_template(TEMPLATE_ID)
    fams = set(t["allowed_exposure_families"].keys())
    assert fams == {"sft_full_finetune", "lora_adapter", "qlora_4bit"}


def test_template_declares_license_whitelist_and_runtime_tiers():
    t = load_research_template(TEMPLATE_ID)
    assert "apache-2.0" in t["allowed_licenses"]
    assert "colab_t4" in t["runtime_tiers"]


def test_v1_excludes_rlhf_and_pretraining():
    """v1 must not surface DPO / RLHF / pretraining strategies."""
    t = load_research_template(TEMPLATE_ID)
    fams = set(t["allowed_exposure_families"].keys())
    forbidden = {"dpo", "rlhf", "rlaif", "pretraining", "continued_pretraining"}
    assert fams.isdisjoint(forbidden)


# ── Composer ──────────────────────────────────────────────────────────────────


def test_compose_candidates_produces_training_run_unit():
    req = ComposeRequest(
        template_id=TEMPLATE_ID,
        domain_input="LLM training comparisons",
        max_candidates=20,
        no_paid_api=True,
    )
    cands = compose_candidates(req)
    # 3 strategies × 5 outcomes = 15
    assert len(cands) >= 6
    for c in cands:
        assert c.unit_of_analysis == "training_run"
        assert c.exposure_family in {"sft_full_finetune", "lora_adapter", "qlora_4bit"}
        assert c.method_template in {
            "paired_run_baseline_vs_treatment",
            "ablation_sweep",
        }


def test_compose_candidates_carries_leakage_threats():
    req = ComposeRequest(
        template_id=TEMPLATE_ID,
        domain_input="LLM training",
        max_candidates=4,
    )
    cands = compose_candidates(req)
    for c in cands:
        assert "train_eval_leakage" in c.key_threats
        assert "eval_set_contamination" in c.key_threats
        assert "train_eval_leakage" in c.mitigations
        assert "eval_set_contamination" in c.mitigations


def test_compose_candidates_no_high_risk():
    req = ComposeRequest(
        template_id=TEMPLATE_ID,
        domain_input="LLM training",
        max_candidates=20,
        no_paid_api=True,
    )
    cands = compose_candidates(req)
    assert all(c.automation_risk in {"low", "medium"} for c in cands)


# ── Feasibility precheck ──────────────────────────────────────────────────────


def test_precheck_passes_for_v1_candidate():
    req = ComposeRequest(template_id=TEMPLATE_ID, domain_input="LLM", max_candidates=4)
    cands = compose_candidates(req)
    result = precheck_candidate(cands[0])
    assert result["overall"] != "fail", f"reasons: {result['reasons']}"


# ── Training feasibility helpers ──────────────────────────────────────────────


def _sample_candidate() -> ComposedCandidate:
    req = ComposeRequest(template_id=TEMPLATE_ID, domain_input="x", max_candidates=4)
    return compose_candidates(req)[0]


def test_license_check_passes_with_default_whitelist():
    t = load_research_template(TEMPLATE_ID)
    c = _sample_candidate()
    status, reason = check_license_in_whitelist(c, t)
    assert status == "pass"
    assert reason is None


def test_license_check_fails_when_whitelist_empty():
    t = load_research_template(TEMPLATE_ID)
    c = _sample_candidate()
    t = dict(t)
    t["allowed_licenses"] = []
    status, _ = check_license_in_whitelist(c, t)
    assert status == "fail"


def test_gpu_check_warns_for_sft_oversize_on_t4():
    t = load_research_template(TEMPLATE_ID)
    # SFT full finetune on T4 caps at 0.5B, but axes include 1.5B → warning
    cands = compose_candidates(
        ComposeRequest(template_id=TEMPLATE_ID, domain_input="x", max_candidates=20)
    )
    sft = next(c for c in cands if c.exposure_family == "sft_full_finetune")
    status, reason = check_gpu_runtime_capacity(sft, t, runtime_tier="colab_t4")
    assert status == "warning"
    assert "gpu_warning" in (reason or "")


def test_gpu_check_passes_for_lora_within_capacity():
    """LoRA cap is 1.5B on T4; axes include 0.135B and 0.5B which fit, but 1.5B
    is exactly at the cap. Largest is 1.5B → at the limit, not over."""
    t = load_research_template(TEMPLATE_ID)
    c = next(
        c for c in compose_candidates(
            ComposeRequest(template_id=TEMPLATE_ID, domain_input="x", max_candidates=20)
        )
        if c.exposure_family == "lora_adapter"
    )
    status, _ = check_gpu_runtime_capacity(c, t, runtime_tier="colab_t4")
    assert status == "pass"


def test_gpu_check_passes_for_qlora_well_under_capacity():
    t = load_research_template(TEMPLATE_ID)
    c = next(
        c for c in compose_candidates(
            ComposeRequest(template_id=TEMPLATE_ID, domain_input="x", max_candidates=20)
        )
        if c.exposure_family == "qlora_4bit"
    )
    status, _ = check_gpu_runtime_capacity(c, t, runtime_tier="colab_t4")
    assert status == "pass"


def test_leakage_plan_check_passes_when_threats_present():
    t = load_research_template(TEMPLATE_ID)
    c = _sample_candidate()
    status, _ = check_train_eval_leakage_plan(c, t)
    assert status == "pass"


def test_leakage_plan_check_fails_when_threats_missing():
    t = load_research_template(TEMPLATE_ID)
    c = _sample_candidate()
    c.key_threats = []
    status, reason = check_train_eval_leakage_plan(c, t)
    assert status == "fail"
    assert "missing_leakage_threats" in (reason or "")


def test_evaluate_training_candidate_aggregates():
    t = load_research_template(TEMPLATE_ID)
    cands = compose_candidates(
        ComposeRequest(template_id=TEMPLATE_ID, domain_input="x", max_candidates=20)
    )
    sft = next(c for c in cands if c.exposure_family == "sft_full_finetune")
    out = evaluate_training_candidate(sft, t, runtime_tier="colab_t4")
    # SFT on T4 with 1.5B model → at least one warning, never a fail
    assert out["overall"] in {"pass", "warning"}
    assert out["shortlist_status"] in {"ready", "review"}


# ── Dev pack writer ───────────────────────────────────────────────────────────


def test_dev_pack_writer_routes_training_candidates_to_llm_writer(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    cands = compose_candidates(
        ComposeRequest(template_id=TEMPLATE_ID, domain_input="LLM", max_candidates=4)
    )
    payload = cands[0].model_dump()
    pack_dir = write_development_pack("test_run", payload)

    expected = {
        "README.md",
        "experiment_config.yaml",
        "training_plan.yaml",
        "evaluation_plan.yaml",
        "colab_notebook_spec.md",
        "claude_task_prompt.md",
        "acceptance_tests.md",
        "implementation_spec.json",
    }
    found = {p.name for p in Path(pack_dir).iterdir()}
    assert expected <= found, f"missing: {expected - found}"


def test_experiment_config_has_smoke_run_and_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOPI_DATA_ROOT", str(tmp_path))
    cands = compose_candidates(
        ComposeRequest(template_id=TEMPLATE_ID, domain_input="LLM", max_candidates=4)
    )
    payload = cands[0].model_dump()
    pack_dir = write_development_pack("test_run2", payload)
    text = (Path(pack_dir) / "experiment_config.yaml").read_text(encoding="utf-8")
    assert "smoke_run:" in text
    assert "experiment_matrix:" in text
    assert "matrix_size:" in text
    assert "train_samples: 500" in text
    assert "eval_samples: 200" in text


def test_strategy_specific_axes_drop_lora_for_full_sft():
    t = load_research_template(TEMPLATE_ID)
    sft_axes = _strategy_specific_axes(t, "sft_full_finetune")
    assert "lora_rank" not in sft_axes
    assert "base_model" in sft_axes
    lora_axes = _strategy_specific_axes(t, "lora_adapter")
    assert "lora_rank" in lora_axes
    qlora_axes = _strategy_specific_axes(t, "qlora_4bit")
    assert "quantization_dtype" in qlora_axes


def test_expand_matrix_round_robin_covers_first_axis_levels():
    """Cap=4 with a 3-level base_model axis must include all 3 levels."""
    axes = {
        "base_model": ["A", "B", "C"],
        "dataset": ["d1"],
        "seed": [1, 2],
    }
    rows = _expand_matrix(axes, cap=4)
    base_models = {r["base_model"] for r in rows}
    assert base_models == {"A", "B", "C"}


# ── Domain-grounded task seeding (v1.1) ──────────────────────────────────────


def test_fallback_task_seeds_cover_three_metric_families():
    """Without an LLM API key the generator returns one task per metric family,
    so the pool stays large enough to review without an LLM."""
    from agents.task_seed_generator import _fallback_seeds

    seeds = _fallback_seeds("Built environment exposure and health outcomes")
    families = {s.metric_family for s in seeds}
    assert families == {"task_accuracy", "instruction_following", "generation_quality"}
    # Each fallback task should reference the user domain so the UI shows it
    assert all("built environment" in s.task_label.lower() for s in seeds)


def test_compose_attaches_task_to_training_candidate():
    """Training_research candidates must carry the task metadata so the
    display formatter can drop the built-environment language."""
    cands = compose_candidates(
        ComposeRequest(
            template_id=TEMPLATE_ID,
            domain_input="Built environment exposure and health outcomes",
            max_candidates=20,
        )
    )
    assert cands, "expected at least one candidate from training_research path"
    for c in cands:
        assert c.outcome_task_id is not None
        assert c.outcome_task_label
        assert c.outcome_task_modality in {
            "text_classification", "sequence_labeling", "extraction",
            "generation", "summarization", "question_answering",
        }
        # outcome_family is now the metric family (one of 3), not the task name
        assert c.outcome_family in {
            "task_accuracy", "instruction_following", "generation_quality",
        }


def test_training_card_drops_built_environment_language():
    """The bug we're fixing: the display card for LLM candidates must not
    mention sociodemographic confounders, US census tracts, or small-area
    variation in built-environment features."""
    from agents.candidate_factory_ideation import format_display_card

    cands = compose_candidates(
        ComposeRequest(
            template_id=TEMPLATE_ID,
            domain_input="Built environment exposure and health outcomes",
            max_candidates=4,
        )
    )
    card = format_display_card(cands[0])
    blob = " ".join(
        str(card.get(k, "")) for k in
        ("display_title", "research_question", "rationale", "contribution_angle", "execution_summary")
    ).lower()

    for forbidden in (
        "sociodemographic confounders",
        "small-area variation",
        "us census tract",
        "policy-actionable lens",
    ):
        assert forbidden not in blob, f"display still contains {forbidden!r}: {blob}"


def test_screening_entry_geography_for_training_is_not_us():
    """The geography field on a training_research screening entry must not
    claim 'United States' — the unit of analysis is training_run."""
    from agents.candidate_factory_ideation import (
        _card_to_screening_entry,
        _to_card,
        format_display_card,
    )

    cands = compose_candidates(
        ComposeRequest(
            template_id=TEMPLATE_ID,
            domain_input="Healthcare claims analysis",
            max_candidates=4,
        )
    )
    c = cands[0]
    card = _to_card(c, "test_title", "test_rq", display=format_display_card(c))
    entry = _card_to_screening_entry(card)
    assert entry["geography"] == "model_release"
    assert entry["unit_of_analysis"] == "training_run"
    assert entry["outcome_task_label"] == c.outcome_task_label


# ── Regression: spatial template still works ─────────────────────────────────


def test_built_environment_template_still_loads():
    t = load_research_template("built_environment_health")
    assert t.get("kind") != "training_research"
    assert t["default_unit_of_analysis"] == "census_tract"


def test_built_environment_compose_unaffected():
    cands = compose_candidates(
        ComposeRequest(
            template_id="built_environment_health",
            domain_input="built env health",
            max_candidates=8,
        )
    )
    assert len(cands) >= 4
    for c in cands:
        assert c.unit_of_analysis == "census_tract"
