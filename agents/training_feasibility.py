"""LLM/DL training-research feasibility sub-checks.

Three deterministic, zero-LLM-cost checks layered on top of G3/G6:

  G3-train  license_in_whitelist
  G6-train  gpu_runtime_capacity
  G6-train  train_eval_leakage_plan_present

Each returns ``("pass" | "warning" | "fail", reason_or_None)``. Per the
agreed plan, gpu_runtime_capacity downgrades a candidate to *review*
(warning) rather than hard-blocking it — the user picks the runtime tier
at HITL.

These functions are pure: they take a ``ComposedCandidate`` and the
already-loaded template dict, and never touch the network. They are
called by ``agents.training_feasibility.evaluate_training_candidate``,
which the candidate factory invokes when
``template.get("kind") == "training_research"``.
"""
from __future__ import annotations

from typing import Any

from agents.logging_config import get_logger
from models.candidate_composer_schema import ComposedCandidate

logger = get_logger(__name__)


_DEFAULT_LICENSE_WHITELIST = (
    "apache-2.0",
    "mit",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "odc-by",
    "bsd-3-clause",
)

# Rough param counts (in billions) for base models referenced in v1.
# Used by gpu_runtime_capacity. Unknown models default to None → warning.
_BASE_MODEL_PARAM_BILLIONS: dict[str, float] = {
    "huggingfacetb/smollm2-135m-instruct": 0.135,
    "huggingfacetb/smollm2-360m-instruct": 0.360,
    "qwen/qwen2.5-0.5b": 0.5,
    "qwen/qwen2.5-1.5b": 1.5,
    "qwen/qwen2.5-3b": 3.0,
    "qwen/qwen2.5-7b": 7.0,
    "meta-llama/llama-3.2-1b": 1.0,
    "meta-llama/llama-3.2-3b": 3.0,
    "meta-llama/llama-3.1-8b": 8.0,
    "google/gemma-2-2b": 2.0,
    "mistralai/mistral-7b-v0.3": 7.0,
}


def _whitelist(template: dict[str, Any]) -> set[str]:
    """Return the license whitelist. Distinguishes absence (use default) from
    explicit empty list (treated as a configuration error by the caller)."""
    if "allowed_licenses" in template:
        raw = template.get("allowed_licenses") or []
    else:
        raw = _DEFAULT_LICENSE_WHITELIST
    return {str(s).strip().lower() for s in raw}


def _runtime_tiers(template: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return template.get("runtime_tiers") or {}


def check_license_in_whitelist(
    candidate: ComposedCandidate, template: dict[str, Any]
) -> tuple[str, str | None]:
    """Hard-fail when any declared license is outside the template whitelist.

    The composer doesn't yet pin specific (base_model, dataset) tuples per
    candidate — the experiment matrix is expanded by the dev-pack writer.
    At candidate-evaluation time we can only enforce that the *template's*
    declared whitelist is non-empty. The actual license inspection is
    deferred to the Colab notebook's preflight cell.
    """
    licenses = _whitelist(template)
    if not licenses:
        return (
            "fail",
            "license_whitelist_empty:training_template_must_declare_allowed_licenses",
        )
    return "pass", None


def _max_params_for_runtime(
    runtime: dict[str, Any], strategy: str
) -> float | None:
    """Return the runtime's max params (billions) for this training strategy."""
    if strategy == "sft_full_finetune":
        return runtime.get("max_param_billions_full_sft")
    if strategy == "lora_adapter":
        return runtime.get("max_param_billions_lora")
    if strategy == "qlora_4bit":
        return runtime.get("max_param_billions_qlora")
    return None


def check_gpu_runtime_capacity(
    candidate: ComposedCandidate,
    template: dict[str, Any],
    runtime_tier: str = "colab_t4",
) -> tuple[str, str | None]:
    """Warn when the *largest* base_model in the experiment_axes exceeds the
    runtime tier's capacity for this strategy.

    Per the approved plan, this never hard-blocks — it surfaces a
    `gpu_warning` reason that downgrades the candidate to the review
    shortlist. Users still see the candidate and can choose a beefier
    runtime tier.
    """
    runtimes = _runtime_tiers(template)
    runtime = runtimes.get(runtime_tier)
    if not runtime:
        return "warning", f"unknown_runtime_tier:{runtime_tier}"

    max_params = _max_params_for_runtime(runtime, candidate.exposure_family)
    if max_params is None:
        return "warning", f"strategy_unknown_for_runtime:{candidate.exposure_family}"

    base_models = (template.get("experiment_axes") or {}).get("base_model") or []
    sizes: list[tuple[str, float]] = []
    unknown: list[str] = []
    for m in base_models:
        size = _BASE_MODEL_PARAM_BILLIONS.get(m.lower())
        if size is None:
            unknown.append(m)
        else:
            sizes.append((m, size))

    if not sizes:
        return "warning", "all_base_models_have_unknown_param_counts"

    too_big = [(m, s) for (m, s) in sizes if s > max_params]
    if too_big:
        biggest = max(too_big, key=lambda t: t[1])
        return (
            "warning",
            (
                f"gpu_warning:base_model_{biggest[0]}_{biggest[1]:.1f}B"
                f"_exceeds_{runtime_tier}_{candidate.exposure_family}"
                f"_max_{max_params:.1f}B"
            ),
        )
    return "pass", None


def check_train_eval_leakage_plan(
    candidate: ComposedCandidate, template: dict[str, Any]
) -> tuple[str, str | None]:
    """Verify the candidate carries leakage threats and mitigations.

    The composer copies `key_threats` and `mitigations` from the template's
    `allowed_methods`. We require the threats list to mention
    `train_eval_leakage` and `eval_set_contamination`, and that both have
    mitigation strings. This is a structural check — the actual leakage
    test runs in the Colab notebook.
    """
    threats = set(candidate.key_threats or [])
    mitigations = set((candidate.mitigations or {}).keys())

    required = {"train_eval_leakage", "eval_set_contamination"}
    missing_threats = required - threats
    if missing_threats:
        return (
            "fail",
            f"missing_leakage_threats:{sorted(missing_threats)}",
        )
    missing_mit = required - mitigations
    if missing_mit:
        return (
            "fail",
            f"missing_leakage_mitigations:{sorted(missing_mit)}",
        )
    return "pass", None


def evaluate_training_candidate(
    candidate: ComposedCandidate,
    template: dict[str, Any],
    runtime_tier: str = "colab_t4",
) -> dict[str, Any]:
    """Run all three training sub-checks and return an aggregate dict.

    Returns:
        {
          "subchecks": {name: "pass" | "warning" | "fail"},
          "reasons":   [str],
          "overall":   "pass" | "warning" | "fail",
          "shortlist_status": "ready" | "review" | "blocked",
        }

    Mapping to overall:
        any fail   → blocked (hard-block; license whitelist or leakage gap)
        any warn   → review  (gpu_warning, unknown runtime, etc.)
        all pass   → ready
    """
    checks = {
        "license_in_whitelist": check_license_in_whitelist(candidate, template),
        "gpu_runtime_capacity": check_gpu_runtime_capacity(
            candidate, template, runtime_tier
        ),
        "train_eval_leakage_plan_present": check_train_eval_leakage_plan(
            candidate, template
        ),
    }
    subchecks = {name: status for name, (status, _) in checks.items()}
    reasons = [reason for _, (_, reason) in checks.items() if reason]

    if any(s == "fail" for s in subchecks.values()):
        overall, shortlist = "fail", "blocked"
    elif any(s == "warning" for s in subchecks.values()):
        overall, shortlist = "warning", "review"
    else:
        overall, shortlist = "pass", "ready"

    logger.debug(
        "training_feasibility candidate_id=%s overall=%s subchecks=%s",
        candidate.candidate_id, overall, subchecks,
    )

    return {
        "gate_id": "G3G6_training",
        "subchecks": subchecks,
        "reasons": reasons,
        "overall": overall,
        "shortlist_status": shortlist,
    }
