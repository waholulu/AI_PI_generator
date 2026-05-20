"""LLM/DL training research development-pack writer (v1).

Activated when a candidate's `unit_of_analysis == "training_run"` (or its
template declares `kind: training_research`). Produces a Colab-ready pack:

  experiment_config.yaml   — full experiment matrix expanded from the
                              template's `experiment_axes`, scoped to this
                              candidate's strategy/outcome pair, with a
                              smoke_run section for the 200/1000-sample loop
  training_plan.yaml       — base_model, dataset, tokenizer, optimizer,
                              LoRA / quantization config, secret/license
                              checklist, checkpoint policy
  evaluation_plan.yaml     — paired baseline (zero-shot or base model)
                              vs treatment (trained adapter), harness
                              tasks, metrics, train/eval leakage check
  colab_notebook_spec.md   — cell-by-cell notebook scaffold the user can
                              paste into a fresh Colab notebook

The spatial-research writer in `agents/development_pack_writer.py` is left
untouched; the dispatcher there branches on unit_of_analysis.
"""
from __future__ import annotations

import itertools
import json
import textwrap
from pathlib import Path
from typing import Any

import yaml

from agents import settings
from agents.logging_config import get_logger
from agents.research_template_loader import load_research_template
from models.candidate_composer_schema import ComposedCandidate

logger = get_logger(__name__)


_DEFAULT_LICENSE_WHITELIST = [
    "apache-2.0",
    "mit",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "odc-by",
    "bsd-3-clause",
]

_DEFAULT_RUNTIME = "colab_t4"

_BASE_TEMPLATE_ID = "llm_training_research"


# ── helpers ───────────────────────────────────────────────────────────────────


def _to_candidate(candidate: dict[str, Any]) -> ComposedCandidate:
    return ComposedCandidate(**candidate)


def _safe_load_template(template_id: str) -> dict[str, Any]:
    """Load the template, tolerating versioned `template_id` fields.

    `ComposedCandidate.template_id` carries the value of the YAML's top-level
    `template_id:` key (e.g. `llm_training_research_v1`), but the file on
    disk is named after the unversioned id (e.g. `llm_training_research`).
    We try the literal id first, then strip a trailing `_v<digits>` suffix,
    then fall back to `_BASE_TEMPLATE_ID`.
    """
    import re

    candidates = [template_id]
    stripped = re.sub(r"_v\d+$", "", template_id)
    if stripped != template_id:
        candidates.append(stripped)
    if _BASE_TEMPLATE_ID not in candidates:
        candidates.append(_BASE_TEMPLATE_ID)

    for cand in candidates:
        try:
            return load_research_template(cand)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Could not load any of: {candidates}"
    )


def _yaml_dump(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def _strategy_specific_axes(template: dict[str, Any], strategy: str) -> dict[str, list]:
    """Return only the experiment_axes relevant to this strategy.

    sft_full_finetune drops lora_rank/lora_alpha; lora_adapter and qlora_4bit
    keep them. quantization_dtype is added for qlora_4bit.
    """
    base = dict(template.get("experiment_axes") or {})
    axes: dict[str, list] = {}

    # Always include
    for k in ("base_model", "dataset", "batch_size", "seed", "eval_metric"):
        if k in base:
            axes[k] = list(base[k])

    if strategy in ("lora_adapter", "qlora_4bit"):
        for k in ("lora_rank", "lora_alpha"):
            if k in base:
                axes[k] = list(base[k])

    if strategy == "qlora_4bit":
        axes["quantization_dtype"] = ["nf4"]

    return axes


def _expand_matrix(axes: dict[str, list], cap: int = 24) -> list[dict[str, Any]]:
    """Cartesian product, capped via stratified sampling.

    A naive `cap` on `itertools.product` only iterates the leading axis, so
    the user never sees a different base_model. Instead we round-robin
    indices across all axes — each row picks index `i` modulo |axis|, which
    cycles through every level of every axis at least once when cap ≥
    max(|axis|).
    """
    if not axes:
        return []
    keys = list(axes.keys())
    values = [axes[k] for k in keys]

    full_size = 1
    for v in values:
        full_size *= max(len(v), 1)
    take = min(cap, full_size)

    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    # Pass 1 — round-robin (gives axis coverage on small caps)
    for i in range(take):
        row = {keys[j]: values[j][i % len(values[j])] for j in range(len(keys))}
        key = tuple(row.values())
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    # Pass 2 — fill remaining slots from the start of the Cartesian product
    if len(rows) < take:
        for combo in itertools.product(*values):
            if combo in seen:
                continue
            seen.add(combo)
            rows.append(dict(zip(keys, combo)))
            if len(rows) >= take:
                break
    return rows


# ── 1. experiment_config.yaml ────────────────────────────────────────────────


def _experiment_config(
    candidate: ComposedCandidate, template: dict[str, Any]
) -> str:
    axes = _strategy_specific_axes(template, candidate.exposure_family)
    matrix = _expand_matrix(axes)
    smoke = template.get("smoke_test", {}) or {}

    payload: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "template_id": candidate.template_id,
        "training_strategy": candidate.exposure_family,
        "evaluation_target": candidate.outcome_family,
        "method": candidate.method_template,
        "claim_strength": candidate.claim_strength,
        "axes": axes,
        "experiment_matrix": matrix,
        "matrix_size": len(matrix),
        "smoke_run": {
            "enabled": True,
            "train_samples": smoke.get("train_samples", 500),
            "eval_samples": smoke.get("eval_samples", 200),
            "max_steps": smoke.get("max_steps", 100),
            "runtime_minutes_max": smoke.get("runtime_minutes_max", 15),
            "purpose": (
                "Prove the data → tokenizer → trainer → eval loop closes "
                "before launching any full run."
            ),
        },
        "runtime": {
            "default_tier": _DEFAULT_RUNTIME,
            "available_tiers": list((template.get("runtime_tiers") or {}).keys()),
        },
        "tracking": {
            "wandb_anonymous": True,
            "log_every_n_steps": 10,
            "eval_every_n_steps": 50,
        },
    }
    if candidate.outcome_task_id:
        payload["domain_task"] = {
            "task_id": candidate.outcome_task_id,
            "task_label": candidate.outcome_task_label,
            "task_description": candidate.outcome_task_description,
            "modality": candidate.outcome_task_modality,
            "dataset_hint": candidate.outcome_task_dataset_hint,
            "user_domain_input": candidate.outcome_task_domain_input,
            "note": (
                "Replace the placeholder dataset / benchmark choices with "
                "your domain-specific data before flipping SMOKE=False."
            ),
        }
    return _yaml_dump(payload)


# ── 2. training_plan.yaml ────────────────────────────────────────────────────


def _training_plan(
    candidate: ComposedCandidate, template: dict[str, Any]
) -> str:
    strategy = candidate.exposure_family
    secrets = list(candidate.required_secrets) or []
    if not any("HuggingFace_Models" in s for s in secrets):
        secrets.append("HuggingFace_Models:hf_token")

    license_whitelist = template.get("allowed_licenses") or _DEFAULT_LICENSE_WHITELIST

    plan: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "training_strategy": strategy,
        "data_sources": {
            "exposure": candidate.exposure_source,
            "outcome": candidate.outcome_source,
            "controls": candidate.join_plan.get("controls", []),
            "boundary": candidate.join_plan.get("boundary_source", []),
        },
        "tokenizer": {
            "load_from": "${base_model}",
            "trust_remote_code": False,
            "padding_side": "right",
        },
        "model_loading": {},
        "trainer": {
            "library": "trl.SFTTrainer",
            "max_seq_length": 1024,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "bf16": True,
            "optim": "adamw_torch",
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "save_strategy": "steps",
            "save_steps": 200,
            "save_total_limit": 2,
            "logging_steps": 10,
            "report_to": "wandb",
        },
        "checkpoint_policy": {
            "save_to": "/content/drive/MyDrive/checkpoints/${candidate_id}/${run_id}",
            "save_only_adapters": strategy != "sft_full_finetune",
            "resume_supported": True,
        },
        "preflight_checks": [
            "license_in_whitelist",
            "vram_estimate_within_runtime_tier",
            "smoke_run_max_steps_reached_without_nan",
            "tokenizer_matches_base_model",
        ],
        "license_whitelist": license_whitelist,
        "required_secrets": secrets,
    }

    if strategy == "sft_full_finetune":
        plan["model_loading"] = {
            "library": "transformers.AutoModelForCausalLM",
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
        }
    elif strategy == "lora_adapter":
        plan["model_loading"] = {
            "library": "transformers.AutoModelForCausalLM",
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
        }
        plan["lora"] = {
            "library": "peft.LoraConfig",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
        }
    elif strategy == "qlora_4bit":
        plan["model_loading"] = {
            "library": "transformers.AutoModelForCausalLM",
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "quantization_config": {
                "library": "transformers.BitsAndBytesConfig",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
            },
        }
        plan["lora"] = {
            "library": "peft.LoraConfig",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
        }

    return _yaml_dump(plan)


# ── 3. evaluation_plan.yaml ──────────────────────────────────────────────────


def _evaluation_plan(
    candidate: ComposedCandidate, template: dict[str, Any]
) -> str:
    outcome_spec = (template.get("allowed_outcome_families") or {}).get(
        candidate.outcome_family, {}
    )
    metrics = list(outcome_spec.get("metrics") or [candidate.outcome_family])
    methods = template.get("allowed_methods") or {}
    method_spec = methods.get(candidate.method_template, {})

    plan: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "evaluation_target": candidate.outcome_family,
        "domain_task": {
            "task_id": candidate.outcome_task_id,
            "task_label": candidate.outcome_task_label,
            "task_description": candidate.outcome_task_description,
            "modality": candidate.outcome_task_modality,
            "dataset_hint": candidate.outcome_task_dataset_hint,
        } if candidate.outcome_task_id else None,
        "harness": "lm-evaluation-harness",
        "harness_pinned_version": ">=0.4.4,<0.5",
        "paired_protocol": {
            "baseline": method_spec.get("baseline", "zero_shot_or_base_model"),
            "treatment": method_spec.get("treatment", "trained_adapter_or_finetune"),
            "comparison_seed_count": 3,
            "comparison_metric_aggregation": "mean ± std across seeds",
        },
        "tasks": metrics,
        "leakage_checks": [
            {
                "check_id": "train_eval_dataset_disjoint",
                "description": (
                    "Verify training corpus and evaluation benchmarks share "
                    "no row identifiers (hash prompt strings if no canonical id)."
                ),
                "must_pass_before_full_run": True,
            },
            {
                "check_id": "eval_release_after_base_model_cutoff",
                "description": (
                    "Prefer benchmarks released after base model knowledge "
                    "cutoff to mitigate pretraining contamination."
                ),
                "must_pass_before_full_run": False,
            },
            {
                "check_id": "tokenizer_matches_base_model",
                "description": "Tokenizer loaded from same checkpoint as weights.",
                "must_pass_before_full_run": True,
            },
        ],
        "report": {
            "primary_metrics": metrics,
            "report_format": "markdown",
            "include": [
                "baseline_vs_treatment_table",
                "per_seed_results",
                "wandb_run_links",
                "license_attestation",
                "leakage_check_results",
            ],
        },
        "expected_outputs": [
            "data/eval/${run_id}/results.json",
            "data/eval/${run_id}/baseline_vs_treatment.csv",
            "output/report/${candidate_id}_evaluation_summary.md",
        ],
    }
    return _yaml_dump(plan)


# ── 4. colab_notebook_spec.md ────────────────────────────────────────────────


def _colab_notebook_spec(
    candidate: ComposedCandidate, template: dict[str, Any]
) -> str:
    strategy = candidate.exposure_family
    smoke = template.get("smoke_test", {}) or {}
    train_samples = smoke.get("train_samples", 500)
    eval_samples = smoke.get("eval_samples", 200)
    max_steps = smoke.get("max_steps", 100)

    runtime_section = textwrap.dedent(
        """
        ## Runtime selection

        - **Default**: Colab T4 (free tier, 16 GB VRAM) — fits SFT for ≤0.5B
          base models, LoRA for ≤1.5B, QLoRA for ≤7B.
        - **Upgrade**: A100 (Colab Pro) when `gpu_warning` is set on the
          candidate card.
        - **Local GPU**: set `runtime: local_gpu` and skip the Colab-only
          cells (drive mount, secrets via `userdata`).
        """
    ).strip()

    qlora_extras = ""
    if strategy == "qlora_4bit":
        qlora_extras = (
            "  - Install `bitsandbytes` (pinned to a release that supports "
            "your CUDA major version)\n"
        )

    return textwrap.dedent(
        f"""
        # Colab Notebook Spec — {candidate.candidate_id}

        | Field | Value |
        |-------|-------|
        | Strategy | `{strategy}` |
        | Outcome | `{candidate.outcome_family}` |
        | Method | `{candidate.method_template}` |
        | Cloud safe | `{'Yes' if candidate.cloud_safe else 'No'}` |
        | Automation risk | `{candidate.automation_risk}` |

        {runtime_section}

        ## Notebook structure (12 cells)

        **Cell 1 — Environment**
          - `!pip install -q -U "transformers>=4.45,<4.50" "datasets>=2.20" "peft>=0.12" "trl>=0.10" "accelerate>=0.34" "wandb>=0.18" "lm-eval>=0.4.4,<0.5"`
        {qlora_extras}  - Verify GPU: `!nvidia-smi`

        **Cell 2 — Secrets & drive**
          - `from google.colab import userdata`
          - `os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")`
          - `from google.colab import drive; drive.mount("/content/drive")`

        **Cell 3 — Load experiment_config.yaml**
          - Read `experiment_config.yaml` from this dev pack.
          - Pick a single matrix row to start; full sweep is `for row in matrix`.
          - Set `SMOKE = True` initially.

        **Cell 4 — License preflight**
          - Inspect base_model and dataset cards via `huggingface_hub`.
          - Assert `license in allowed_licenses` (from `training_plan.yaml`).
          - Bail with a clear message if not.

        **Cell 5 — Data load**
          - `ds = load_dataset(dataset_id, split="train")`
          - If `SMOKE`: `ds_train = ds.select(range(min({train_samples}, len(ds))))`
          - Hold out `ds_eval = ds.select(range(len(ds_train), len(ds_train)+{eval_samples}))`
          - Hash prompts on both splits; assert disjoint.

        **Cell 6 — Tokenizer & model load**
          - Tokenizer from `base_model`; verify chat template if instruction tuning.
          - Model load per `training_plan.yaml.model_loading` (sft / lora / qlora).
          - For LoRA: `peft.get_peft_model(model, peft.LoraConfig(...))`.
          - Print `model.print_trainable_parameters()`.

        **Cell 7 — Smoke training**
          - `TrainingArguments(max_steps={max_steps}, ...)`
          - Run `trainer.train()`; expect loss to decrease and finish < 15 min.
          - Save adapter / model to drive.

        **Cell 8 — Smoke eval**
          - Run lm-eval-harness on a *subset* of the configured tasks.
          - Compare baseline (base model) vs treatment (trained adapter).
          - Assert metrics finite and treatment ≠ baseline (non-trivial change).

        **Cell 9 — Leakage check**
          - Re-run the train/eval prompt-hash disjoint assertion on the harness inputs.
          - Log result to wandb.

        **Cell 10 — Flip to full run**
          - Set `SMOKE = False`.
          - Iterate over `experiment_matrix` (or a slice you can afford).
          - Each row becomes an independent run with its own wandb run id.

        **Cell 11 — Aggregate results**
          - Pull all wandb runs for this candidate.
          - Build `baseline_vs_treatment.csv` and `evaluation_summary.md`.

        **Cell 12 — Cleanup**
          - Free GPU: `del model, trainer; torch.cuda.empty_cache()`.
          - Optionally upload the adapter to a private HF repo.

        ## Hard requirements before any full run

        1. Smoke cell (7) finished without NaN, in under
           `runtime_minutes_max` of `smoke_test`.
        2. License preflight (cell 4) passed.
        3. Leakage check (cell 9) returned disjoint.
        4. wandb run id captured for both baseline and treatment.
        """
    ).strip()


# ── 5. claude task prompt (training-flavoured) ───────────────────────────────


def _claude_task_prompt(
    candidate: ComposedCandidate, template: dict[str, Any]
) -> str:
    smoke = template.get("smoke_test", {}) or {}
    return textwrap.dedent(
        f"""
        # Implementation task — {candidate.candidate_id}

        You are implementing a paired baseline-vs-treatment LLM training
        experiment. Read the four planning files in this directory before
        writing any code:

        - `experiment_config.yaml` — axes and matrix
        - `training_plan.yaml` — model loading, trainer, secrets, license
        - `evaluation_plan.yaml` — paired protocol, harness tasks, leakage
        - `colab_notebook_spec.md` — cell-by-cell scaffold

        ## Goal

        Compare {candidate.exposure_family} (treatment) against the
        zero-shot / base-model baseline on the task
        `{candidate.outcome_task_label or candidate.outcome_family}`
        (metric family: `{candidate.outcome_family}`). Claim strength is
        `{candidate.claim_strength}`.
        {('Task description: ' + candidate.outcome_task_description) if candidate.outcome_task_description else ''}
        {('Suggested data: ' + candidate.outcome_task_dataset_hint) if candidate.outcome_task_dataset_hint else ''}

        ## Key constraints

        - **Smoke first.** Run a {smoke.get('train_samples', 500)}-train /
          {smoke.get('eval_samples', 200)}-eval / {smoke.get('max_steps', 100)}-step
          smoke pass before any full run.
        - **License preflight is mandatory.** Refuse to train if the base
          model or dataset license is not in the whitelist.
        - **Pin versions.** transformers, peft, bitsandbytes, lm-eval —
          all pinned in requirements.txt.
        - **Leakage check.** Hash prompt strings to assert train/eval splits
          are disjoint at the row level.
        - **Three seeds for treatment.** Report mean ± std.

        ## Deliverables

        1. `train.py` — runs one matrix row given `--config` + `--row-id`.
        2. `eval.py` — runs lm-eval-harness on a saved adapter or model.
        3. `requirements.txt` — pinned versions.
        4. `results/baseline_vs_treatment.csv` after the smoke pass.
        5. `report.md` summarising smoke results, deferred to user before
           launching the full sweep.

        ## What you must NOT do

        - Do not run the full matrix without an explicit human go-ahead.
        - Do not skip the license preflight.
        - Do not download gated weights without HF_TOKEN.
        - Do not remove the leakage check.
        """
    ).strip()


# ── 6. acceptance_tests.md (training-flavoured) ──────────────────────────────


def _acceptance_tests(candidate: ComposedCandidate) -> str:
    return textwrap.dedent(
        f"""
        # Acceptance tests — {candidate.candidate_id}

        ## Smoke run must pass

        - [ ] `train.py --smoke` finishes without NaN train_loss
        - [ ] Smoke run wall-clock under `runtime_minutes_max`
        - [ ] Adapter / model checkpoint saved to drive (or local cache)
        - [ ] `eval.py --smoke` returns finite metrics for every task

        ## Paired comparison sanity

        - [ ] Baseline metrics computed against the *same* harness version
              and task set as treatment
        - [ ] Treatment metrics differ from baseline by more than
              `2 × stderr` on at least one primary task
              (otherwise: investigate, do not promote to full run)
        - [ ] Three seeds for treatment; mean ± std reported

        ## Hygiene

        - [ ] `requirements.txt` pins exact versions for
              transformers / peft / trl / lm-eval / bitsandbytes
        - [ ] License preflight rejects datasets / models outside the
              whitelist
        - [ ] Train / eval prompt hashes disjoint
        - [ ] wandb run ids captured for baseline and treatment
        - [ ] No raw HF_TOKEN written to disk or notebook output

        ## Reproducibility

        - [ ] Seed set on Python, NumPy, torch, and Trainer
        - [ ] base_model and dataset revisions pinned (not `main`)
        - [ ] Final report links to the wandb runs and lists exact
              package versions
        """
    ).strip()


# ── 7. README.md ─────────────────────────────────────────────────────────────


def _readme(candidate: ComposedCandidate) -> str:
    task_row = ""
    if candidate.outcome_task_label:
        task_row = (
            f"        | Domain task | {candidate.outcome_task_label} |\n"
            f"        | Task modality | {candidate.outcome_task_modality or 'n/a'} |\n"
            f"        | Metric family | {candidate.outcome_family} |\n"
        )
    return textwrap.dedent(
        f"""
        # Development Pack: {candidate.candidate_id} (LLM training)

        | Field | Value |
        |-------|-------|
        | Strategy (X) | {candidate.exposure_family} |
        | Outcome (Y) | {candidate.outcome_task_label or candidate.outcome_family} |
{task_row}        | Method | {candidate.method_template} |
        | Unit of analysis | {candidate.unit_of_analysis} |
        | Automation risk | {candidate.automation_risk} |
        | Cloud safe | {'Yes' if candidate.cloud_safe else 'No'} |

        ## Files

        | File | Purpose |
        |------|---------|
        | `experiment_config.yaml` | Axes + Cartesian matrix + smoke parameters |
        | `training_plan.yaml` | Model loading, trainer, LoRA / quantization, secrets, license whitelist |
        | `evaluation_plan.yaml` | Paired baseline-vs-treatment protocol, harness tasks, leakage checks |
        | `colab_notebook_spec.md` | Cell-by-cell Colab scaffold |
        | `claude_task_prompt.md` | Task brief for an implementation agent |
        | `acceptance_tests.md` | Smoke-run, paired-comparison, hygiene, reproducibility checks |
        | `implementation_spec.json` | Pydantic-validated implementation contract |

        ## Quick start

        1. Open the Colab notebook spec.
        2. Run cells 1–9 (smoke pass).
        3. If acceptance tests pass, flip `SMOKE = False` and iterate the matrix.
        """
    ).strip()


# ── public entry point ───────────────────────────────────────────────────────


def write_llm_development_pack(
    run_id: str, candidate_payload: dict[str, Any]
) -> Path:
    """Write the LLM-training development pack and return the pack directory."""
    token = settings.activate_run_scope(run_id)
    try:
        candidate = _to_candidate(candidate_payload)
        template = _safe_load_template(candidate.template_id or _BASE_TEMPLATE_ID)

        pack_dir = settings.development_packs_dir() / candidate.candidate_id
        pack_dir.mkdir(parents=True, exist_ok=True)

        (pack_dir / "README.md").write_text(_readme(candidate), encoding="utf-8")
        (pack_dir / "experiment_config.yaml").write_text(
            _experiment_config(candidate, template), encoding="utf-8"
        )
        (pack_dir / "training_plan.yaml").write_text(
            _training_plan(candidate, template), encoding="utf-8"
        )
        (pack_dir / "evaluation_plan.yaml").write_text(
            _evaluation_plan(candidate, template), encoding="utf-8"
        )
        (pack_dir / "colab_notebook_spec.md").write_text(
            _colab_notebook_spec(candidate, template), encoding="utf-8"
        )
        (pack_dir / "claude_task_prompt.md").write_text(
            _claude_task_prompt(candidate, template), encoding="utf-8"
        )
        (pack_dir / "acceptance_tests.md").write_text(
            _acceptance_tests(candidate), encoding="utf-8"
        )

        # Minimal implementation_spec.json so downstream tooling that expects
        # this file still works. Full ImplementationSpec is spatial-shaped;
        # we write a slim training-shaped variant here.
        spec = {
            "candidate_id": candidate.candidate_id,
            "template_id": candidate.template_id,
            "kind": "training_research",
            "training_strategy": candidate.exposure_family,
            "evaluation_target": candidate.outcome_family,
            "method": candidate.method_template,
            "claim_strength": candidate.claim_strength,
            "exposure_source": candidate.exposure_source,
            "outcome_source": candidate.outcome_source,
            "controls": candidate.join_plan.get("controls", []),
            "boundary": candidate.join_plan.get("boundary_source", []),
            "key_threats": candidate.key_threats,
            "mitigations": candidate.mitigations,
            "automation_risk": candidate.automation_risk,
            "cloud_safe": candidate.cloud_safe,
            "required_secrets": candidate.required_secrets,
            "smoke_test_plan": [
                f"smoke train: {(template.get('smoke_test') or {}).get('train_samples', 500)} samples",
                f"smoke eval: {(template.get('smoke_test') or {}).get('eval_samples', 200)} samples",
                f"smoke max steps: {(template.get('smoke_test') or {}).get('max_steps', 100)}",
            ],
            "expected_outputs": [
                "data/eval/${run_id}/results.json",
                "data/eval/${run_id}/baseline_vs_treatment.csv",
                "output/report/${candidate_id}_evaluation_summary.md",
            ],
        }
        (pack_dir / "implementation_spec.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            "wrote_llm_dev_pack candidate_id=%s pack_dir=%s",
            candidate.candidate_id, pack_dir,
        )
        return pack_dir
    finally:
        settings.deactivate_run_scope(token)
