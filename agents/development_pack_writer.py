from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from agents import settings
from agents.implementation_spec_builder import build_implementation_spec
from models.candidate_composer_schema import ComposedCandidate

_SMOKE_GEOGRAPHY = "Cambridge, Massachusetts"


def _to_candidate(candidate: dict[str, Any]) -> ComposedCandidate:
    return ComposedCandidate(**candidate)


# ── claude_task_prompt.md ──────────────────────────────────────────────────────

def _claude_task_prompt(candidate: ComposedCandidate, spec: dict[str, Any]) -> str:
    acq_steps = spec.get("data_acquisition_steps", [])
    exposure_step = next((s for s in acq_steps if s.get("source_role") == "exposure"), {})
    outcome_step = next((s for s in acq_steps if s.get("source_role") == "outcome"), {})
    control_steps = [s for s in acq_steps if s.get("source_role") == "control"]
    boundary_steps = [s for s in acq_steps if s.get("source_role") == "boundary"]

    analysis_steps = spec.get("analysis_steps", [])
    analysis_formula = analysis_steps[0].get("formula_or_model", "outcome ~ exposure + controls") if analysis_steps else "outcome ~ exposure + controls"

    secrets = candidate.required_secrets
    secrets_note = (
        "**No API keys or secrets required** — all data sources are open access."
        if not secrets
        else f"Required secrets: {', '.join(secrets)}"
    )

    threat_lines = "\n".join(
        f"  - **{t}**: {candidate.mitigations.get(t, 'See analysis plan for mitigation strategy.')}"
        for t in candidate.key_threats
    ) or "  - No specific causal threats declared; treat as descriptive association."

    control_source_names = ", ".join(s.get("source_name", "ACS") for s in control_steps) or "ACS"
    boundary_source_names = ", ".join(s.get("source_name", "TIGER_Lines") for s in boundary_steps) or "TIGER_Lines"

    smoke_plan = spec.get("smoke_test_plan", [])
    smoke_lines = "\n".join(f"  - {s}" for s in smoke_plan) or "  - Run on Cambridge, MA; verify non-empty output."

    osmnx_plan = spec.get("osmnx_feature_plan") or {}
    osmnx_section = ""
    if osmnx_plan:
        features = osmnx_plan.get("expected_features", [])
        osmnx_section = textwrap.dedent(f"""
            ## OSMnx Feature Engineering

            This candidate uses street-network features computed via `osmnx`.

            **Expected features** (first five): {', '.join(features[:5])}

            **CI policy**: In automated tests, load graph from
            `data/fixtures/osmnx_cambridge.graphml` (checked in) instead of
            making a live API call.  Use `use_fixture=True` when calling
            `build_osmnx_features()`.

            **Smoke test**: Live OSMnx call on {_SMOKE_GEOGRAPHY}.
        """).strip()

    prompt = textwrap.dedent(f"""
        # Claude Code Task: {candidate.candidate_id}

        ## Goal

        Build a **cloud-safe, reproducible Python research pipeline** that estimates
        the association between **{candidate.exposure_family}** (exposure) and
        **{candidate.outcome_family}** (outcome) at the **{candidate.unit_of_analysis}**
        level in the United States.

        This pipeline must run end-to-end without human intervention, produce a
        quantitative analysis report, and pass an automated smoke test on a small
        geography ({_SMOKE_GEOGRAPHY}) in under {8} minutes.

        ## Candidate Summary

        | Field | Value |
        |-------|-------|
        | Candidate ID | `{candidate.candidate_id}` |
        | Exposure family | {candidate.exposure_family} |
        | Outcome family | {candidate.outcome_family} |
        | Method | {candidate.method_template} |
        | Unit of analysis | {candidate.unit_of_analysis} |
        | Automation risk | {candidate.automation_risk} |
        | Cloud safe | {'Yes' if candidate.cloud_safe else 'No'} |
        | Causal claim strength | {candidate.claim_strength} |

        ## Data Sources

        ### Exposure
        - **Source**: `{exposure_step.get('source_name', candidate.exposure_source)}`
        - **Acquisition method**: `{exposure_step.get('method', 'api')}`
        - **Expected output file**: `{', '.join(exposure_step.get('expected_files', []))}`
        - **Variables**: {', '.join(candidate.exposure_variables) or 'see feature_plan.yaml'}

        ### Outcome
        - **Source**: `{outcome_step.get('source_name', candidate.outcome_source)}`
        - **Acquisition method**: `{outcome_step.get('method', 'api')}`
        - **Expected output file**: `{', '.join(outcome_step.get('expected_files', []))}`
        - **Variables**: {', '.join(candidate.outcome_variables) or f'{candidate.outcome_family} prevalence'}

        ### Controls
        - **Source(s)**: {control_source_names}
        - Include: poverty rate, median household income, educational attainment,
          race/ethnicity composition, population density

        ### Boundary
        - **Source**: {boundary_source_names}
        - Join key: **GEOID** (11-digit census tract FIPS code)

        ## Required Outputs

        1. `data/processed/tract_features.csv` — merged exposure + outcome + controls + GEOID
        2. `output/tables/model_summary.csv` — regression coefficients, SE, p-values, N
        3. `output/report/technical_summary.md` — narrative summary with limitations section

        ## Repo Files to Create or Modify

        - `pipeline/{candidate.candidate_id}/acquire.py` — data acquisition
        - `pipeline/{candidate.candidate_id}/features.py` — feature engineering
        - `pipeline/{candidate.candidate_id}/analyze.py` — statistical analysis
        - `pipeline/{candidate.candidate_id}/smoke_test.py` — smoke test runner
        - `pipeline/{candidate.candidate_id}/README.md` — pipeline documentation
        - `tests/test_{candidate.candidate_id}.py` — pytest unit tests

        ## Implementation Steps

        1. **Acquire exposure data** from `{exposure_step.get('source_name', candidate.exposure_source)}`
           using method `{exposure_step.get('method', 'api')}`.
           Write to `{', '.join(exposure_step.get('expected_files', ['data/raw/exposure.csv']))}`

        2. **Acquire outcome data** from `{outcome_step.get('source_name', candidate.outcome_source)}`
           using method `{outcome_step.get('method', 'api')}`.
           Write to `{', '.join(outcome_step.get('expected_files', ['data/raw/outcome.csv']))}`

        3. **Acquire control variables** from {control_source_names}.
           Pull at {candidate.unit_of_analysis} grain.

        4. **Acquire boundary geometry** from {boundary_source_names}.
           Filter to target geography; ensure GEOID is present.

        5. **Feature engineering**: Aggregate all sources to {candidate.unit_of_analysis} level,
           join on GEOID, produce `data/processed/tract_features.csv`.

        6. **Analysis**: Run `{candidate.method_template}` regression:
           ```
           {analysis_formula}
           ```
           Report coefficients, robust/clustered SE, N observations.

        7. **Write outputs**: `output/tables/model_summary.csv` and
           `output/report/technical_summary.md`.

        {osmnx_section}

        ## Smoke Test

        {smoke_lines}

        ## Identification Threats and Mitigations

        {threat_lines}

        ## Failure Handling

        - If any remote data call fails (network error, HTTP 4xx/5xx), log a warning
          and write an empty placeholder file — do **not** raise an unhandled exception.
        - Use `try/except` around all external I/O; emit structured log messages.
        - Smoke test failures must produce a clear error message indicating which step failed.

        ## Cloud Constraints

        {secrets_note}

        - Do **not** store raw street-view images or any PII.
        - Do **not** require any manual download steps.
        - Runtime must stay under 10 minutes for smoke test geography.
        - All external HTTP calls must have a 30-second timeout.
        - Required Python extras: `{', '.join(spec.get('required_python_extras', ['geospatial']))}`

        ## Policy Constraints

        - This is a **descriptive / associational analysis** — do not claim causality unless
          the method_template explicitly supports it.
        - Include a **"Limitations" section** in `technical_summary.md`.
        - Do not use paid APIs.
        - Do not enable experimental or street-view sources.

        ## Acceptance Criteria

        - [ ] `pytest tests/test_{candidate.candidate_id}.py` passes with no errors
        - [ ] Smoke test completes in < 8 minutes on {_SMOKE_GEOGRAPHY}
        - [ ] `data/processed/tract_features.csv` is non-empty (≥ 10 rows)
        - [ ] `output/tables/model_summary.csv` contains a coefficient row for the
              main exposure variable
        - [ ] `output/report/technical_summary.md` contains a "Limitations" section
        - [ ] No paid API calls, no raw image storage, no required secrets (unless listed above)

        ## Do Not Do

        - Do not use `plt.show()` or any interactive display calls.
        - Do not hardcode local file paths — use `pathlib.Path` relative to repo root.
        - Do not use `print()` for logging — use `logging.getLogger(__name__)`.
        - Do not add dependencies outside the standard library and the extras listed above.
        - Do not implement street-view imagery collection or deep learning vision models.
    """).strip()

    return prompt


# ── data_contract.yaml ────────────────────────────────────────────────────────

def _data_contract(candidate: ComposedCandidate, spec: dict[str, Any]) -> str:
    acq_steps = spec.get("data_acquisition_steps", [])
    exp_step = next((s for s in acq_steps if s.get("source_role") == "exposure"), {})
    out_step = next((s for s in acq_steps if s.get("source_role") == "outcome"), {})
    ctrl_steps = [s for s in acq_steps if s.get("source_role") == "control"]
    bnd_steps = [s for s in acq_steps if s.get("source_role") == "boundary"]

    exp_vars = candidate.exposure_variables or ["exposure_metric"]
    out_vars = candidate.outcome_variables or [f"{candidate.outcome_family}_prevalence"]
    ctrl_vars = ["poverty_rate", "median_income", "pct_no_hs_diploma", "pct_nonwhite", "pop_density"]

    ctrl_sources = ", ".join(s.get("source_name", "ACS") for s in ctrl_steps) or "ACS"
    bnd_sources = ", ".join(s.get("source_name", "TIGER_Lines") for s in bnd_steps) or "TIGER_Lines"

    lines = [
        f"# Data contract for candidate: {candidate.candidate_id}",
        "analysis_unit: census_tract",
        "join_key: GEOID",
        "",
        "inputs:",
        "  exposure:",
        f"    source: {exp_step.get('source_name', candidate.exposure_source)}",
        f"    expected_columns: [{', '.join(exp_vars)}]",
        f"    grain: {candidate.unit_of_analysis}",
        f"    acquisition_method: {exp_step.get('method', 'api')}",
        "  outcome:",
        f"    source: {out_step.get('source_name', candidate.outcome_source)}",
        f"    expected_columns: [{', '.join(out_vars)}]",
        f"    grain: {candidate.unit_of_analysis}",
        f"    acquisition_method: {out_step.get('method', 'api')}",
        "  controls:",
        f"    source: {ctrl_sources}",
        f"    expected_columns: [{', '.join(ctrl_vars)}]",
        f"    grain: {candidate.unit_of_analysis}",
        "    acquisition_method: api",
        "  boundary:",
        f"    source: {bnd_sources}",
        "    expected_geometry: Polygon",
        "    grain: census_tract",
        "    acquisition_method: download",
        "",
        "outputs:",
        "  tract_features:",
        "    path: data/processed/tract_features.csv",
        f"    required_columns: [GEOID, {', '.join(exp_vars[:2])}, {out_vars[0]}, poverty_rate, median_income]",
        "  model_summary:",
        "    path: output/tables/model_summary.csv",
        "    required_columns: [variable, coefficient, std_error, p_value, n_obs]",
        "  technical_report:",
        "    path: output/report/technical_summary.md",
        "    required_sections: [Overview, Data, Methods, Results, Limitations]",
    ]
    return "\n".join(lines)


# ── feature_plan.yaml ─────────────────────────────────────────────────────────

def _feature_plan(candidate: ComposedCandidate, spec: dict[str, Any]) -> str:
    osmnx_plan = spec.get("osmnx_feature_plan") or {}
    all_features = candidate.exposure_variables or []
    if osmnx_plan:
        all_features = osmnx_plan.get("expected_features", all_features)

    ctrl_vars = ["poverty_rate", "median_income", "pct_no_hs_diploma", "pct_nonwhite", "pop_density"]

    lines = [
        f"# Feature plan for candidate: {candidate.candidate_id}",
        "",
        "exposure_features:",
    ]
    for f in all_features:
        lines.append(f"  - name: {f}")
        lines.append(f"    source: {candidate.exposure_source}")
        lines.append(f"    aggregation: mean_per_{candidate.unit_of_analysis}")
        lines.append(f"    missingness_policy: drop_if_pct_missing_gt_50")

    lines += [
        "",
        "outcome_features:",
        f"  - name: {candidate.outcome_family}_prevalence",
        f"    source: {candidate.outcome_source}",
        f"    grain: {candidate.unit_of_analysis}",
        "    unit: age_adjusted_percent",
        "",
        "control_features:",
    ]
    for cv in ctrl_vars:
        lines.append(f"  - name: {cv}")
        lines.append("    source: ACS")
        lines.append(f"    grain: {candidate.unit_of_analysis}")

    if osmnx_plan:
        lines += [
            "",
            "osmnx_notes:",
            f"  graph_type: {osmnx_plan.get('graph_type', 'walk')}",
            f"  network_type: {osmnx_plan.get('network_type', 'walk')}",
            f"  fixture_path: {osmnx_plan.get('fixture_path', 'data/fixtures/osmnx_cambridge.graphml')}",
        ]

    lines += [
        "",
        "spatial_join:",
        "  join_key: GEOID",
        f"  spatial_unit: {candidate.unit_of_analysis}",
        "  boundary_source: TIGER_Lines",
        "  join_type: left  # keep all tracts; exposure=NaN for tracts outside data coverage",
        "",
        "validation:",
        "  min_non_null_exposure_fraction: 0.5",
        "  min_non_null_outcome_fraction: 0.5",
        "  required_columns: [GEOID]",
    ]
    return "\n".join(lines)


# ── acceptance_tests.md ───────────────────────────────────────────────────────

def _acceptance_tests(candidate: ComposedCandidate) -> str:
    cid = candidate.candidate_id
    return textwrap.dedent(f"""
        # Acceptance Tests: {cid}

        ## Unit Tests (`pytest tests/test_{cid}.py`)

        - [ ] `test_acquire_exposure` — exposure source returns non-empty DataFrame with expected columns
        - [ ] `test_acquire_outcome` — outcome source returns non-empty DataFrame with GEOID column
        - [ ] `test_acquire_controls` — ACS returns poverty_rate, median_income, pct_no_hs_diploma
        - [ ] `test_acquire_boundary` — TIGER_Lines returns GeoDataFrame with geometry column
        - [ ] `test_feature_engineering` — tract_features.csv has ≥ 10 rows and required columns
        - [ ] `test_no_duplicate_geoids` — GEOID is unique in tract_features.csv
        - [ ] `test_exposure_coverage` — ≥ 50% of tracts have non-null exposure value
        - [ ] `test_outcome_coverage` — ≥ 50% of tracts have non-null outcome value
        - [ ] `test_model_output_schema` — model_summary.csv has columns: variable, coefficient, std_error, p_value, n_obs
        - [ ] `test_report_has_limitations` — technical_summary.md contains "Limitations" section

        ## Smoke Test (`python pipeline/{cid}/smoke_test.py`)

        - [ ] Completes in < 8 minutes on {_SMOKE_GEOGRAPHY}
        - [ ] `data/processed/tract_features.csv` has ≥ 10 rows
        - [ ] `output/tables/model_summary.csv` exists and is non-empty
        - [ ] `output/report/technical_summary.md` exists and is non-empty
        - [ ] No unhandled exceptions raised
        - [ ] No paid API calls made (verify with network mock or offline run)

        ## Schema Tests

        - [ ] `tract_features.csv` passes `pandera` schema: GEOID str, exposure float, outcome float, controls float
        - [ ] `model_summary.csv` passes column-type validation
        - [ ] `implementation_spec.json` validates against `ImplementationSpec` Pydantic model

        ## Policy Tests

        - [ ] `test_no_paid_api` — assert no HTTP calls to paid API domains
        - [ ] `test_no_raw_image_storage` — assert no image files written to disk
        - [ ] `test_no_required_secrets` — pipeline runs successfully without any env secrets

        ## Regression Tests

        - [ ] `test_stable_candidate_id` — candidate_id `{cid}` is stable across re-runs
        - [ ] `test_no_experimental_sources` — technology_tags does not contain "experimental"
        - [ ] `test_automation_risk` — automation_risk is "low" or "medium" (not "high")
    """).strip()


# ── Main writer ───────────────────────────────────────────────────────────────

def write_development_pack(run_id: str, candidate_payload: dict[str, Any]) -> Path:
    token = settings.activate_run_scope(run_id)
    try:
        candidate = _to_candidate(candidate_payload)
        spec_obj = build_implementation_spec(candidate)
        spec = spec_obj.model_dump()

        pack_dir = settings.development_packs_dir() / candidate.candidate_id
        pack_dir.mkdir(parents=True, exist_ok=True)

        (pack_dir / "README.md").write_text(
            textwrap.dedent(f"""
                # Development Pack: {candidate.candidate_id}

                | Field | Value |
                |-------|-------|
                | Exposure | {candidate.exposure_family} |
                | Outcome | {candidate.outcome_family} |
                | Method | {candidate.method_template} |
                | Unit of analysis | {candidate.unit_of_analysis} |
                | Automation risk | {candidate.automation_risk} |
                | Cloud safe | {'Yes' if candidate.cloud_safe else 'No'} |

                ## Files

                | File | Description |
                |------|-------------|
                | `implementation_spec.json` | Full implementation contract (Pydantic-validated) |
                | `data_contract.yaml` | Input/output column contracts per data source |
                | `feature_plan.yaml` | Feature engineering plan with aggregation and validation rules |
                | `analysis_plan.yaml` | Statistical method and formula |
                | `acceptance_tests.md` | Pytest checklist and smoke test criteria |
                | `claude_task_prompt.md` | Task prompt to hand to Claude Code for implementation |

                ## Quick Start

                Copy `claude_task_prompt.md` and hand it to Claude Code to begin implementation.
            """).strip(),
            encoding="utf-8",
        )

        (pack_dir / "implementation_spec.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pack_dir / "data_contract.yaml").write_text(
            _data_contract(candidate, spec), encoding="utf-8"
        )
        (pack_dir / "feature_plan.yaml").write_text(
            _feature_plan(candidate, spec), encoding="utf-8"
        )
        (pack_dir / "analysis_plan.yaml").write_text(
            textwrap.dedent(f"""
                # Analysis plan for candidate: {candidate.candidate_id}

                method: {candidate.method_template}
                outcome: {candidate.outcome_family}
                exposure: {candidate.exposure_family}
                unit_of_analysis: {candidate.unit_of_analysis}
                claim_strength: {candidate.claim_strength}

                formula: >
                  {(spec.get('analysis_steps') or [{}])[0].get('formula_or_model', 'outcome ~ exposure + controls')}

                robustness_checks:
                  - alternative_exposure_definitions
                  - clustered_standard_errors
                  - spatial_lag_controls

                reporting:
                  - main_coefficient_table
                  - confidence_intervals_95pct
                  - n_observations
                  - limitations_section
            """).strip(),
            encoding="utf-8",
        )
        (pack_dir / "acceptance_tests.md").write_text(
            _acceptance_tests(candidate), encoding="utf-8"
        )
        (pack_dir / "claude_task_prompt.md").write_text(
            _claude_task_prompt(candidate, spec), encoding="utf-8"
        )

        # Keep a candidate-scoped copy of implementation_spec.json
        candidate_dir = settings.candidates_dir() / candidate.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "implementation_spec.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return pack_dir
    finally:
        settings.deactivate_run_scope(token)
