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

    # Use SourceUseSpec for richer column info when available
    sus_list = spec.get("source_use_specs") or []
    exp_sus = next((s for s in sus_list if s.get("role") == "exposure"), {})
    out_sus = next((s for s in sus_list if s.get("role") == "outcome"), {})

    exp_vars = exp_sus.get("raw_columns") or candidate.exposure_variables or ["exposure_metric"]
    out_vars = out_sus.get("raw_columns") or candidate.outcome_variables or [f"{candidate.outcome_family}_prevalence"]
    ctrl_vars = ["poverty_rate", "median_income", "pct_no_hs_diploma", "pct_nonwhite", "pop_density"]

    ctrl_sources = ", ".join(s.get("source_name", "ACS") for s in ctrl_steps) or "ACS"
    bnd_sources = ", ".join(s.get("source_name", "TIGER_Lines") for s in bnd_steps) or "TIGER_Lines"

    exp_native_unit = exp_sus.get("native_unit") or candidate.unit_of_analysis
    exp_agg_method = exp_sus.get("aggregation_method") or ""
    exp_join_recipe = exp_sus.get("join_recipe") or {}

    lines = [
        f"# Data contract for candidate: {candidate.candidate_id}",
        f"analysis_unit: {candidate.unit_of_analysis}",
        "join_key: GEOID",
        "",
        "inputs:",
        "  exposure:",
        f"    source: {exp_step.get('source_name', candidate.exposure_source)}",
        f"    native_grain: {exp_native_unit}",
        f"    target_grain: {candidate.unit_of_analysis}",
        f"    raw_columns: [{', '.join(exp_vars[:6])}]",
        f"    acquisition_method: {exp_step.get('method', 'api')}",
    ]
    if exp_agg_method:
        lines.append(f"    aggregation_method: {exp_agg_method}")
    if exp_join_recipe:
        lines.append(f"    join_recipe: {exp_join_recipe.get('recipe_id', 'see_data_source_notes')}")

    lines += [
        "  outcome:",
        f"    source: {out_step.get('source_name', candidate.outcome_source)}",
        f"    native_grain: {out_sus.get('native_unit') or candidate.unit_of_analysis}",
        f"    target_grain: {candidate.unit_of_analysis}",
        f"    raw_columns: [{', '.join(out_vars[:4])}]",
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
        f"    required_columns: [GEOID, {', '.join((exp_sus.get('derived_features') or exp_vars)[:2])}, {out_vars[0]}, poverty_rate, median_income]",
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

    # Use SourceUseSpec derived features when available
    sus_list = spec.get("source_use_specs") or []
    exp_sus = next((s for s in sus_list if s.get("role") == "exposure"), {})
    out_sus = next((s for s in sus_list if s.get("role") == "outcome"), {})

    all_features = exp_sus.get("derived_features") or candidate.exposure_variables or []
    if osmnx_plan:
        all_features = osmnx_plan.get("expected_features", all_features)

    exp_native = exp_sus.get("native_unit") or candidate.unit_of_analysis
    exp_agg = exp_sus.get("aggregation_method") or f"mean_per_{candidate.unit_of_analysis}"
    exp_join_recipe = exp_sus.get("join_recipe") or {}

    out_cols = out_sus.get("raw_columns") or [f"{candidate.outcome_family}_prevalence"]

    ctrl_vars = ["poverty_rate", "median_income", "pct_no_hs_diploma", "pct_nonwhite", "pop_density"]

    lines = [
        f"# Feature plan for candidate: {candidate.candidate_id}",
        "",
        "exposure_features:",
    ]
    for f in all_features:
        lines.append(f"  - name: {f}")
        lines.append(f"    source: {candidate.exposure_source}")
        lines.append(f"    native_grain: {exp_native}")
        lines.append(f"    target_grain: {candidate.unit_of_analysis}")
        if exp_native != candidate.unit_of_analysis.replace("census_", ""):
            lines.append(f"    aggregation_method: {exp_agg}")
            if exp_join_recipe:
                lines.append(f"    aggregation_recipe: {exp_join_recipe.get('recipe_id', 'see_data_source_notes')}")
        lines.append(f"    missingness_policy: drop_if_pct_missing_gt_50")

    lines += [
        "",
        "outcome_features:",
    ]
    for col in out_cols[:3]:
        lines += [
            f"  - name: {col}",
            f"    source: {candidate.outcome_source}",
            f"    grain: {out_sus.get('native_unit') or candidate.unit_of_analysis}",
            "    unit: age_adjusted_percent",
        ]

    lines += [
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


# ── pytest skeleton ──────────────────────────────────────────────────────────

def _pytest_skeleton(candidate: ComposedCandidate) -> str:
    cid = candidate.candidate_id
    exp_var = (candidate.exposure_variables or ["exposure_metric"])[0]
    out_var = f"{candidate.outcome_family}_prevalence"
    return textwrap.dedent(f'''
        """Auto-generated pytest skeleton for {cid}.

        Fill in each test body — replace `raise NotImplementedError` with real assertions.
        Run: pytest tests/generated/test_{cid}_contract.py
        """
        from __future__ import annotations

        import json
        from pathlib import Path

        import pytest

        # ── Paths ──────────────────────────────────────────────────────────────
        REPO_ROOT = Path(__file__).parent.parent.parent
        FEATURES_CSV = REPO_ROOT / "data" / "processed" / "tract_features.csv"
        MODEL_CSV = REPO_ROOT / "output" / "tables" / "model_summary.csv"
        REPORT_MD = REPO_ROOT / "output" / "report" / "technical_summary.md"
        SPEC_JSON = REPO_ROOT / "output" / "development_packs" / "{cid}" / "implementation_spec.json"


        # ── Schema validation ──────────────────────────────────────────────────
        def test_implementation_spec_is_valid_json():
            assert SPEC_JSON.exists(), f"Spec not found: {{SPEC_JSON}}"
            data = json.loads(SPEC_JSON.read_text())
            assert data.get("candidate_id") == "{cid}"


        # ── Feature engineering contract ───────────────────────────────────────
        @pytest.mark.skipif(not FEATURES_CSV.exists(), reason="Run pipeline first")
        def test_tract_features_non_empty():
            import pandas as pd
            df = pd.read_csv(FEATURES_CSV)
            assert len(df) >= 10, f"Expected >= 10 rows, got {{len(df)}}"


        @pytest.mark.skipif(not FEATURES_CSV.exists(), reason="Run pipeline first")
        def test_tract_features_has_geoid():
            import pandas as pd
            df = pd.read_csv(FEATURES_CSV)
            assert "GEOID" in df.columns


        @pytest.mark.skipif(not FEATURES_CSV.exists(), reason="Run pipeline first")
        def test_geoid_is_unique():
            import pandas as pd
            df = pd.read_csv(FEATURES_CSV)
            assert df["GEOID"].nunique() == len(df), "GEOID must be unique"


        @pytest.mark.skipif(not FEATURES_CSV.exists(), reason="Run pipeline first")
        def test_exposure_coverage():
            import pandas as pd
            df = pd.read_csv(FEATURES_CSV)
            if "{exp_var}" in df.columns:
                pct_non_null = df["{exp_var}"].notna().mean()
                assert pct_non_null >= 0.5, f"Exposure coverage {{pct_non_null:.0%}} < 50%"


        @pytest.mark.skipif(not FEATURES_CSV.exists(), reason="Run pipeline first")
        def test_outcome_coverage():
            import pandas as pd
            df = pd.read_csv(FEATURES_CSV)
            if "{out_var}" in df.columns:
                pct_non_null = df["{out_var}"].notna().mean()
                assert pct_non_null >= 0.5, f"Outcome coverage {{pct_non_null:.0%}} < 50%"


        # ── Model output contract ──────────────────────────────────────────────
        @pytest.mark.skipif(not MODEL_CSV.exists(), reason="Run pipeline first")
        def test_model_summary_columns():
            import pandas as pd
            df = pd.read_csv(MODEL_CSV)
            required = {{"variable", "coefficient", "std_error", "p_value", "n_obs"}}
            assert required.issubset(df.columns), f"Missing: {{required - set(df.columns)}}"


        @pytest.mark.skipif(not MODEL_CSV.exists(), reason="Run pipeline first")
        def test_model_summary_has_exposure_row():
            import pandas as pd
            df = pd.read_csv(MODEL_CSV)
            assert len(df) >= 1, "Model summary must have at least one coefficient row"


        # ── Report contract ────────────────────────────────────────────────────
        @pytest.mark.skipif(not REPORT_MD.exists(), reason="Run pipeline first")
        def test_report_has_limitations_section():
            text = REPORT_MD.read_text()
            assert "Limitations" in text or "limitations" in text


        # ── Policy tests ───────────────────────────────────────────────────────
        def test_automation_risk_not_high():
            """Candidate must not be high automation risk."""
            spec = json.loads(SPEC_JSON.read_text()) if SPEC_JSON.exists() else {{}}
            risk = spec.get("automation_risk", "low")
            assert risk != "high", f"automation_risk={{risk}} blocks claude_code_ready"


        def test_no_experimental_tags():
            spec = json.loads(SPEC_JSON.read_text()) if SPEC_JSON.exists() else {{}}
            tags = set(spec.get("technology_tags", []))
            experimental = tags & {{"streetview_cv", "deep_learning", "satellite_cv", "experimental"}}
            assert not experimental, f"Experimental tags present: {{experimental}}"
    ''').strip()


# ── data_source_notes.md ─────────────────────────────────────────────────────

def _data_source_notes(candidate: ComposedCandidate, spec: dict[str, Any]) -> str:
    """Generate data_source_notes.md from SourceUseSpec entries in the spec."""
    sus_list = spec.get("source_use_specs") or []

    lines = [
        f"# Data Source Notes: {candidate.candidate_id}",
        "",
        "This file describes exactly how each data source is used in this candidate's",
        "pipeline — including native grain, required aggregation, join keys, concrete",
        "column names, and known limitations.",
        "",
    ]

    for sus in sus_list:
        source_id = sus.get("source_id", "unknown")
        role = sus.get("role", "unknown")
        native_unit = sus.get("native_unit", "unknown")
        target_unit = sus.get("target_unit", candidate.unit_of_analysis)
        raw_cols = sus.get("raw_columns", [])
        derived = sus.get("derived_features", [])
        acq_method = sus.get("acquisition_method", "")
        acq_url = sus.get("acquisition_url", "")
        join_recipe = sus.get("join_recipe") or {}
        agg_method = sus.get("aggregation_method", "")
        validation = sus.get("validation_rules", [])
        limitations = sus.get("known_limitations", [])

        lines += [
            f"## {source_id} ({role})",
            "",
            f"- **Native spatial unit**: `{native_unit}`",
            f"- **Target analysis unit**: `{candidate.unit_of_analysis}`",
        ]

        if native_unit and target_unit and native_unit != target_unit.replace("census_", ""):
            lines.append(
                f"- **Aggregation required**: Yes — `{native_unit}` → `{target_unit}` "
                f"via **{agg_method or 'population_weighted_mean'}**"
            )
        else:
            lines.append("- **Aggregation required**: No — source native unit matches analysis unit")

        if acq_method:
            lines.append(f"- **Acquisition method**: `{acq_method}`")
        if acq_url:
            lines.append(f"- **Source URL**: {acq_url}")

        if raw_cols:
            lines += ["", "**Raw columns used**:", ""]
            for col in raw_cols[:8]:
                lines.append(f"  - `{col}`")

        if derived:
            lines += ["", "**Derived features**:", ""]
            for feat in derived[:8]:
                lines.append(f"  - `{feat}`")

        if join_recipe:
            recipe_id = join_recipe.get("recipe_id", "")
            left_key = join_recipe.get("left_key", "")
            right_key = join_recipe.get("right_key", "")
            if recipe_id:
                lines += [
                    "",
                    f"**Join recipe**: `{recipe_id}`",
                    f"  - Left key: `{left_key}` → Right key: `{right_key}`",
                ]
                for w in join_recipe.get("warnings", []):
                    lines.append(f"  - ⚠️ {w}")

        if validation:
            lines += ["", "**Validation rules**:", ""]
            for r in validation[:4]:
                lines.append(f"  - `{r}`")

        if limitations:
            lines += ["", "**Known limitations**:", ""]
            for lim in limitations[:3]:
                lines.append(f"  - {lim}")

        lines.append("")

    return "\n".join(lines)


# ── data_lineage_plan.yaml ────────────────────────────────────────────────────

def _data_lineage_plan_yaml(candidate: ComposedCandidate, spec: dict[str, Any]) -> str:
    """Serialize the data lineage plan as YAML."""
    plan = spec.get("data_lineage_plan") or {}

    lines = [
        f"# Data lineage plan for candidate: {candidate.candidate_id}",
        "",
        f"candidate_id: {candidate.candidate_id}",
        f"analysis_unit: {plan.get('analysis_unit', candidate.unit_of_analysis)}",
        f"final_join_key: {plan.get('final_join_key', 'GEOID')}",
        "",
        "lineage_steps:",
    ]

    for step in plan.get("lineage_steps", []):
        step_name = step.get("step", "unknown")
        lines.append(f"  - step: {step_name}")
        for k, v in step.items():
            if k == "step":
                continue
            if isinstance(v, list):
                lines.append(f"    {k}:")
                for item in v[:6]:
                    lines.append(f"      - {item}")
            elif v is not None:
                lines.append(f"    {k}: {v}")

    return "\n".join(lines)


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
                | `data_contract.yaml` | Input/output column contracts per data source (source-aware) |
                | `feature_plan.yaml` | Feature engineering plan with native/target grain and aggregation |
                | `analysis_plan.yaml` | Statistical method and formula |
                | `acceptance_tests.md` | Pytest checklist and smoke test criteria |
                | `claude_task_prompt.md` | Task prompt to hand to Claude Code for implementation |
                | `data_source_notes.md` | Per-source grain, join recipes, raw columns, limitations |
                | `data_lineage_plan.yaml` | Step-by-step data lineage from raw sources to analysis unit |

                An **executable pytest skeleton** is also written to
                `tests/generated/test_{candidate.candidate_id}_contract.py`.

                ## Quick Start

                Copy `claude_task_prompt.md` and hand it to Claude Code to begin implementation.
                Review `data_source_notes.md` for source-specific grain and join requirements.
            """).strip(),
            encoding="utf-8",
        )

        # Augment spec with runtime policy fields before writing
        spec.setdefault("failure_policy", {
            "on_acquisition_error": "log_and_continue",
            "on_join_error": "raise",
            "on_model_error": "raise",
        })
        spec.setdefault("runtime_budget", {
            "smoke_test_max_minutes": 8,
            "full_run_max_minutes": 60,
        })
        spec.setdefault("network_timeout_seconds", 30)
        spec.setdefault("fixture_policy", {
            "use_fixture_in_ci": True,
            "fixture_path": "data/fixtures",
            "live_calls_only_in_smoke_test": True,
        })
        spec.setdefault("data_quality_checks", [
            "min_non_null_exposure_fraction:0.5",
            "min_non_null_outcome_fraction:0.5",
            "geoid_uniqueness",
            "no_duplicate_rows",
        ])
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
        (pack_dir / "data_source_notes.md").write_text(
            _data_source_notes(candidate, spec), encoding="utf-8"
        )
        (pack_dir / "data_lineage_plan.yaml").write_text(
            _data_lineage_plan_yaml(candidate, spec), encoding="utf-8"
        )

        # Executable pytest skeleton — written to tests/generated/ at repo root
        generated_dir = settings.output_dir().parent / "tests" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        (generated_dir / f"test_{candidate.candidate_id}_contract.py").write_text(
            _pytest_skeleton(candidate), encoding="utf-8"
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
