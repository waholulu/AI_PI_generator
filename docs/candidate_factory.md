# Candidate Factory

The candidate factory is the core engine that turns a research template into a ranked list of concrete, automatable research designs — each with a full development pack ready for Claude Code to implement.

## Pipeline overview

```
Research Template
       │
       ▼
1. Candidate Composer       ← enumerate X × Y × method
       │
       ▼
2. Feasibility Precheck     ← deterministic gate (no LLM)
       │
       ▼
3. Candidate Repair         ← auto-fix blocked candidates
       │
       ▼
4. Ranker + Scorer          ← weighted composite score
       │
       ▼
5. Implementation Spec      ← declarative build spec
       │
       ▼
6. Development Pack         ← files for Claude Code
```

---

## 1. Research Template

Templates live in `config/research_templates/`.  Each template defines:

- `allowed_exposure_families` — exposure variable families with preferred sources
- `allowed_outcome_families` — health outcomes with preferred sources
- `allowed_methods` — identification strategies (cross-sectional, DiD, IV, RDD, …)
- `default_unit_of_analysis` — e.g. `census_tract`
- `default_boundary_source` — e.g. `TIGER_Lines`
- `default_controls` — e.g. `ACS`

Template IDs are passed to the API or CLI as `template_id`.

---

## 2. Source Capability Registry

`config/data_sources.yaml` is the single source of truth for every data source.  Each entry declares:

| Field | Description |
|-------|-------------|
| `tier` | `stable` / `tier2` / `experimental` |
| `roles` | `exposure` / `outcome` / `boundary` / `control` |
| `machine_readable` | whether the source can be fetched programmatically |
| `auth_required` | whether an API key is needed |
| `cost_required` | whether the source is paid |
| `cloud_safe` | whether the source is safe for automated cloud runs |
| `spatial_units` | resolutions available (tract, county, …) |

The registry is loaded by `agents/source_registry.py` and consulted at composition and feasibility time.

---

## 3. Candidate Composer

`agents/candidate_composer.py` performs a Cartesian product of exposure families × outcome families × methods, filtered by:

- `enable_experimental` — experimental-tier sources allowed?
- `enable_tier2` — tier-2 sources allowed?
- `no_paid_api` — paid APIs blocked?
- `no_manual_download` — manual-download sources blocked?

Each candidate gets:
- `automation_risk`: `low` / `medium` / `high`
- `required_secrets`: list of env-var names required at runtime
- `technology_tags`: `osmnx`, `remote_sensing`, `mobility`, `experimental`, …
- `initial_shortlist_status`: `ready` / `review` / `blocked`

**Guardrail**: candidates with `required_secrets` or `automation_risk=high` start at `review`, not `ready`.  Paid API candidates with `no_paid_api=True` start at `blocked`.

---

## 4. Feasibility Precheck

`agents/candidate_feasibility.py` runs six deterministic subchecks (no LLM):

| Subcheck | Pass condition |
|----------|---------------|
| `source_exists` | both sources resolve in registry |
| `role_coverage` | exposure source has "exposure" role; outcome has "outcome" |
| `machine_readable` | all sources are machine-readable |
| `spatial_join_path` | a boundary or implicit join anchor is available |
| `cloud_automation_feasibility` | no experimental sources with auth/cost |
| `identification_threats` | threats non-empty, mitigations cover ≥ 80% |

Overall verdict: `pass` → `ready`, `warning` → `review`, `fail` → `blocked`.

---

## 5. Candidate Repair

`agents/candidate_repair.py` attempts to auto-fix `blocked` candidates by:

1. Replacing missing exposure/outcome sources with template defaults
2. Adding missing boundary sources (TIGER_Lines)
3. Filling in identification threats from the method template

Up to `VALIDATION_MAX_SUBSTITUTIONS` (default: 2) rounds are attempted.

---

## 6. Ranker + Scorer

`agents/final_ranker.py` computes a weighted composite score:

| Component | Weight |
|-----------|--------|
| `data_feasibility` | 30% |
| `automation_feasibility` | 25% |
| `identification_quality` | 20% |
| `novelty` | 15% |
| `technology_innovation` | 10% |

**Guardrails (Step 3 policy)**:
- `automation_risk=high` → `overall` capped at **0.65**
- `required_secrets` present → `automation_feasibility` capped at **0.45**
- Previously blocked → `overall` capped at **0.45**

---

## 7. Implementation Spec

`agents/implementation_spec_builder.py` transforms a `ComposedCandidate` into an `ImplementationSpec`:

- `data_acquisition_steps` — per-source method (download / api / osmnx)
- `feature_engineering_steps` — library tags + pseudocode
- `analysis_steps` — formula + robustness checks
- `smoke_test_plan` — small geography (Cambridge, MA default)
- `osmnx_feature_plan` — populated for street-network candidates (see `agents/feature_modules/osmnx_features.py`)

---

## 8. Development Pack

`agents/development_pack_writer.py` writes a directory of files under `output/development_packs/{candidate_id}/`:

| File | Purpose |
|------|---------|
| `README.md` | Candidate metadata |
| `implementation_spec.json` | Full declarative spec |
| `data_contract.yaml` | Input table schema |
| `feature_plan.yaml` | Feature list |
| `analysis_plan.yaml` | Method + outcome |
| `acceptance_tests.md` | Test requirements |
| `claude_task_prompt.md` | Prompt for Claude Code |

**Auto-generation**: `run_candidate_factory_ideation()` automatically generates a development pack for every `ready` candidate — no manual trigger needed.  `review` and `blocked` candidates do not get packs generated automatically.

---

## 9. Claude Code Ready

`agents/development_pack_status.evaluate_development_pack_readiness()` applies a 10-point checklist to determine whether a candidate is `claude_code_ready`:

| Check | Condition |
|-------|-----------|
| `implementation_spec.json` exists | pack file present and non-empty |
| `claude_task_prompt.md` exists | pack file present and non-empty |
| `data_contract.yaml` exists | pack file present and non-empty |
| `feature_plan.yaml` exists | pack file present and non-empty |
| `analysis_plan.yaml` exists | pack file present and non-empty |
| `acceptance_tests.md` exists | pack file present and non-empty |
| `automation_risk` not high | `low` or `medium` only |
| `required_secrets` empty | no env-vars needed at runtime |
| no experimental tags | none of `streetview_cv`, `deep_learning`, `satellite_cv`, `experimental` |
| gate not failed | overall gate is `pass` or `warning`, not `fail` |

`shortlist_status = "review"` with non-file blocking reasons → `development_pack_status = "review_required"`.

Candidates with `claude_code_ready = true` get a **Claude Code Ready** badge in the UI and a visible **Copy Task Prompt** button on the candidate card.

---

## 10. Run Output Files

After a candidate factory run, the following files are written to `output/`:

| File | Content |
|------|---------|
| `candidate_cards.json` | All candidate cards with scores, gate_status, claude_code_ready |
| `feasibility_report.json` | Summary counts (ready/review/blocked/risk tiers) + per-candidate subchecks |
| `development_pack_index.json` | Mapping of candidate_id → pack status + file paths |
| `gate_trace.json` | Per-candidate gate trace + repair history + scores |
| `topic_screening.json` | Candidates formatted for HITL / literature stage |
| `repair_history.json` | Flat log of all repair actions |
| `development_packs/{id}/` | One directory per ready candidate |

These files are also accessible via the REST API:

```
GET /runs/{run_id}/feasibility-report
GET /runs/{run_id}/development-pack-index
GET /runs/{run_id}/candidates/{id}/claude-task-prompt
GET /runs/{run_id}/development-packs/{id}
```

---

## End-to-end evaluation

```bash
python scripts/eval_candidate_factory.py \
  --template built_environment_health \
  --domain "Built environment and health outcomes" \
  --max-candidates 40 \
  --enable-experimental false \
  --check-thresholds \
  --output output/eval_candidate_factory.json
```

### v1 thresholds

| Metric | Threshold |
|--------|-----------|
| `candidate_count` | ≥ 20 |
| `score_completion_rate` | ≥ 95% |
| `implementation_spec_completion_rate` | ≥ 85% |
| `development_pack_ready_rate` | ≥ 80% |
| `low_or_medium_automation_risk_rate` | ≥ 70% |
| `experimental_candidate_count` | = 0 (when experimental disabled) |
