# Using Development Packs with Claude Code

A development pack is a self-contained directory of files that tells Claude Code exactly what to build for a specific research candidate.  When a candidate is marked **Claude Code Ready**, you can hand the pack to Claude Code and it will implement the full data pipeline autonomously.

---

## What is a development pack?

Each ready candidate gets a `output/development_packs/{candidate_id}/` directory containing:

| File | What Claude Code uses it for |
|------|------------------------------|
| `claude_task_prompt.md` | The task description — paste this to start |
| `implementation_spec.json` | Data sources, feature steps, analysis steps |
| `data_contract.yaml` | Input/output table schema and join key |
| `feature_plan.yaml` | Exposure variables to compute |
| `analysis_plan.yaml` | Regression method and outcome |
| `acceptance_tests.md` | What "done" looks like |

---

## How to start a development session

### From the UI

1. Open the **Candidate Review** tab
2. Filter by **Claude Code Ready**
3. Click **Copy Task Prompt** on any ready card
4. Paste into a new Claude Code session

### From the API

```bash
# Get the prompt directly
curl -s "$AUTOPI_API_URL/runs/{run_id}/candidates/{candidate_id}/claude-task-prompt" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['prompt'])"
```

### From the file system

```bash
cat output/development_packs/{candidate_id}/claude_task_prompt.md
```

---

## Claude Code Ready conditions

A candidate is `claude_code_ready` only when **all** of the following are true:

- All 6 required pack files exist and are non-empty
- `automation_risk` is `low` or `medium` (not `high`)
- `required_secrets` list is empty (no API keys needed)
- No experimental technology tags (`streetview_cv`, `deep_learning`, etc.)
- Gate check is `pass` or `warning` (not `fail`)

If a candidate shows **Review Required** or **Experimental Review Required**, it is not automatically safe for Claude Code.  You can still copy the prompt after confirming the warning dialog in the UI.

---

## What Claude Code will build

The task prompt instructs Claude Code to produce:

| Output | Path |
|--------|------|
| Processed feature table | `data/processed/tract_features.csv` |
| Model results | `output/tables/model_summary.csv` |
| Technical summary | `output/report/technical_summary.md` |

The pipeline must:
- Use only programmatic data access (no manual downloads)
- Run a smoke test on a small geography (Cambridge, MA by default)
- Complete smoke test in under 10 minutes
- Not use paid APIs unless explicitly enabled
- Not store raw street view images

---

## Checking readiness before handing off

```bash
# Check the full development pack for a candidate
curl -s "$AUTOPI_API_URL/runs/{run_id}/development-packs/{candidate_id}" \
  | python3 -m json.tool

# Get the full development pack index for the run
curl -s "$AUTOPI_API_URL/runs/{run_id}/development-pack-index" \
  | python3 -m json.tool
```

The response includes `claude_code_ready`, `blocking_reasons`, and a `readiness_checklist`.

---

## High-risk candidates

Candidates with `automation_risk=high` or experimental tags are **not** automatically Claude Code Ready.  If you want to use them anyway:

1. The UI will show an **Experimental Review Required** badge
2. Clicking **Copy Task Prompt** shows a confirmation with the blocking reasons
3. You must manually confirm before the prompt is copied

This protects against accidentally handing an expensive or credential-dependent task to Claude Code in a cloud environment.
