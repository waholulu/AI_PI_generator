---
name: run-auto-pi-research
description: Run the Auto-PI built-environment health research workflow inside Codex using a manifest-backed local Python engine. Use when a user asks Codex to start, resume, advance, inspect, or complete an Auto-PI research run, including field scan planning, candidate selection, novelty assessment, literature planning, and drafting the final research memo.
---

# Run Auto-PI Research

## Overview

Use the bundled launcher to drive Auto-PI runs from Codex. The Python engine performs deterministic retrieval, candidate generation, validation, artifact writing, and resume; Codex supplies judgment-heavy inputs at explicit `awaiting_input` checkpoints.

## Quick Start

From the repository root:

```bash
python .agents/skills/run-auto-pi-research/scripts/autopi.py init --domain "Built environment exposure and health outcomes"
python .agents/skills/run-auto-pi-research/scripts/autopi.py status --run-id <run_id>
python .agents/skills/run-auto-pi-research/scripts/autopi.py submit --run-id <run_id> --kind search_plan --file plan.json
python .agents/skills/run-auto-pi-research/scripts/autopi.py advance --run-id <run_id>
```

All commands print JSON to stdout. Treat stderr as diagnostics.

## Workflow

1. Initialize a run with `init --domain`.
2. Check `pending_action` in `status`.
3. When `kind` is `search_plan`, `novelty_search_plan`, or `literature_search_plan`, write a JSON `SearchPlan`.
4. When `kind` is `candidate_selection`, choose one `candidate_id` from `output/topic_screening.json`.
5. When `kind` is `novelty_assessment`, inspect `output/novelty_search_results.json` and write a `NoveltyAssessment`.
6. When `kind` is `novelty_decision`, submit `proceed`, `reselect`, or `abort`.
7. When `kind` is `research_memo`, write the final Markdown memo with the required headings and citation keys from `output/references.bib`.
8. After every `submit`, run `advance --run-id <run_id>` until the run returns `awaiting_input`, `completed`, `completed_with_warnings`, `failed`, or `aborted`.

## Contracts

Read `references/workflow-contract.md` before producing any JSON input. Read `references/research-review-rubric.md` before producing a novelty assessment or final memo.

## Resume And Safety

Runs live under `AUTOPI_DATA_ROOT/runs/<run_id>` and can resume across Codex threads. Never edit `manifest.json` manually; use `submit`, `advance`, `retry`, or `abort`.
