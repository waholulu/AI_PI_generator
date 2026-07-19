# CLAUDE.md

## Project Overview

Auto-PI is a Codex-native research workflow for built-environment health studies. This branch replaces the old LangGraph/API/UI product path with a repository-level Codex Skill and a manifest-backed deterministic Python runtime.

The active workflow is:

```text
Codex judgment inputs
  -> autopi_engine deterministic stages
  -> AUTOPI_DATA_ROOT/runs/<run_id> artifacts
```

## Quick Start

Use Python 3.11-3.12.

```bash
pip install -e .

python .agents/skills/run-auto-pi-research/scripts/autopi.py init \
  --domain "Built environment exposure and health outcomes"

python .agents/skills/run-auto-pi-research/scripts/autopi.py status --run-id <run_id>
```

The installed console script is equivalent:

```bash
autopi --help
```

## Active Commands

```bash
autopi init --domain "Built environment and health" [--run-id <run_id>]
autopi list
autopi status --run-id <run_id>
autopi submit --run-id <run_id> --kind <kind> --file <path>
autopi advance --run-id <run_id>
autopi retry --run-id <run_id>
autopi abort --run-id <run_id>
```

Commands emit JSON on stdout and diagnostics on stderr.

## Runtime Contract

Runs are stored under:

```text
AUTOPI_DATA_ROOT/runs/<run_id>/
  manifest.json
  inputs/
  output/
  data/
  config/
```

`manifest.json` is the cross-thread resume contract. Do not edit it manually; use the CLI.

Statuses:

- `active`
- `awaiting_input`
- `completed`
- `completed_with_warnings`
- `failed`
- `aborted`

Codex must provide explicit inputs when the manifest is `awaiting_input`.

## Awaited Input Kinds

- `search_plan`
- `candidate_selection`
- `novelty_search_plan`
- `novelty_assessment`
- `novelty_decision`
- `literature_search_plan`
- `research_memo`

Use `.agents/skills/run-auto-pi-research/references/workflow-contract.md` for exact JSON and memo contracts.

## Active Architecture

- `autopi_engine/` - CLI, manifest storage, file locking, contracts, workflow stages, memo validation.
- `.agents/skills/run-auto-pi-research/` - Codex Skill instructions, launcher script, workflow references.
- `agents/` - deterministic candidate factory, literature harvesting, source registry, rule engine, development-pack utilities.
- `models/` - Pydantic schemas used by deterministic modules.
- `config/research_capability_registry.yaml` - deterministic capability registry formerly represented as `skill_registry.yaml`.

## Supported Scope

Only the stable `built_environment_health` flow is exposed. Other templates may remain in `config/research_templates/` as preserved assets, but must not be surfaced by the CLI without a separate implementation pass.

Do not reintroduce these removed runtime surfaces in this branch:

- FastAPI / uvicorn API server
- Streamlit UI
- LangGraph orchestration
- LangChain provider layer
- DeepSeek/Gemini external LLM runtime
- Railway/Docker deployment path

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `AUTOPI_DATA_ROOT` | Root directory for run manifests and artifacts |
| `OPENALEX_EMAIL` | Optional OpenAlex polite-pool email |
| `OPENALEX_API_KEY` | Optional OpenAlex API key |
| `OPENALEX_QUERY_REWRITE_PER_QUERY_LIMIT` | Per-query OpenAlex cap |
| `LITERATURE_FINAL_LIMIT` | OpenAlex papers retained in literature harvest |
| `ARXIV_SEARCH_ENABLED` | Enables arXiv supplement |
| `ARXIV_FINAL_LIMIT` | arXiv papers retained |
| `LOG_LEVEL` | Logging verbosity |

## Tests And Validation

```bash
py -3.11 -m pytest tests -q
python C:/Users/wahol/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/run-auto-pi-research
py -3.11 -m pip install -e . --dry-run
```

For a no-network full workflow smoke, mock `multi_search_openalex` and `multi_search_arxiv` in `autopi_engine.workflow` and `agents.literature_agent`, then advance from `search_plan` through `research_memo` to `completed`.

## Notes For Future Agents

Keep behavior manifest-first and deterministic. Codex should do research judgment through explicit submitted files; Python should do repeatable retrieval, validation, artifact generation, and state transitions.
