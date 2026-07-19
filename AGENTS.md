# AGENTS.md

## Project Overview

Auto-PI is a Codex-native research workflow for built-environment health studies. The active product path is a repository Skill plus a manifest-backed Python runtime.

## Active Entry Points

```bash
python .agents/skills/run-auto-pi-research/scripts/autopi.py --help
python -m autopi_engine --help
autopi --help
```

The CLI supports:

- `init --domain [--run-id]`
- `list`
- `status --run-id`
- `submit --run-id --kind --file`
- `advance --run-id`
- `retry --run-id`
- `abort --run-id`

Commands write JSON to stdout and diagnostics to stderr.

## Architecture

- `autopi_engine/` owns manifest state, CLI, contracts, locking, resume, and deterministic workflow stages.
- `.agents/skills/run-auto-pi-research/` owns Codex Skill instructions, launcher, and input contracts.
- `agents/` retains deterministic candidate, literature, source registry, and development-pack utilities.
- `models/` retains Pydantic contracts used by deterministic utilities.
- `config/research_capability_registry.yaml` replaces the old skill-registry naming.

## Workflow

The engine stores runs under `AUTOPI_DATA_ROOT/runs/<run_id>`:

```text
manifest.json
inputs/
output/
data/
config/
```

Python stages run only when the manifest is `active`. Codex must provide explicit inputs when the manifest is `awaiting_input`.

## Supported Scope

Only `built_environment_health` is exposed through the Skill CLI. Other templates may remain in `config/research_templates/` but must not be surfaced as CLI options without a separate implementation pass.

Do not reintroduce FastAPI, Streamlit, LangGraph, LangChain, DeepSeek, Gemini, Railway, Docker deployment, or external LLM provider runtime in this branch.

## Testing

```bash
pytest -q
python C:/Users/wahol/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/run-auto-pi-research
```
