# Auto-PI

Auto-PI is now a Codex-native research workflow for built-environment health studies. It combines a repository-level Codex Skill with a deterministic Python engine:

- Codex handles research judgment: search plans, candidate choice, novelty assessment, literature plan, and the final memo.
- Python handles deterministic work: OpenAlex/arXiv retrieval, candidate generation, manifest state, artifact writing, development packs, and validation.

## Quick Start

```bash
pip install -e .

python .agents/skills/run-auto-pi-research/scripts/autopi.py init \
  --domain "Built environment exposure and health outcomes"

python .agents/skills/run-auto-pi-research/scripts/autopi.py status --run-id <run_id>
```

When the manifest returns `awaiting_input`, submit the requested JSON or Markdown file:

```bash
python .agents/skills/run-auto-pi-research/scripts/autopi.py submit \
  --run-id <run_id> \
  --kind search_plan \
  --file plan.json

python .agents/skills/run-auto-pi-research/scripts/autopi.py advance --run-id <run_id>
```

The packaged console entrypoint is equivalent:

```bash
autopi init --domain "Built environment exposure and health outcomes"
```

## Commands

- `init --domain [--run-id]`
- `list`
- `status --run-id`
- `submit --run-id --kind --file`
- `advance --run-id`
- `retry --run-id`
- `abort --run-id`

Commands emit JSON on stdout and diagnostics on stderr.

## Run Storage

Runs are stored under:

```text
AUTOPI_DATA_ROOT/runs/<run_id>/
  manifest.json
  inputs/
  output/
  data/
  config/
```

`manifest.json` is the resume contract across Codex threads. Do not edit it manually.

## Supported Flow

This branch intentionally exposes only the stable `built_environment_health` workflow. Other template/config files remain in the repository for future work, but the Skill CLI does not expose template selection.

The old FastAPI server, Streamlit UI, LangGraph orchestration, external LLM provider layer, and cloud deployment assets have been removed from the active product path.

## Environment

Python 3.11-3.12 is supported.

| Variable | Purpose |
| --- | --- |
| `AUTOPI_DATA_ROOT` | Root for run manifests and artifacts |
| `OPENALEX_EMAIL` | Optional OpenAlex polite-pool email |
| `OPENALEX_API_KEY` | Optional OpenAlex API key |
| `LITERATURE_FINAL_LIMIT` | OpenAlex papers retained during literature harvest |
| `ARXIV_SEARCH_ENABLED` | Enable arXiv supplement |
| `ARXIV_FINAL_LIMIT` | arXiv papers retained |

## Tests

```bash
pytest -q
python C:/Users/wahol/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/run-auto-pi-research
```
