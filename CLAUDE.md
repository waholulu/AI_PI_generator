# CLAUDE.md

## Project Overview

Auto-PI (AI PI Generator) is a multi-agent research automation pipeline that assists with early-stage academic research. It chains 5 specialized agents via LangGraph to scan literature, generate research topics, harvest papers, draft documents, and collect datasets — with a human-in-the-loop checkpoint after ideation.

## Quick Start

```bash
# Requires Python 3.10
uv sync            # or: pip install -e .
cp .env.example .env
# Fill in API keys in .env (at minimum GEMINI_API_KEY)
python main.py     # interactive CLI
```

Streamlit UI: `streamlit run ui/app.py`
API server (local): `uvicorn api.server:app --reload`

## Architecture

### Pipeline (linear, LangGraph orchestrated)

```
START → field_scanner → ideation → idea_validator → [HITL checkpoint] → literature → drafter → data_fetcher → END
```

- **Field Scanner** (`agents/field_scanner_agent.py`) — Scans OpenAlex for landscape analysis, extracts high-traction concepts
- **Ideation** (`agents/ideation_agent.py`) — Generates 30 candidates → screens to 10 → ranks top 3 → enriches → produces `research_plan.json`
- **Idea Validator** (`agents/idea_validator_agent.py`) — Validates top candidates for novelty via OpenAlex recent-paper search and data availability via registry matching; auto-substitutes failed ideas from backup pool
- **Literature** (`agents/literature_agent.py`) — Multi-query OpenAlex search, dedup, evidence cards, BibTeX
- **Drafter** (`agents/drafter_agent.py`) — Synthesizes academic draft from plan + literature via Gemini Pro
- **Data Fetcher** (`agents/data_fetcher_agent.py`) — Collects/simulates public datasets (currently mock)

### Shared Utilities

- **KeywordPlanner** (`agents/keyword_planner.py`) — LLM-based query reformulation for OpenAlex searches
- **MemoryRetriever** (`agents/memory_retriever.py`) — CSV-based persistent idea memory to avoid repeating failed directions
- **Settings** (`agents/settings.py`) — Centralized path configuration; all paths relative to `AUTOPI_DATA_ROOT`
- **Logging** (`agents/logging_config.py`) — Structured JSON logging; use `get_logger(__name__)` in all agents

### Orchestrator (`agents/orchestrator.py`)

- State (`ResearchState`) passes data between agents via **file path references**, not inline data
- Checkpointer: SQLite locally, PostgreSQL in cloud (auto-selected based on `DATABASE_URL`)
- HITL interrupt before the `literature` node

## Key Directories

```
agents/     Core agent implementations
api/        FastAPI REST API (server.py, run_manager.py, log_store.py, models.py)
config/     Research plan template (research_plan.json)
data/       Generated artifacts: literature cards, raw data
memory/     Persistent idea memory (CSV + JSONL)
output/     Final outputs: field scan, drafts, BibTeX, checkpoints
prompts/    System prompts for LLM calls + data source registry
scripts/    CLI wrappers for cloud operations
tests/      Pytest test suite
ui/         Streamlit monitoring interface
```

## Tech Stack

- **Python 3.10** (strictly pinned)
- **LangGraph** — Agent orchestration, state graph, checkpointing
- **LangChain** — LLM abstraction layer
- **FastAPI + uvicorn** — REST API for cloud management
- **Google Gemini** — Primary LLM (`gemini-2.0-flash-lite` for fast tasks, `gemini-2.5-pro` for complex tasks)
- **Anthropic Claude** — Fallback for data fetcher
- **pyalex** — OpenAlex API client
- **Pydantic** — Data validation schemas throughout
- **SQLite/PostgreSQL** — LangGraph state persistence (SQLite local, Postgres cloud)

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google AI API key (required) |
| `GEMINI_FAST_MODEL` | Fast model name (default: `gemini-2.0-flash-lite`) |
| `GEMINI_PRO_MODEL` | Pro model name (default: `gemini-2.5-pro`) |
| `OPENALEX_API_KEY` | OpenAlex API key |
| `OPENALEX_EMAIL` | OpenAlex polite pool email |
| `OPENALEX_QUERY_REWRITE_ENABLED` | Toggle LLM query reformulation |
| `LITERATURE_FINAL_LIMIT` | Papers kept after dedup (default: 3) |
| `AUTOPI_DATA_ROOT` | Root dir for all outputs (default: `.`; set to `/app/storage` in cloud) |
| `DATABASE_URL` | PostgreSQL URL for cloud checkpointing; omit for SQLite |
| `LOG_LEVEL` | Logging level (default: `INFO`) |
| `NOVELTY_CHECK_YEARS` | Novelty check lookback years (default: `2`) |
| `NOVELTY_QUERIES_PER_IDEA` | Search queries per idea for novelty check (default: `3`) |
| `NOVELTY_RESULTS_PER_QUERY` | Papers returned per novelty query (default: `10`) |
| `DATA_REGISTRY_FUZZY_THRESHOLD` | Fuzzy match cutoff for data source registry (default: `0.6`) |
| `VALIDATION_MAX_SUBSTITUTIONS` | Max idea substitution rounds (default: `2`) |

## Testing

```bash
pytest                                    # All tests (excludes live API)
pytest tests/ -m "not live_openalex and not live_llm"  # Mock tests only
pytest tests/test_field_scan_live.py -v   # Live OpenAlex test
```

**Markers** (defined in `pyproject.toml`):
- `live_openalex` — Requires `OPENALEX_API_KEY` and `OPENALEX_EMAIL`
- `live_llm` — Requires `GEMINI_API_KEY`

## Conventions

- **State passing**: Agents communicate via file paths in `ResearchState`, not by passing data inline
- **Paths**: Always use `agents/settings.py` helpers (never hardcode `"output/"` etc.)
- **Logging**: Use `get_logger(__name__)` from `agents/logging_config.py` — no `print()` in agents
- **Pydantic schemas**: All data contracts (topics, plans, evidence cards) use strict Pydantic models
- **Idea memory**: CSV append-only log + JSONL archive for enriched candidates
- **Fallback/degradation**: Agents handle missing API keys gracefully; KeywordPlanner falls back to raw input
- **UI strings**: `main.py` uses Chinese for CLI prompts and status messages
- **Output files**: Generated artifacts go to `output/` and `data/`, both gitignored

## Cloud Operations (via Claude Code)

The pipeline is deployed as a REST API on Railway. Use these commands to manage it:

### Deploy

```bash
# Push to main triggers auto-deploy via GitHub Actions
git push origin main

# Or deploy manually
railway up --detach

# Trigger via GitHub Actions
gh workflow run deploy.yml
```

### Run the pipeline

```bash
export AUTOPI_API_URL=https://<your-app>.up.railway.app

# Start a run
./scripts/start_run.sh "GeoAI and Urban Planning"
# Returns: {"run_id": "...", "status": "starting", ...}

# Check status (poll until awaiting_approval or completed)
./scripts/status.sh <run_id>

# View logs
./scripts/logs.sh <run_id>
./scripts/logs.sh <run_id> ERROR   # filter by level

# List all runs
./scripts/list_runs.sh
```

### Handle the HITL checkpoint

After ideation completes, the run pauses with `status: "awaiting_approval"`.

```bash
# Review the research plan (it's an output file)
./scripts/outputs.sh <run_id>
curl -s "$AUTOPI_API_URL/runs/<run_id>/outputs/topic_screening.json" | python3 -m json.tool

# Approve (continue to literature → drafter → data_fetcher)
./scripts/approve.sh <run_id>

# Or reject (abort the run)
curl -s -X POST "$AUTOPI_API_URL/runs/<run_id>/reject" | python3 -m json.tool
```

### Inspect state and outputs

```bash
# Full LangGraph checkpoint state
curl -s "$AUTOPI_API_URL/runs/<run_id>/state" | python3 -m json.tool

# Download output files
curl -O "$AUTOPI_API_URL/runs/<run_id>/outputs/Draft_v1.md"
curl -O "$AUTOPI_API_URL/runs/<run_id>/outputs/research_context.json"
```

### Manage secrets on Railway

```bash
railway variables set GEMINI_API_KEY=...
railway variables set OPENALEX_API_KEY=...
railway variables set OPENALEX_EMAIL=...
railway variables list
```

### View Railway logs

```bash
railway logs          # tail live logs
railway logs --tail 100
```

## Docker (local testing)

```bash
docker build -t autopi .
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/storage:/app/storage \
  -e AUTOPI_DATA_ROOT=/app/storage \
  autopi
```

## Known Limitations

- **Data Fetcher**: Mock implementation only — no real public data adapters yet
- **Drafter prompt**: `prompts/academic_drafter.txt` is a placeholder
- **Literature sources**: Only OpenAlex (no arXiv, Crossref, or Unpaywall integration yet)
- **Run isolation**: All runs share the same output directory (per `AUTOPI_DATA_ROOT`)
