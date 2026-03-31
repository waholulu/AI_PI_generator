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
START → field_scanner → ideation → [HITL checkpoint] → literature → drafter → data_fetcher → END
```

- **Field Scanner** (`agents/field_scanner_agent.py`) — Scans OpenAlex for landscape analysis; uses KeywordPlanner to reformulate domain into multi-query searches; extracts high-traction concepts, citation counts, year trends. Outputs `output/field_scan.json`. Fails gracefully if OpenAlex is unavailable.
- **Ideation** (`agents/ideation_agent.py`) — 5-step pipeline: generate 30 candidates (Gemini Fast) → gate-screen to 10 (impact/novelty/publishability) → rank top 3 (Gemini Pro) → enrich with full specs → produce `research_plan.json`. Uses MemoryRetriever to avoid repeating failed directions. Outputs: `topic_screening.json`, `research_plan.json`, `research_context.json`, `topic_ranking.csv`, `ideas_graveyard.json`, JSONL archive.
- **Literature** (`agents/literature_agent.py`) — Multi-query OpenAlex search, dedup by openalex_id/DOI, generates evidence cards (metadata, abstract, citations, open-access info), attempts PDF download, generates BibTeX. Outputs: `data/literature/cards/*.json`, `data/literature/index.json`, `output/references.bib`.
- **Drafter** (`agents/drafter_agent.py`) — Reads research plan + evidence cards, synthesizes structured markdown draft via Gemini Pro. Falls back to placeholder if LLM fails. Outputs: `output/Draft_v1.md`.
- **Data Fetcher** (`agents/data_fetcher_agent.py`) — Mock implementation; extracts data sources from research plan, creates manifest with placeholder datasets, generates `run_index.json` linking all artifacts. Outputs: `data/raw/manifest.json`.

### Shared Utilities

- **KeywordPlanner** (`agents/keyword_planner.py`) — LLM-based query reformulation into structured OpenAlex search queries; falls back to raw domain input if LLM unavailable.
- **MemoryRetriever** (`agents/memory_retriever.py`) — CSV append-only log + JSONL archive for tracking past candidates/rejections; supports domain-based retrieval to avoid repeating failed directions. Files: `memory/idea_memory.csv`, `memory/enriched_top_candidates.jsonl`.
- **OpenAlex Utils** (`agents/openalex_utils.py`) — Work metadata extraction, abstract reconstruction, PDF download helpers.
- **Settings** (`agents/settings.py`) — Centralized path configuration; all paths relative to `AUTOPI_DATA_ROOT`.
- **Logging** (`agents/logging_config.py`) — Structured JSON logging with `run_id`/`node`/`step` context fields; use `get_logger(__name__)` in all agents.

### Orchestrator (`agents/orchestrator.py`)

- State (`ResearchState`) passes data between agents via **file path references**, not inline data.
- Checkpointer: SQLite locally (`output/checkpoints.sqlite`), PostgreSQL in cloud (auto-selected based on `DATABASE_URL`).
- HITL interrupt before the `literature` node.

## Key Directories

```
agents/     Core agent implementations + shared utilities
api/        FastAPI REST API (server.py, run_manager.py, log_store.py, models.py)
config/     Research plan template (research_plan.json)
data/       Generated artifacts
  literature/cards/   Per-paper evidence card JSON files
  literature/pdfs/    Downloaded PDF files
  raw/                Dataset manifests
memory/     Persistent idea memory (idea_memory.csv + enriched_top_candidates.jsonl)
output/     Final outputs: field_scan.json, Draft_v1.md, references.bib, run_index.json
prompts/    System prompts for LLM calls (academic_drafter.txt)
scripts/    CLI wrappers for cloud operations
tests/      Pytest test suite (23 files)
ui/         Streamlit monitoring interface (app.py)
```

## Tech Stack

- **Python 3.10** (strictly pinned)
- **LangGraph 1.0.10+** — Agent orchestration, state graph, checkpointing, HITL interrupts
- **LangChain** — LLM abstraction layer (Anthropic + Google adapters)
- **FastAPI 0.115+ + uvicorn** — REST API for cloud management
- **Google Gemini** — Primary LLM (`gemini-2.0-flash-lite` for fast tasks, `gemini-2.5-pro` for complex tasks)
- **Anthropic Claude** — Fallback for data fetcher
- **pyalex 0.21+** — OpenAlex API client
- **Pydantic** — Strict data validation schemas throughout (LightCandidateTopic, TopicScore, QuantitativeSpecs, ResearchPlanSchema, KeywordPlan, etc.)
- **SQLite/PostgreSQL** — LangGraph state persistence (SQLite local, Postgres cloud)
- **Streamlit 1.40+** — Web monitoring UI

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `GEMINI_API_KEY` | Google AI API key (required; also `GOOGLE_API_KEY`) | — |
| `GEMINI_FAST_MODEL` | Fast model for screening/generation | `gemini-2.0-flash-lite` |
| `GEMINI_PRO_MODEL` | Pro model for ranking/drafting | `gemini-2.5-pro` |
| `OPENALEX_API_KEY` | OpenAlex API key | — |
| `OPENALEX_EMAIL` | OpenAlex polite-pool email | — |
| `OPENALEX_QUERY_REWRITE_ENABLED` | Toggle LLM query reformulation | `true` |
| `OPENALEX_QUERY_REWRITE_MAX_QUERIES` | Max queries per pool | `10` |
| `OPENALEX_QUERY_REWRITE_PER_QUERY_LIMIT` | Results per query | `20` |
| `OPENALEX_QUERY_REWRITE_MODEL` | Model override for query rewrite | GEMINI_FAST_MODEL |
| `MIN_CANDIDATE_TOPICS` | Candidates to generate in ideation | `30` |
| `INITIAL_SCREEN_TOPN` | Finalists after initial screening | `10` |
| `FINAL_TOP_N` | Top candidates for enrichment | `3` |
| `SCORING_MODEL` | Model for screening: `"pro"` or `"fast"` | `pro` |
| `LITERATURE_FINAL_LIMIT` | Papers kept after dedup | `3` |
| `AUTOPI_DATA_ROOT` | Root dir for all outputs | `.` (local) / `/app/storage` (cloud) |
| `DATABASE_URL` | PostgreSQL URL for cloud checkpointing; omit for SQLite | — |
| `LOG_LEVEL` | Logging level | `INFO` |
| `AUTOPI_API_URL` | Cloud API endpoint for remote scripts | `http://localhost:8000` |

## Testing

```bash
pytest                                                        # All tests (excludes live API)
pytest tests/ -m "not live_openalex and not live_llm"        # Mock tests only
pytest tests/test_field_scan_live.py -v                      # Live OpenAlex test
```

**Markers** (defined in `pyproject.toml`):
- `live_openalex` — Requires `OPENALEX_API_KEY` and `OPENALEX_EMAIL`
- `live_llm` — Requires `GEMINI_API_KEY` / `GOOGLE_API_KEY`

**Key test files:** `test_field_scanner.py`, `test_ideation.py`, `test_literature.py`, `test_drafter.py`, `test_orchestrator.py`, `e2e_simple_test.py`, `test_modules_1_and_2.py`.

**Known flakiness:** `test_literature_live.py` may fail with empty OpenAlex results in some environments (not a code bug).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/runs` | Start new pipeline run (returns run_id) |
| `GET` | `/runs` | List all runs |
| `GET` | `/runs/{run_id}/status` | Get run status |
| `GET` | `/runs/{run_id}/state` | Get LangGraph checkpoint state |
| `GET` | `/runs/{run_id}/logs` | Stream logs (filterable by level) |
| `POST` | `/runs/{run_id}/approve` | Approve HITL and resume pipeline |
| `POST` | `/runs/{run_id}/reject` | Abort run |
| `GET` | `/runs/{run_id}/outputs` | List generated files |
| `GET` | `/runs/{run_id}/outputs/{filename}` | Download output file |
| `GET` | `/health` | Health check |

## Conventions

- **State passing**: Agents communicate via file paths in `ResearchState`, not by passing data inline.
- **Paths**: Always use `agents/settings.py` helpers (never hardcode `"output/"` etc.).
- **Logging**: Use `get_logger(__name__)` from `agents/logging_config.py` — no `print()` in agents.
- **Pydantic schemas**: All data contracts (topics, plans, evidence cards) use strict Pydantic models.
- **Idea memory**: CSV append-only log + JSONL archive for enriched candidates.
- **Fallback/degradation**: Agents handle missing API keys gracefully; KeywordPlanner falls back to raw input.
- **UI strings**: `main.py` uses Chinese for CLI prompts and status messages.
- **Output files**: Generated artifacts go to `output/` and `data/`, both gitignored.

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

- **Data Fetcher**: Mock implementation only — no real public data adapters.
- **Drafter prompt**: `prompts/academic_drafter.txt` is a minimal placeholder.
- **Literature sources**: Only OpenAlex (no arXiv, Crossref, or Unpaywall integration yet).
- **Run isolation**: All runs share the same output directory (per `AUTOPI_DATA_ROOT`) — concurrent runs can overwrite each other's files.
- **PDF download**: Attempted from OpenAlex candidate URLs but often unavailable.
