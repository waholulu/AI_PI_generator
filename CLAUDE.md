# CLAUDE.md

## Project Overview

Auto-PI (AI PI Generator) is a multi-agent research automation pipeline that assists with early-stage academic research. It chains 5 specialized agents via LangGraph to scan literature, generate research topics, harvest papers, draft documents, and collect datasets — with a human-in-the-loop checkpoint after ideation.

## Quick Start

```bash
# Requires Python 3.10
uv sync            # or: pip install -e .
cp .env.example .env
# Fill in API keys in .env (at minimum GEMINI_API_KEY)
python main.py                         # Level 2 interactive CLI (default)
python main.py --mode level_1 --user-topic inputs/my_topic.yaml   # Level 1
python main.py --legacy-ideation       # Legacy pipeline (V0)
python main.py --help                  # All CLI flags
```

Streamlit UI: `streamlit run ui/app.py`
API server (local): `uvicorn api.server:app --reload`

## Architecture

### Pipeline (conditional, LangGraph orchestrated)

```
# Level 2 (domain auto-generation, default):
START → field_scanner → ideation → idea_validator → [HITL checkpoint] → literature → drafter → data_fetcher → END

# Level 1 (user-provided structured topic):
START → ideation → idea_validator → [HITL checkpoint] → literature → drafter → data_fetcher → END
```

- **Field Scanner** (`agents/field_scanner_agent.py`) — Scans OpenAlex for landscape analysis, extracts high-traction concepts
- **Ideation** (`agents/ideation_agent.py`) — Thin router → V2 (default) or V0 (legacy). V2 runs 7-gate evaluation + reflection loop; produces `research_plan.json` + `tentative_pool.json`
- **Idea Validator** (`agents/idea_validator_agent.py`) — Validates top candidates for novelty via OpenAlex recent-paper search and data availability via registry matching; auto-substitutes failed ideas from backup pool
- **Literature** (`agents/literature_agent.py`) — Multi-query OpenAlex search, dedup, evidence cards, BibTeX
- **Drafter** (`agents/drafter_agent.py`) — Synthesizes academic draft from plan + literature via Gemini Pro
- **Data Fetcher** (`agents/data_fetcher_agent.py`) — Collects/simulates public datasets (currently mock)

### Module 1 Upgrade (branch: `claude/auto-pi-module1-upgrade-JLv30`)

The ideation engine now features structured slot schema, a seven-gate evaluation system, per-topic reflection loop, and a TENTATIVE topic pool:

#### Structured Topic Schema (`models/topic_schema.py`)
Topics are represented as five-dimensional slot structures: `ExposureX`, `OutcomeY`, `SpatialScope`, `TemporalScope`, `IdentificationStrategy`. The `Topic` model provides a `four_tuple_signature()` method (MD5 hash of X_family + Y_family + geography + method) used for oscillation detection.

#### Seven-Gate Evaluation (`config/gate_config.yaml`, `agents/rule_engine.py`)

| Gate | Name | Type | Judge |
|------|------|------|-------|
| G1 | mechanism_plausibility | refinable | LLM (threshold 4/5) |
| G2 | scale_alignment | **hard_blocker** | rule_engine (rank diff ≤ 4) |
| G3 | data_availability | **hard_blocker** | catalog lookup |
| G4 | identification_validity | refinable | rule_engine + LLM |
| G5 | novelty | refinable | OpenAlex 4-tuple |
| G6 | automation_feasibility | **hard_blocker** | skill_registry |
| G7 | contribution_clarity | refinable | LLM (threshold 4/5) |

#### Reflection Loop (`agents/reflection_loop.py`)
Per-topic iterative refinement (max 3 rounds) with:
- Hard-blocker fail (G2/G3/G6) → immediate **REJECTED** (no LLM cost)
- All refinable pass → **ACCEPTED**
- 1-2 refinable fail → **REFINE** (next round)
- ≥3 refinable fail → **TENTATIVE** (human review pool)
- Oscillation detection (same 4-tuple for 3 rounds) → **TENTATIVE**
- Early stop (score delta < 0.5 after min 2 rounds) → **TENTATIVE**

Traces persisted to `output/ideation_traces/{topic_id}_trace.json`.

#### Dual Budget Guard (`agents/budget_tracker.py`)
- Per-run budget ($1.50 default) + per-topic budget ($0.10 default)
- Three tiers: warn at 70%, stop new topics at 90%, kill switch at 100%

#### Dual-Mode Entry (`agents/ideation_agent_v2.py`)
- **Level 1**: User provides `--user-topic path.yaml` → validates structured topic → hard-blockers checked → `max_rounds=1` reflection → raises `HITLInterruption` on failure
- **Level 2**: Domain text → generate 30 seeds → parallel reflection loop (5 workers) → rank ACCEPTED → write TENTATIVE pool

#### TENTATIVE Topic Status
Topics with ≤2 refinable gate failures are held in `output/tentative_pool.json` rather than auto-rejected. The Streamlit UI has a dedicated **TENTATIVE Review** tab with Promote / Kill / Re-run buttons.

#### Backward Compatibility
- `--legacy-ideation` flag (or `LEGACY_IDEATION=1` env) → routes to `IdeationAgentV0`
- `legacy_six_gates` field present in every candidate in `topic_screening.json`
- Feature flags in `config/reflection_config.yaml` allow per-feature rollback
- Schema re-exports in `agents/ideation_agent.py` keep existing import paths valid

### Shared Utilities

- **KeywordPlanner** (`agents/keyword_planner.py`) — LLM-based query reformulation for OpenAlex searches
- **MemoryRetriever** (`agents/memory_retriever.py`) — CSV-based persistent idea memory to avoid repeating failed directions
- **Settings** (`agents/settings.py`) — Centralized path configuration; all paths relative to `AUTOPI_DATA_ROOT`
- **Logging** (`agents/logging_config.py`) — Structured JSON logging; use `get_logger(__name__)` in all agents
- **RuleEngine** (`agents/rule_engine.py`) — Zero-LLM-cost deterministic gate checks (G2/G3/G4/G6); `run_hard_blockers()` accepts `use_role_based_g3=True` for candidate factory path
- **BudgetTracker** (`agents/budget_tracker.py`) — Thread-safe dual-layer LLM cost guard
- **OpenAlexVerifier** (`agents/openalex_verifier.py`) — Four-tuple novelty check via pyalex
- **ReflectionLoop** (`agents/reflection_loop.py`) — Per-topic iterative refinement with trace persistence
- **DevelopmentPackStatus** (`agents/development_pack_status.py`) — `evaluate_development_pack_readiness()` determines `claude_code_ready` via 10-point checklist
- **CandidateOutputWriter** (`agents/candidate_output_writer.py`) — Writes `feasibility_report.json`, `development_pack_index.json`, `gate_trace.json` per run; called automatically by candidate factory

### Orchestrator (`agents/orchestrator.py`)

- State (`_Module1State`, extends `ResearchState` with `total=False` fields) passes data between agents via **file path references**, not inline data
- Conditional fork at `START`: when `user_topic_path` is set, `field_scanner` is skipped (Level 1 mode)
- Checkpointer: SQLite locally, PostgreSQL in cloud (auto-selected based on `DATABASE_URL`)
- HITL interrupt before the `literature` node

## Key Directories

```
agents/         Core agent implementations
agents/_legacy/ Legacy IdeationAgentV0 (observation period; may be deleted later)
api/            FastAPI REST API (server.py, run_manager.py, log_store.py, models.py)
config/         YAML configs (gate_config, reflection_config, data_sources, etc.)
                + research_plan.json template
data/           Generated artifacts: literature cards, raw data
memory/         Persistent idea memory (CSV + JSONL)
models/         Pydantic models (topic_schema.py, data_source.py)
output/         Final outputs: field scan, drafts, BibTeX, checkpoints,
                candidate_cards.json, feasibility_report.json,
                development_pack_index.json, gate_trace.json,
                repair_history.json, development_packs/{candidate_id}/,
                ideation_traces/, tentative_pool.json, ideation_run_summary.json
prompts/        System prompts for LLM calls + data source registry
scripts/        CLI wrappers for cloud operations + diagnostic report generator
tests/          Pytest test suite
ui/             Streamlit monitoring interface (3-tab layout)
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
# Run all mock tests (no API keys required) — use uv run for correct venv
uv run python -m pytest tests/ -m "not live_openalex and not live_llm" -q

# Run only Module 1 upgrade tests
uv run python -m pytest tests/test_topic_schema.py tests/test_settings_new_paths.py \
  tests/test_rule_engine.py tests/test_budget_tracker.py tests/test_openalex_verifier.py \
  tests/test_reflection_loop.py tests/test_ideation_v2.py tests/test_hitl_helpers_v2.py \
  tests/test_integration_module1_v2.py -v

# Candidate factory eval — enforces thresholds in CI
uv run python scripts/eval_candidate_factory.py \
  --template built_environment_health \
  --max-candidates 40 \
  --enable-experimental false \
  --check-thresholds

# Live tests (require API keys)
uv run python -m pytest tests/test_field_scan_live.py -v   # Live OpenAlex test
```

**Markers** (defined in `pyproject.toml`):
- `live_openalex` — Requires `OPENALEX_API_KEY` and `OPENALEX_EMAIL`
- `live_llm` — Requires `GEMINI_API_KEY`

### Module 1 Diagnostic Report

```bash
# Generate gate pass rates, refine frequency, budget stats from traces
python scripts/generate_diagnostic_report.py
# Output: output/diagnostic_report.md
```

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
