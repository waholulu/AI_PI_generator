# Auto-PI (AI PI Generator)

Auto-PI is a multi-agent research automation pipeline for early-stage academic research.  
It orchestrates field scanning, ideation, validation, literature harvesting, drafting, and data collection with a HITL checkpoint.

At its core is the **Candidate Factory** — a deterministic pipeline that turns a research template into a ranked list of concrete, automatable research designs, each with a full development pack ready for Claude Code to implement.

---

## Two modes

| Mode | What it does | LLM keys required |
|------|-------------|-------------------|
| **Mode 1: Candidate Factory** | Template → 20–40 ranked research designs with development packs | None (deterministic) |
| **Mode 2: Legacy LLM Ideation** | LLM generates ideas → gates → HITL → literature | GEMINI_API_KEY |

Mode 1 is the recommended starting point. Mode 2 extends the pipeline with full literature harvesting and draft writing.

## Quick start

```bash
# 1. Install (Python 3.11–3.12 required)
pip install -e ".[geospatial]"

# 2. Run the test suite
uv run pytest -q -m "not live_openalex and not live_llm"

# 3. Validate the candidate factory meets thresholds
uv run python scripts/eval_candidate_factory.py \
  --template built_environment_health \
  --check-thresholds

# 4. Start the API server
uvicorn api.server:app --reload

# 5. Run E2E acceptance test (server must be running)
python scripts/e2e_acceptance_test.py

# 6. Build Docker image
docker build -t ai-pi-generator .
```

Copy `.env.example` to `.env`. `GEMINI_API_KEY` is only required for Mode 2.

---

## Candidate factory: generate research designs

The candidate factory takes a template and produces 20–40 concrete research candidates, each with:
- Exposure × Outcome × Method specification
- Feasibility gate check (no LLM needed)
- Auto-repair for fixable issues
- Composite score and ranking
- Development pack for Claude Code (for ready candidates)

### Via the UI

```bash
uvicorn api.server:app --reload
# Open http://localhost:8000
```

**User flow:**

1. Open the **New Run** form
2. Enter a domain: `Built environment exposure and health outcomes`
3. Select template: **Built Environment → Health Outcomes**
4. Check **OSMnx** and **Remote sensing** under technology options
5. Leave **Street View** and **Deep Learning** unchecked
6. Click **Generate Candidates**
7. In the **Candidate Review** tab, filter by **Claude Code Ready**
8. Click any ready candidate → **View details**
9. Click **Copy Task Prompt**
10. Paste into a new Claude Code session and let it implement the pipeline

### Via the API

```bash
# Start a run with max_candidates=40 (new default)
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "built_environment_health",
    "domain_input": "Built environment and health outcomes",
    "max_candidates": 40,
    "enable_experimental": false,
    "cloud_constraints": {"no_paid_api": true}
  }'

# List candidates (after run completes)
curl http://localhost:8000/runs/{run_id}/candidates

# Get the Claude Code task prompt for a ready candidate
curl http://localhost:8000/runs/{run_id}/outputs/development_packs/{candidate_id}/claude_task_prompt.md
```

### Via the CLI

```bash
python main.py --mode level_2 --domain "GeoAI and Urban Planning"
```

---

## Architecture

```
# Candidate Factory (deterministic, no LLM):
Template → Composer → Precheck → Repair → Ranker → Dev Pack → Claude Code

# Full pipeline (LLM-orchestrated, LangGraph):
START → [field_scanner] → ideation → idea_validator
      → [HITL checkpoint]
      → literature → drafter → data_fetcher → END
```

Field scanner is skipped in Level 1 mode (`--user-topic path.yaml`).

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google AI API key (required for full pipeline) |
| `OPENALEX_EMAIL` | OpenAlex polite pool email |
| `AUTOPI_DATA_ROOT` | Output root directory (default: `.`) |
| `DATABASE_URL` | PostgreSQL URL for cloud checkpointing |
| `LOG_LEVEL` | Logging level (default: `INFO`) |

See `.env.example` for the full list.

---

## Tests

Four test tiers run in CI order:

| Tier | Command | Keys needed | In CI |
|------|---------|-------------|-------|
| Unit/mock | `uv run pytest -q -m "not live_openalex and not live_llm"` | none | ✓ |
| Candidate factory eval | `uv run python scripts/eval_candidate_factory.py --check-thresholds` | none | ✓ |
| E2E acceptance | `python scripts/e2e_acceptance_test.py` | none (local server) | manual |
| Live OpenAlex | `uv run pytest tests/test_field_scan_live.py` | OPENALEX_API_KEY | manual |

```bash
# All mock tests
uv run pytest tests/ -q -m "not live_openalex and not live_llm"

# Candidate factory eval with threshold enforcement
uv run python scripts/eval_candidate_factory.py \
  --template built_environment_health \
  --max-candidates 40 \
  --enable-experimental false \
  --check-thresholds

# E2E acceptance (requires uvicorn api.server:app --reload in another terminal)
python scripts/e2e_acceptance_test.py
```

**Eval thresholds (v1, enforced in CI):**

| Metric | Threshold |
|--------|-----------|
| `candidate_count` | ≥ 20 |
| `score_completion_rate` | ≥ 95% |
| `implementation_spec_completion_rate` | ≥ 85% |
| `development_pack_ready_rate` | ≥ 80% |
| `low_or_medium_automation_risk_rate` | ≥ 70% |
| `experimental_candidate_count` | = 0 (when experimental disabled) |

**E2E acceptance thresholds:**

| Check | Threshold |
|-------|-----------|
| `candidate_count` | ≥ 20 |
| `exposure_families` | ≥ 6 |
| `pass_or_warning_count` | ≥ 20 |
| `claude_code_ready_count` | ≥ 8 |
| `experimental_candidate_count` | = 0 (when disabled) |

---

## Docker

```bash
docker build -t ai-pi-generator .
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/storage:/app/storage \
  -e AUTOPI_DATA_ROOT=/app/storage \
  ai-pi-generator
```

---

## Docs

- [`docs/candidate_factory.md`](docs/candidate_factory.md) — full pipeline reference
- [`docs/claude_code_development_pack.md`](docs/claude_code_development_pack.md) — using dev packs with Claude Code
- [`docs/experimental_mode.md`](docs/experimental_mode.md) — street view, deep learning, paid APIs
- [`docs/cloud_deployment.md`](docs/cloud_deployment.md) — Railway deployment and secrets
- [`CLAUDE.md`](CLAUDE.md) — full codebase reference for Claude Code
