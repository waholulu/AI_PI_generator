# Cloud Deployment

AutoPI is deployed as a FastAPI REST service on Railway.  The candidate factory, development pack generator, and HITL checkpoint all run in the cloud.  Literature, drafter, and data fetcher stages require LLM API keys and run only when those are set.

---

## Architecture

```
Client
  │
  ▼
FastAPI (api/server.py)
  │
  ├─ /runs          → RunManager (api/run_manager.py)
  ├─ /candidates    → CandidateFactory pipeline
  ├─ /templates     → config/research_templates/
  └─ /health        → liveness probe
```

State is persisted to PostgreSQL (cloud) or SQLite (local) via LangGraph checkpointing.

All generated artifacts go to `AUTOPI_DATA_ROOT` (default: `.`, cloud: `/app/storage`).

---

## Deploy

### Push to main (auto-deploy)

```bash
git push origin main
# GitHub Actions triggers Railway deploy
```

### Manual deploy

```bash
railway up --detach
```

### GitHub Actions manual trigger

```bash
gh workflow run deploy.yml
```

---

## Secrets

Set Railway environment variables:

```bash
# Required
railway variables set GEMINI_API_KEY=...

# Recommended
railway variables set OPENALEX_API_KEY=...
railway variables set OPENALEX_EMAIL=your@email.com

# Experimental (optional)
railway variables set GOOGLE_STREET_VIEW_API_KEY=...
railway variables set MAPILLARY_TOKEN=...

# Cloud storage
railway variables set AUTOPI_DATA_ROOT=/app/storage
railway variables set DATABASE_URL=postgresql://...

railway variables list
```

---

## Run lifecycle

### Start a run

```bash
export AUTOPI_API_URL=https://<your-app>.up.railway.app

curl -s -X POST "$AUTOPI_API_URL/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "built_environment_health",
    "domain_input": "Built environment and health outcomes",
    "max_candidates": 40,
    "enable_experimental": false
  }'
# Returns: {"run_id": "...", "status": "starting"}
```

### Poll for status

```bash
./scripts/status.sh <run_id>
# awaiting_approval → ready for HITL
```

### Handle HITL checkpoint

The run pauses at `awaiting_approval` after the candidate factory completes.

```bash
# Review candidates
curl -s "$AUTOPI_API_URL/runs/<run_id>/candidates" | python3 -m json.tool

# Approve top candidate (continues to literature → drafter → data_fetcher)
./scripts/approve.sh <run_id>

# Or reject (aborts run)
curl -s -X POST "$AUTOPI_API_URL/runs/<run_id>/reject"
```

### View logs

```bash
./scripts/logs.sh <run_id>
./scripts/logs.sh <run_id> ERROR   # filter by level
railway logs --tail 100            # live Railway logs
```

---

## Run-scoped artifact isolation

Each run gets its own directory under `AUTOPI_DATA_ROOT/runs/<run_id>/`.  Use the `settings` token manager in all agents:

```python
from agents import settings

token = settings.activate_run_scope(run_id)
try:
    pack_dir = settings.development_packs_dir() / candidate_id
    # write artifacts...
finally:
    settings.deactivate_run_scope(token)
```

Never hardcode `"output/"` — always use `settings` helpers so cloud paths work correctly.

---

## Health check

```bash
curl -s "$AUTOPI_API_URL/health"
# {"status": "ok", "version": "..."}
```

Railway uses this endpoint as its liveness probe.

---

## Cloud smoke test

Run after every deploy to verify the full candidate factory → development pack path:

```bash
export AUTOPI_API_URL=https://<your-app>.up.railway.app
python scripts/cloud_smoke_test.py
```

The smoke test:
1. `GET /health`
2. `GET /templates`
3. `POST /runs` with `built_environment_health`
4. Polls until `awaiting_approval` or `completed`
5. `GET /candidates` — verifies ≥ 20 candidates
6. Selects top `ready` candidate
7. `POST /{candidate_id}/development-pack`
8. `GET claude_task_prompt.md` — verifies non-empty download

The smoke test does **not** test literature, drafter, or data fetcher (those require LLM keys and take longer).

---

## Docker (local testing)

```bash
docker build -t autopi .
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/storage:/app/storage \
  -e AUTOPI_DATA_ROOT=/app/storage \
  autopi

# Then run smoke test against local server
python scripts/cloud_smoke_test.py --url http://localhost:8000
```

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | — | Required for ideation + drafter |
| `GEMINI_FAST_MODEL` | `gemini-2.0-flash-lite` | Fast tasks |
| `GEMINI_PRO_MODEL` | `gemini-2.5-pro` | Complex tasks |
| `OPENALEX_API_KEY` | — | Literature search |
| `OPENALEX_EMAIL` | — | Polite pool |
| `AUTOPI_DATA_ROOT` | `.` | Root for all outputs |
| `DATABASE_URL` | — | PostgreSQL for cloud checkpointing |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LITERATURE_FINAL_LIMIT` | `3` | Papers kept after dedup |
| `GOOGLE_STREET_VIEW_API_KEY` | — | Experimental street view |
| `MAPILLARY_TOKEN` | — | Experimental Mapillary imagery |

See `.env.example` for the full list.
