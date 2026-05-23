# Test Suite Guide

## Running tests

```bash
# Default: offline, no API keys needed (~90s)
uv run pytest -q

# Include slow / live tests by name or marker:
uv run pytest -m live_openalex      # real OpenAlex calls
uv run pytest -m live_llm           # real Gemini calls
```

`pyproject.toml` sets default markers to `not slow and not live_openalex and not live_llm`,
so plain `pytest` skips everything that needs secrets or runs the full pipeline.

## Categories

| Marker | What it needs | Run when |
|---|---|---|
| *(none)* | Mocks only | Every push |
| `live_openalex` | `OPENALEX_API_KEY`, `OPENALEX_EMAIL` | Manual |
| `live_llm` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Manual |
| `slow` | Real APIs, full pipeline | Manual |

## Live tests

| File | Marker | Exercises |
|---|---|---|
| `test_field_scan_live.py` | `live_openalex` | Real `FieldScannerAgent._search_openalex` |
| `test_literature_live.py` | `live_openalex` | Real `LiteratureHarvester.search_openalex` |
| `test_ideation_live.py` | `live_llm` | Real Gemini call inside `ideation_node` |
| `test_drafter_live.py` | `live_llm` | Real Gemini call inside `drafter_node` |

## CI recommendations

- Every push: `uv run pytest -q` (fast, no secrets needed).
- Nightly / manual: `pytest -m "live_openalex or live_llm"` (needs secrets).

## Environment variables

```
GEMINI_API_KEY=...                       # or GOOGLE_API_KEY
GEMINI_FAST_MODEL=gemini-2.0-flash-lite
GEMINI_PRO_MODEL=gemini-2.5-pro

OPENALEX_API_KEY=...                     # polite-pool key
OPENALEX_EMAIL=you@example.com
```
