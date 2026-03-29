# Test Suite Guide

## Overview

All tests live under `tests/`.  They are divided into two distinct categories:

| Category | Marker | Requires external services? |
|---|---|---|
| Offline / unit / integration | *(none)* | No – uses mocks and fakes |
| Live – OpenAlex API | `live_openalex` | Yes – real OpenAlex calls |
| Live – LLM API | `live_llm` | Yes – real Gemini calls |

---

## Running tests

### Offline tests only (default, fast, no API keys needed)

```bash
# Exclude all live tests
pytest -m "not live_openalex and not live_llm"

# Or simply run all tests – live ones will FAIL if keys are absent
pytest
```

### Live OpenAlex tests

```bash
pytest -m live_openalex
```

Requires in `.env`:
- `OPENALEX_API_KEY`
- `OPENALEX_EMAIL`
- `pyalex` installed (already a project dependency)

### Live LLM tests (Gemini)

```bash
pytest -m live_llm
```

Requires in `.env`:
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `GEMINI_FAST_MODEL` (e.g. `gemini-1.5-flash`)
- `GEMINI_PRO_MODEL` (e.g. `gemini-2.5-pro`)

### All live tests at once

```bash
pytest -m "live_openalex or live_llm"
```

---

## File-by-file reference

### Offline unit / integration tests

| File | Module under test | Notes |
|---|---|---|
| `test_keyword_planner.py` | `agents/keyword_planner.py` | Tests LLM path, fallback, dedup, and max-query cap with mocked LLM |
| `test_field_scanner.py` | `agents/field_scanner_agent.py` | Uses `_OfflineFieldScanner` subclass; also tests multi-query, dedup, and `summarize_field_scan` |
| `test_memory_and_field_scan.py` | `field_scanner_agent`, `memory_retriever` | Tests with fake OpenAlex and synthetic CSV |
| `test_memory_retriever.py` | `agents/memory_retriever.py` | Tests `store_idea`, `retrieve_domain_context`, edge cases |
| `test_openalex_utils.py` | `agents/openalex_utils.py` | `configure_openalex`, `reconstruct_abstract`, `extract_work_metadata`, `download_pdf` (all HTTP mocked) |
| `test_openalex_enrichment.py` | `agents/openalex_utils.py`, `literature_agent.py` | Enrichment flow with patched `download_pdf` |
| `test_literature.py` | `agents/literature_agent.py` | `literature_node` with mocked OpenAlex search and download |
| `test_ideation.py` | `agents/ideation_agent.py` | `ideation_node` with fully mocked Gemini LLM |
| `test_drafter.py` | `agents/drafter_agent.py` | Happy-path with fake LLM + fallback path with forced LLM failure |
| `test_data_fetcher.py` | `agents/data_fetcher_agent.py` | Mock fetch path with manifest structure assertions |
| `test_orchestrator.py` | `agents/orchestrator.py` | Graph wiring, node presence, interrupt config, `ResearchState` schema |
| `test_ui_app.py` | `ui/app.py` | Import-level smoke test with stubbed `streamlit` and mocked orchestrator |
| `test_main_cli.py` | `main.py` | CLI behaviour: empty input, valid input, graph exception handling |
| `test_modules_1_and_2.py` | `ideation_agent`, `literature_agent` | Sequential Module 1+2 integration test (uses real APIs when available) |
| `e2e_simple_test.py` | Full pipeline | End-to-end orchestrator stream test |

### Live tests (require environment setup)

| File | Marker | What it exercises |
|---|---|---|
| `test_field_scan_live.py` | `live_openalex` | Real `FieldScannerAgent._search_openalex`; asserts non-empty results |
| `test_literature_live.py` | `live_openalex` | Real `LiteratureHarvester.search_openalex`; asserts non-empty inventory |
| `test_ideation_live.py` | `live_llm` | Real Gemini call inside `ideation_node`; asserts plan and candidate structure |
| `test_drafter_live.py` | `live_llm` | Real Gemini call inside `drafter_node`; asserts non-fallback draft content |
| `test_api_keys.py` | *(none)* | Quick Gemini API key sanity check (can run standalone) |

---

## CI recommendations

- **Every push**: run `pytest -m "not live_openalex and not live_llm"` – fast, no secrets needed.
- **Nightly / manual**: run `pytest -m "live_openalex or live_llm"` – needs secrets injected.
- Store API keys in CI secrets; never commit them to the repository.

---

## Environment variables reference

```
# .env (project root)
GEMINI_API_KEY=...                       # or GOOGLE_API_KEY
GEMINI_FAST_MODEL=gemini-2.0-flash-lite
GEMINI_PRO_MODEL=gemini-2.5-pro

OPENALEX_API_KEY=...                     # polite-pool key
OPENALEX_EMAIL=you@example.com

# OpenAlex keyword rewriting (KeywordPlanner)
OPENALEX_QUERY_REWRITE_ENABLED=true      # "false" to skip LLM and use raw domain_input
OPENALEX_QUERY_REWRITE_MAX_QUERIES=10    # max queries per scan
OPENALEX_QUERY_REWRITE_PER_QUERY_LIMIT=20 # results fetched per query
# OPENALEX_QUERY_REWRITE_MODEL=...       # defaults to GEMINI_FAST_MODEL

# Literature harvester
LITERATURE_FINAL_LIMIT=3                 # papers kept after dedup
```
