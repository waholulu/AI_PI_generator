# Auto-PI (AI PI Generator)

Auto-PI is a multi-agent research automation pipeline for early-stage academic research.  
It orchestrates field scanning, ideation, validation, literature harvesting, drafting, and data collection with a HITL checkpoint.

## Quick start

```bash
pip3 install -e .
cp .env.example .env
# Fill at least GEMINI_API_KEY
python3 main.py --help
```

## CLI modes

### Level 2 (default): domain -> generated topics

```bash
python3 main.py --mode level_2 --domain "GeoAI and Urban Planning"
```

### Level 1: user-provided structured topic YAML

```bash
python3 main.py --mode level_1 --user-topic inputs/my_topic.yaml
```

## CLI flags (Module 1 v0.2 semantics)

- `--mode {level_1,level_2}`: select ideation entry mode
- `--domain TEXT`: domain description (required for level_2)
- `--user-topic PATH`: structured topic yaml (level_1)
- `--legacy-ideation`: force legacy ideation v0
- `--budget-override-usd FLOAT`: override per-run budget passed into `IdeationAgentV2`
- `--skip-reflection`: pass-through to `IdeationAgentV2` to disable iterative reflection

## Environment variables (key subset)

- `GEMINI_API_KEY`: Gemini API key
- `GEMINI_FAST_MODEL`: default fast model
- `GEMINI_PRO_MODEL`: default pro model
- `OPENALEX_EMAIL`: polite-pool email for OpenAlex
- `OPENALEX_API_KEY`: OpenAlex API key (optional)
- `AUTOPI_DATA_ROOT`: output root directory
- `DATABASE_URL`: postgres checkpoint backend (optional)
- `LOG_LEVEL`: logging level

## Tests

```bash
python3 -m pytest tests/test_topic_schema.py \
                tests/test_rule_engine.py \
                tests/test_budget_tracker.py \
                tests/test_reflection_loop.py \
                tests/test_openalex_verifier.py -q

bash scripts/qa_module1_v0_2.sh
```
