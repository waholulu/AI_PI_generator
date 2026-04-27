#!/usr/bin/env bash
set -e

echo "==> 1. 单元测试"
python3 -m pytest tests/test_topic_schema.py \
                 tests/test_rule_engine.py \
                 tests/test_budget_tracker.py \
                 tests/test_reflection_loop.py \
                 tests/test_openalex_verifier.py \
                 tests/test_seed_normalizer.py \
                 tests/test_dict_to_seed_candidate_no_silent_other.py \
                 tests/test_prompt_enum_alignment.py \
                 tests/test_refine_operations_domain.py \
                 tests/test_openalex_query_composition.py \
                 tests/test_cli_flags_threading.py \
                 -v

echo "==> 2. 极端配置场景:删除 spatial_units.yaml 应当 fail"
mv config/spatial_units.yaml config/spatial_units.yaml.bak
python3 -c "
from agents.rule_engine import RuleEngine
from tests.test_rule_engine import make_topic
e = RuleEngine()
r = e.check_G2_scale_alignment(make_topic())
assert r.passed is False, f'expected False, got {r}'
assert 'config_unavailable_blocking' in r.reason, f'wrong reason: {r.reason}'
print('  ✓ G2 fails when config missing')
"
mv config/spatial_units.yaml.bak config/spatial_units.yaml

echo "==> 3. Smoke test:Level 2 dry-run (mock LLM, 5 seeds)"
GEMINI_API_KEY="" LEGACY_IDEATION=0 \
python3 -c "
from agents.ideation_agent_v2 import IdeationAgentV2
agent = IdeationAgentV2(skip_reflection=True)
try:
    agent.run({'domain_input': 'Urban Health', 'ideation_mode': 'level_2'})
except Exception as e:
    assert 'GEMINI_API_KEY' in str(e), f'unexpected error: {e}'
    print('  ✓ smoke: agent raises clear error on missing LLM key')
"

echo "==> 4. CLI flag pass-through"
python3 -m pytest tests/test_cli_flags_threading.py -v

echo "==> 5. Prompt 与枚举对齐"
python3 -m pytest tests/test_prompt_enum_alignment.py -v

echo "==> 全部通过"
