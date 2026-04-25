# Module 1 Upgrade v0.2 Changelog

本文件记录 v0.2 缺陷修复（P0~P3）对应改动与测试覆盖。

> Commit hash 在完成提交后补充。

## P0

### P0-1 Seed -> Topic 归一化修复
- `agents/seed_normalizer.py`（新增）
- `agents/ideation_agent_v2.py` `_dict_to_seed_candidate`
- `models/topic_schema.py`（mitigations/target_venues 配套）
- Tests:
  - `tests/test_seed_normalizer.py`
  - `tests/test_dict_to_seed_candidate_no_silent_other.py`

### P0-2 Prompt 与枚举对齐
- `scripts/generate_prompt_enums.py`（新增）
- `prompts/_enums.txt`（生成产物）
- `prompts/ideation_seed.txt`
- `prompts/reflection_refine.txt`
- `prompts/reflection_critique.txt`
- `agents/ideation_agent_v2.py`（enum_block 注入）
- Tests:
  - `tests/test_prompt_enum_alignment.py`

### P0-3 G2/G3 不再放水
- `agents/rule_engine.py`
- Tests:
  - `tests/test_rule_engine.py`
  - `tests/test_rule_engine_config_missing.py`

### P0-4 CLI 死参修复
- `agents/orchestrator.py`
- `main.py`
- `agents/ideation_agent.py`
- `agents/_legacy/ideation_agent_v0.py`
- Tests:
  - `tests/test_cli_flags_threading.py`

## P1

### P1-2 领域化 refine 操作 + free_form + 反震荡
- `config/refine_operations.yaml`
- `agents/reflection_loop.py`
- Tests:
  - `tests/test_refine_operations_domain.py`
  - `tests/test_reflection_loop.py`（扩展）

### P1-3 OpenAlex verifier 三修
- `agents/openalex_verifier.py`
- Tests:
  - `tests/test_openalex_verifier.py`
  - `tests/test_openalex_query_composition.py`

### P1-4 / P1-5 / P1-7 critique与fallback修正
- `prompts/reflection_critique.txt`
- `agents/reflection_loop.py`

### P1-6 legacy_six_gates 键名
- `agents/ideation_agent_v2.py`
- `tests/test_ideation_v2.py`

## P2

### P2-1 / P2-2 schema 与字段补齐
- `models/topic_schema.py`
- `agents/ideation_agent_v2.py`
- `agents/rule_engine.py`

### P2-3 trace 补字段
- `agents/reflection_loop.py`
- `tests/test_ideation_v2.py`
- `tests/test_integration_module1_v2.py`

### P2-4 run summary 字段补齐
- `agents/ideation_agent_v2.py`

### P2-5 诊断报告统计增强
- `scripts/generate_diagnostic_report.py`

## P3

### P3-1 反思配置补全
- `config/reflection_config.yaml`
- `agents/ideation_agent_v2.py`（模型读取）

### P3-2 BudgetTracker 强化
- `agents/budget_tracker.py`
- `tests/test_budget_tracker.py`（原有用例持续通过）

### P3-3 HITLInterruption 出口处理
- `agents/ideation_agent.py`
- `main.py`

### P3-4 文档与QA
- `README.md`
- `docs/MODULE1_DESIGN.md`（新增）
- `docs/MODULE1_UPGRADE_v0.2.md`（本文件）
- `scripts/qa_module1_v0_2.sh`（新增）
