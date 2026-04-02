# Auto-PI 现有 Idea 生成流程总结与下一步提升计划（含数据可得性终审门）

## 1. 现有 Idea 生成流程（基于当前代码实现）

> 位置：`agents/orchestrator.py` + `agents/ideation_agent.py`  
> 编排：`field_scanner -> ideation -> [HITL中断] -> literature`

### 1.1 上下游关系与核心产物
- ideation 输出：
  - `output/topic_screening.json`
  - `config/research_plan.json`
  - `output/research_context.json`
  - `output/topic_ranking.csv`
  - `output/ideas_graveyard.json`
  - `memory/idea_memory.csv`
  - `memory/enriched_top_candidates.jsonl`

### 1.2 现有步骤（代码实装）
1. **Step 0**：加载 field scan 摘要作为约束。  
2. **Step 1**：生成轻量候选（默认目标 30，去重，多轮尝试）。  
3. **Step 2**：单次结构化评分+过闸+排名，选 topN（默认3）。  
4. **Step 3**：对 topN 做 enrichment（impact、方法、数据源等）。  
5. **Step 3.5**：novelty gate（OpenAlex + LLM overlap），剔除已发表近似题并补位。  
6. **Step 4**：基于 top1/top3 生成 `research_plan.json`。  
7. **Step 5**：记忆写回与长期归档。  

## 2. 当前短板（重点）
1. 评分依赖单次大调用，鲁棒性有限。  
2. 记忆检索偏字符串匹配，语义复用弱。  
3. novelty 有门控但证据可解释性仍可提升。  
4. **缺少“数据公开且易获取”的硬验证门**（目前只在文本层声明，未形成可执行终审条件）。  

## 3. 新增关键步骤：Data Accessibility Gate（建议插入 Step 3.8）

> 插入位置：**Step 3.5（novelty）之后，Step 4（plan生成）之前**  
> 目的：确保“终审 Top3”每个 idea 的核心数据都可公开获取且抓取难度可控。  

### 3.8.1 验证对象
对终审候选（目标3个）逐一验证其 `data_sources`。  

### 3.8.2 验证维度（硬门槛 + 评分）
对每个数据源计算 `accessibility_score (0-100)`，并生成证据：

- **公开性（硬门槛）**
  - 非付费墙、非受限账号、非人工审批后才能拿到
  - 有明确开放链接/API/下载入口
- **可获取性（硬门槛）**
  - URL 可访问（HEAD/GET 200~399）
  - 支持机器可读格式（CSV/JSON/GeoJSON/Parquet 等）
- **易用性（评分项）**
  - 是否有文档/字段说明
  - 是否支持程序化访问（API/直链下载）
  - 是否有稳定更新频率与样例数据
  - 是否有明确 license（优先 open license）

### 3.8.3 Idea 级通过规则（建议）
每个终审 idea 必须同时满足：
1. 至少 `2` 个通过硬门槛的数据源；  
2. `mean(accessibility_score) >= 70`；  
3. 对其 `quantitative_specs.outcomes + exposures` 的覆盖率 >= `80%`（可构建映射）；  
4. 任一核心变量不可出现“仅封闭数据可得”的情况。  

### 3.8.4 不通过处理
- 若某 idea 失败：标记 `failed_data_accessibility_gate=true`，写入 reject reason；
- 从 `passed_candidates_pool` 中补位候选，执行：
  - enrichment -> novelty -> data accessibility gate
- 直至补满 3 个或候选池耗尽。  

### 3.8.5 输出字段（写入 `topic_screening.json`）
为每个终审 idea 增加：
- `data_accessibility_gate_passed`
- `data_accessibility_score`
- `data_accessibility_checks`（每源证据：url/status/license/format/docs/api）
- `variable_coverage`（outcomes/exposures 覆盖明细）

## 4. 下一步提升计划（更新版优先级）

### P0（优先）
1. **P0-1：新增 Step 3.8 Data Accessibility Gate**（本次新增重点）
2. P0-2：拒绝原因标准化（`reason_code + reason_text`）
3. P0-3：TopN 多样性硬约束
4. P0-4：评分双通道复核（边界样本仲裁）

### P1
1. 记忆检索升级为“关键词+语义”混合
2. novelty 证据卡输出（可追溯）
3. 候选分层生成（方法/场景/数据导向并行）

### P2
1. Ideation Benchmark（可回归评估）
2. 跨阶段闭环学习（下游失败信号反哺 ideation）

## 5. 落地顺序建议
先做 `P0-1`（Data Accessibility Gate）+ `P0-2`，再做多样性和评分复核。  
这样最快把“终审Top3可执行性”从软约束升级为硬门槛。  
