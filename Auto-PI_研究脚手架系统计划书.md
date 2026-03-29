# Auto-PI：多智能体驱动的前期研究脚手架自动化系统

> **版本**: v2.0
> **最后更新**: 2026-03-03
> **定位**: 自动化完成选题、文献调研、初稿撰写、数据采集四大前期环节，在**公开数据下载完成**后自动停机；**不做分析**（不做清洗/建模/统计推断/结果解释/可视化），将后续分析的主导权完全交还研究者。

---

## 目录

1. [项目概述与设计理念](#一项目概述与设计理念)
2. [系统核心架构](#二系统核心架构-auto-pi-system)
3. [四大核心模块详解](#三四大核心模块详解)
4. [技术栈与工具分配矩阵](#四技术栈与工具分配矩阵)
5. [实施计划（执行步骤）](#五实施计划执行步骤)
6. [成本估算与资源需求](#六成本估算与资源需求)
7. [风险分析与缓解策略](#七风险分析与缓解策略)
8. [学术伦理与合规声明](#八学术伦理与合规声明)
9. [项目目录结构](#九项目目录结构)

---

## 一、项目概述与设计理念

### 1.1 核心目标

构建一个多智能体协作系统，将学术研究中最耗时的前期准备工作（选题细化、文献检索、框架起草、公开数据获取）自动化为一条可复现的流水线。系统在完成原始数据下载后**即刻停机**，不涉及任何后续的数据清洗、统计分析或模型训练。

### 1.4 选题硬门槛（所有候选题必须同时满足）

> 本系统只保留同时满足以下门槛的选题；任何一项不满足则**直接淘汰**，并把淘汰原因与证据保存下来，便于人工复核与迭代。

- **足够的影响力**：存在明确的现实/政策/产业重要性或广泛关注度；可用公开可核验的“影响力证据”支撑（例如政策文件引用、官方指标体系、行业标准、主流机构报告、广泛使用的数据产品/平台等）。
- **必须是定量研究（定量可操作性）**：题目必须能被表达为“可检验假设 + 可观测变量 + 可复现的统计/因果识别路径”，并满足：
  - **清晰的被解释变量（Outcome）**：可由公开数据直接构造（给出字段/计算式或数据字典链接）；
  - **清晰的核心解释变量/处理（Exposure/Treatment）**：可测量，且方向与机制可陈述；
  - **单位与时间尺度明确**：样本单位（人/户/企业/行政区/网格等）、空间尺度与时间窗口清晰；
  - **识别/估计策略明确**：描述 estimand（例如 ATE/ITT/弹性/边际效应）、以及对应的模型族与识别假设（不要求系统执行估计，但要求计划可脚本化复现）。
- **没什么人研究过（新颖性）**：通过公开检索给出“研究空白证据”，但**新颖性不等于冷门**。优先定义为“在高牵引方向中的可验证空白”，常见可发表空白类型包括：
  - **问题空白**：某个机制/解释变量/异质性尚未被系统检验；
  - **测量空白**：关键变量的测量/代理变量仍粗糙（可用公开数据改进）；
  - **尺度空白**：特定空间/时间尺度（更细分辨率、更长面板、跨地区可比）缺失；
  - **方法空白**：已成熟的可复现方法在该问题上尚未被规范化应用；
  - **可复现空白**：缺少公开脚本/数据字典/可复现管线的研究（可形成“可复现性贡献”）。
  系统需记录检索式、命中计数、以及“空白类型”归类与证据摘要。
- **比较容易发表（可发表性）**：题目范围可控、变量定义清晰、可形成标准 IMRaD 结构；能匹配至少 2–3 个目标期刊/会议的投稿方向，并给出理由。
- **研究方法与分析可由 AI (如 Gemini) 自动解决（可自动化性）**：方法路径可被明确描述为可复现的脚本化流程（例如回归/因果推断/空间计量/面板模型/分层模型等），并能列出成熟的开源工具链（Python/R 包）作为“可落地证据”。
- **数据公开且易获取（数据可得性）**：数据源无付费墙、无需人工审批、可通过 API 或批量下载稳定获取；优先选择“**Census 类**”官方统计/普查/行政公开数据（结构化、版本稳定、字段清晰），并记录许可证/使用条款摘要。

### 1.5 “更容易被引用”的操作化（用于加速选题，而非替代学术判断）

> 目标是最大化“可发表 + 可被看到”的概率。引用指标只作为**优先级线索**，必须与第 1.4 节硬门槛同时成立。

- **牵引度（Traction）**：优先关注近 3–5 年的“引用增长速度（citation velocity）/被引分布/主题簇热度”，而不是只看累计被引。
- **归一化**：同一年份、同一子领域间比较更可信；避免把跨领域的引用规模差异误判为“更容易被引”。
- **反作弊**：避免追逐综述/工具论文的高被引幻象；系统需区分研究型论文 vs 综述/方法/数据集论文，并在报告中标注。
- **可落地证据优先**：牵引度高但数据不可得/方法不可复现/贡献难讲清的题目仍应淘汰。

### 1.6 结果保存原则（清晰、有组织、可追溯）

- **全量留痕**：候选题清单、筛选打分与证据、最终选题、文献索引、数据清单均落盘。
- **结构化长效记忆库（Persistent Memory System）**：所有成功的选题设计、废弃创意回收（Idea Vault，带战败原因如“数据不可得”或“方法不可复现”）、以及提取到的高优学术趋势、主观评价建议，统一写入本地系统记忆库（基于 SQLite 或轻量级矢量数据库）。新研究任务将优先跨周期检索这部分记忆，研究者也可在未来环境变化（如某封闭数据库开源）时随时唤醒这些高价值灵感，最大限度减少 API 调用浪费并避免重复踩坑。
- **证据可复核**：所有“影响力/新颖性/可发表性/可自动化性/数据可得性”的判断必须附带可核验证据（链接、检索式、摘要、计数、数据字典/字段示例等）。
- **统一索引**：每次运行生成 `output/run_index.json`，集中列出本次产生的关键文件路径与摘要，便于后续人工接管与归档。

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **边界清晰** | 系统严格限定在"前期脚手架"范围内，下载完成即停机 |
| **数据合规** | 仅对接公开、免费、无授权壁垒的数据源（如 US Census Bureau、OpenStreetMap、NASA EarthData 公开数据集等） |
| **可审计** | 每一步输出均保存为文件，研究者可逐环节审查与修正 |
| **模块解耦** | 四大模块独立运行，单模块失败不阻断其余模块的执行 |
| **辅助定位** | 系统产出仅为初稿草案与参考材料，最终学术成果须由研究者亲自完善 |

### 1.3 系统工作流总览

```
用户输入（宏观领域描述）
        │
        ▼
┌──────────────────────────┐
│  Module 1: 选题与研究设计  │  ← Gemini Gemini API
│  输出: research_plan.json │
│       + topic_screening.* │
│       + field_scan.*      │
│       + topic_ranking.csv │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Module 2: 文献挖掘与仓储   │  ← OpenAlex + arXiv + Crossref + Unpaywall
│ 输出: data/literature/*   │
│      output/references.bib│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Module 3: 学术起草（可选） │  ← Gemini 3.1 Pro
│ 输出: output/Draft_v1.md  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Module 4: 数据采集（最后） │  ← 适配器下载 + 完整性校验
│ 输出: data/raw/*          │
│      data/raw/manifest.json│
└──────────┬───────────────┘
           │
           ▼
output/run_index.json（统一索引入口）
           │
           ▼
[SYSTEM HALT] 数据采集完成即停机（不做分析），等待人工介入
```

---

## 二、系统核心架构 (Auto-PI System)

### 2.1 编排层选型

> **重要说明**: 原方案中提及的 "Google Antigravity" 并非实际存在的产品（Python 中 `import antigravity` 仅为一个彩蛋）。以下为经过评估的可行替代方案：

| 方案 | 工具 | 适用场景 | 优势 | 劣势 |
|------|------|----------|------|------|
| **A（推荐）** | **LangGraph** (LangChain 生态) | 本地开发，需要复杂的有状态工作流 | 原生支持有向图工作流、检查点与回退机制、与 LangChain 工具生态无缝集成 | 学习曲线中等 |
| B | CrewAI | 快速原型，多 Agent 角色扮演 | 上手极快，Agent 角色定义直观 | 深度定制能力有限 |
| C | Google Colab + 自编排脚本 | 轻量级原型验证 | 零配置、免费 GPU、便于协作 | 不适合长期生产环境 |

**推荐方案 A：使用 LangGraph 作为主编排框架**，在本地 Python 环境中开发。LangGraph 的有向无环图（DAG）工作流天然适配本系统"四模块串行（模块内可并行）"的执行拓扑，且支持节点级检查点（Checkpoint），便于在任意环节失败后从断点恢复。

### 2.2 模块间通信机制

各模块通过**文件系统**进行松耦合通信，核心中介文件为 `config/research_plan.json`：

```
Module 1  ──写入──▶  config/research_plan.json  ◀──读取──  Module 2
                                                  ◀──读取──  Module 3
                                                  ◀──读取──  Module 4
```

所有模块的中间产物均落盘为文件，确保可审计性与可复现性。

此外，编排层在每次运行结束时生成 `output/run_index.json` 作为“统一入口”，列出：
- 本次最终选题与筛选摘要（指向 `output/topic_screening.*`）
- 文献索引与引用库位置（指向 `data/literature/index.json` 与 `output/references.bib`）
- 原始数据清单（指向 `data/raw/manifest.json`）
- 关键元信息（时间戳、数据源端点、检索式摘要、版本号）

### 2.3 实时监控与交互可视化 UI (Agent Monitoring UI)

为了直观、可靠地追踪多智能体的运行状态并避免运行黑盒效应，系统集成一个基于 **LangGraph Studio**（或轻量级 Streamlit/Gradio 仪表盘）的监控 UI：
- **实时状态追踪 (Real-time Tracker)**：以有向无环图 (DAG) 的形式实时高亮显示当前正在执行的 Agent 节点及子步骤（规划、打分、下载、合成）。
- **流式追踪与内部思绪 (Streaming Logs)**：在控制台面板实时打印 Agent 的内部思考过程 (Thought Process)、工具调用详情及 API 响应耗时。
- **人工干预与审批枢纽 (Human-in-the-Loop)**：当遇到重大决策节点（如候选题目选择、数据源异常过大触发熔断）时，系统在 UI 上自动挂起暂停，弹出对比卡片供研究者人工审阅和选择，点击确认后流程继续。
- **资源监控面板 (Cost & Resource Dashboard)**：实时统计各阶段消耗的 Token 数量、预计产生的 API 费用以及数据抓取磁盘下载进度。

---

## 三、四大核心模块详解

### 3.1 Module 1: 选题与研究设计引擎 (Ideation & Design Agent)

| 项目 | 内容 |
|------|------|
| **核心工具** | Google Gemini 3.1 Pro / Flash API（利用基础搜索与结构化输出能力） |
| **输入** | 用户提供的宏观领域描述（自然语言），例如：*"利用 GeoAI 优化城市基础设施规划"* |
| **输出** | `config/research_plan.json`（最终选题研究设计）+ `output/topic_screening.json/.md`（候选题筛选与证据归档）+ `output/field_scan.json/.md`（领域牵引扫描，含引用趋势与主题簇）+ `output/topic_ranking.csv`（候选题打分排序，便于人工快速挑选） |

**工作流：**

0. **长期记忆检索（Long-term Memory RAG，防止重造轮子）**：接收新领域输入后，Agent 首要行为是查询本地记忆库（当前以单一 CSV `memory/idea_memory.csv` 为主，滚动记录过往 `research_plan.json` 及 `ideas_graveyard.json` 中的关键候选与淘汰原因）。
   - 寻找是否存在类似宏观领域的已知学术趋势线索。
   - 调取类似选题之前得到的评价和被淘汰原因（如某方向曾明确验证为“数据不公开”或“缺乏因果识别路径”），并按领域简单过滤后，作为本次选题的**反向过滤约束条件**（Negative Constraints）。
   - 若记忆为空（CSV 不存在或为空文件），则使用空上下文继续执行，避免因记忆缺失阻断主流程；本轮选题结束后，再将新的候选与结果追加写入记忆。

1. **领域牵引扫描（Bibliometric + Trend Scouting）**：在提取记忆补丁后，针对记忆库中不足的领域知识，自动向外构建本次领域的“高牵引主题簇”，作为后续生成与筛选的约束条件与优先级线索。
   - 数据来源建议：OpenAlex API（引用/年份/领域概念标签/开放获取状态）+ arXiv（新近预印本）+ 会议/期刊目录页（近 2–3 年主题变化）。
   - 当前 MVP 阶段，`output/field_scan.json` 至少包含：扫描的领域 `domain_scanned`、OpenAlex 返回的 `openalex_traction.top_results`，以及基于论文 `broad_concepts` 频数统计得到的高频概念列表 `keywords.high_traction`，用于近似刻画“领域牵引关键词”；同时附带简要元信息 `meta.scan_status`（full / empty），标记本轮扫描是否命中有效论文，即使扫描结果为空也会生成最小 `field_scan.json` 以保证后续模块仍可运行。

2. **受约束的候选题生成（规模化 + 多样性）**：协同历史记忆与当前热度扫描，调用中端推理模型（如 Gemini 1.5/2.0 Pro）搭配普通 Web Search 工具，在“高牵引主题簇”内批量生成 20–40 个“可发表规模”的候选题（强制多样性：不同数据源/不同空间尺度/不同方法路径），并要求每个题目同时给出：
   - (a) 影响力叙述（含可核验证据线索）
   - (b) 新颖性/研究空白叙述（必须标注“空白类型”，见 1.4）
   - (c) 可发表性：2–3 个目标 venue + 匹配理由（主题契合/方法常见度/数据类型）
   - (d) **定量研究规格（必须）**：以“可脚本化”的形式给出下列要素，否则该候选题视为无效并直接淘汰：
     - **Outcome / Exposure / Controls**：变量定义、数据表/字段映射、计算式（如需）；
     - **单位与面板结构**：cross-sectional / panel / repeated cross-sections，空间单位、时间频率；
     - **estimand 与识别策略**：例如 ATE/ITT/事件研究动态效应/弹性，说明识别假设；
     - **模型族**：例如 OLS/GLM/固定效应面板/DiD/事件研究/断点回归/合成控制/IV/空间计量（只允许成熟、可复现、脚本化的标准方法路径）；
     - **稳健性与异质性清单**：至少 6 条可脚本化检查（替代度量、安慰剂、窗口敏感性、并行趋势检验、聚类稳健标准误、空间权重敏感性等，按题目类型选择）；
     - **预期“结果表/图”清单**：仅列出将产出的表格/图类型与所需统计量（不要求系统计算结果）。
   - (e) 可自动化的方法路径（可脚本化步骤，面向后续 AI 自动化分析接管）
   - (f) 可获得的公开数据源列表（优先官方统计/普查/行政公开数据）
2. **硬门槛筛选（Gate）**：对每个候选题逐项判断是否满足第 1.4 节的硬门槛；任何一项不满足则淘汰，并保存淘汰原因与证据（链接/检索式/计数/摘要/数据字典片段）。
3. **排序打分（Rank）**：对通过硬门槛的候选题进行多目标打分并排序，输出 `output/topic_ranking.csv`（并在 `topic_screening.*` 中解释评分依据）。建议最小可行评分维度：
   - **牵引度**（来自 field_scan）：主题簇热度/引用增长（归一化后）
   - **贡献清晰度**：一句话贡献能否成立（可操作、可检验、可复现）
   - **可落地性**：数据端点可达 + 字段清晰 + 方法工具链成熟
   - **叙事完整度**：IMRaD 是否顺滑（引言-缺口-贡献-数据-方法）
4. **数据源匹配与可达性验证（Verify）**：对排序靠前的题目，自动匹配**绝对公开可用**的数据源，并验证其 API 端点可达性与下载可行性（必要时提供降级数据源）。候选数据源白名单：
   - US Census Bureau API (`api.census.gov`)
   - OpenStreetMap Overpass API
   - NASA EarthData 公开数据集
   - USGS 地质调查数据
   - World Bank Open Data API
5. **研究设计产出（Finalize，不执行分析）**：从 Top 1–3 个题目中产出最终 `research_plan.json`（允许并行产出多个 plan 供人工快速择优），根据研究问题性质推荐“可脚本化复现”的研究方法与分析计划（例如 SAR/GWR/固定效应面板/断点回归/合成控制等），但**只输出方案与可运行骨架，不运行任何统计分析**。

**输出文件示例 (`research_plan.json`)：**

```json
{
  "project_title": "基于官方统计与 OSM 的城市绿地可达性不平等：以 Census Tract 为单位的可发表研究设计",
  "study_type": "quantitative",
  "topic_screening": {
    "hard_gates": {
      "impact": true,
      "quantitative": true,
      "novelty": true,
      "publishability": true,
      "automation_feasibility": true,
      "data_accessibility": true
    },
    "evidence": {
      "impact": ["官方指标体系/政策文件/行业报告等的引用线索（附链接）"],
      "novelty": ["检索式与命中计数摘要（附链接/摘要）"],
      "publishability": ["2-3 个目标期刊/会议与匹配理由"],
      "automation_feasibility": ["可复现工具链：Python/R 包清单与可脚本化步骤摘要"],
      "data_accessibility": ["数据源许可证/条款摘要、字段示例、API/下载链接"]
    }
  },
  "research_questions": [
    "城市绿地空间分布与社区人口结构之间是否存在显著的空间自相关？",
    "哪些地理要素对绿地可达性的影响最为显著？"
  ],
  "hypotheses": [
    "低收入社区的绿地可达性显著低于高收入社区"
  ],
  "unit_of_analysis": "Census Tract",
  "outcomes": [
    {"name": "green_space_accessibility", "definition_notes": "由 OSM 公园要素 + 路网/距离度量构造（给出脚本化构造步骤）"}
  ],
  "exposures": [
    {"name": "median_household_income", "source_variable": "ACS.B19013_001E"}
  ],
  "keywords": ["GeoAI", "urban green space", "accessibility", "spatial analysis"],
  "data_sources": [
    {
      "name": "US Census Bureau ACS 5-Year",
      "api_endpoint": "https://api.census.gov/data/2022/acs/acs5",
      "variables": ["B01003_001E", "B19013_001E"],
      "license_notes": "公开数据；遵守 Census API 使用条款；记录变量字典链接",
      "format": "json"
    },
    {
      "name": "OpenStreetMap - Parks & Green Spaces",
      "api_endpoint": "https://overpass-api.de/api/interpreter",
      "query_template": "[out:json];area[name='City Name']->.a;(way['leisure'='park'](area.a););out body;",
      "license_notes": "遵守 ODbL；记录查询模板与时间戳",
      "format": "geojson"
    }
  ],
  "methodology": {
    "primary": "Geographically Weighted Regression (GWR)",
    "secondary": "Random Forest Feature Importance",
    "spatial_unit": "Census Tract"
  }
}
```

### 3.2 Module 2: 文献挖掘与仓储模块 (Literature Harvester)

| 项目 | 内容 |
|------|------|
| **核心工具** | OpenAlex API + arXiv API + Crossref API + Unpaywall API（LLM 仅用于检索式扩展与重排） |
| **输入** | `research_plan.json` 中的 `keywords` 字段 |
| **输出** | `data/literature/`（PDF/摘要文本）+ `data/literature/index.json`（获取状态）+ `output/references.bib` + `output/literature_index.json`（便于快速浏览的摘要索引） |

**工作流：**

1. **关键词提取与检索式扩展**：从 JSON 配置中读取关键词列表，自动扩展同义词（如 GeoAI → Geospatial AI, Spatial Machine Learning）；LLM 仅负责生成候选检索式，不直接决定最终文献集。
2. **元数据并行检索（可复现主链）**：
   - **OpenAlex API**：通过 `Works().search(query).sort(cited_by_count="desc")` 检索高相关/高引用文献，获取 DOI、引用信息、领域概念标签与开放获取线索；作为主链检索来源。
   - **arXiv API**：针对计算机科学与定量方法领域的预印本进行补充检索。
   - **Crossref API**：以 DOI 回填权威元数据并统一 BibTeX 字段。
3. **开放获取落地策略**（动态回退机制）：
   - 在请求 PDF 链接前，利用 `requests.head()` 探测是否遭遇 `403 Forbidden` 或 CAPTCHA。
   - 若遇到强反爬机制，立即短路跳过当前 PDF 获取，直接回退并仅保存 Abstract，避免耗时重试和 IP 封禁。
   ```
   优先级 1: Crossref/Publisher 的 OA 直链（可达且合规）
   优先级 2: OpenAlex OA URL（open_access.oa_url 字段）
   优先级 3: arXiv 预印本 PDF
   优先级 4: Unpaywall API
   优先级 5: 仅保存 Abstract（最终回退）
   ```
4. **高可用设计与缓存层**：
   - 引入 Tenacity 指数退避重试机制应对学术 API（OpenAlex, Crossref 等）严格的 Rate Limits（HTTP 429 错误）。
   - 引入本地缓存（SQLite/DiskCache），以 DOI 为主键缓存检索结果。即使工作流中断，重启时可避免二次网络开销。
5. **文献入库与证据卡片生成**：
   - PDF 文件保存至 `data/literature/`，文件名格式：`AuthorYear_ShortTitle.pdf`
   - 自动生成 BibTeX 引用文件 `output/references.bib`
   - 生成文献索引 `data/literature/index.json`，记录每篇文献的获取状态
   - 额外生成 `output/literature_index.json`：从 `data/literature/index.json` 抽取“标题/年份/venue/引用数/开放获取状态/关键词匹配”作为快速浏览索引
   - 生成 `data/literature/cards/*.json`：每篇文献的结构化证据卡片（研究问题、数据来源、方法、关键结论、局限、可复用变量、citation_key、证据片段）

**关键约束：**
- 严格遵守各 API 的速率限制（OpenAlex、arXiv、Crossref、Unpaywall）；OpenAlex 免费 API Key 提供每日 100,000 次配额，须在 `.env` 中配置 `OPENALEX_API_KEY`
- 单次运行文献上限设为 50 篇，防止过量下载

### 3.3 Module 3: 学术起草模块 (Academic Drafter)

| 项目 | 内容 |
|------|------|
| **核心工具** | Gemini Deep Research API（或 3.1 Pro API），基于结构化证据卡片跨文献合成长文 |
| **输入** | `research_plan.json` + `data/literature/cards/*.json` + 文献库索引 |
| **输出** | `output/Draft_v1.md` — Markdown 格式的论文前五部分初稿 |

**工作流：**

1. **两段式上下文组装**：
   - 阶段 A（结构化）：加载 `research_plan.json` 与 `data/literature/cards/*.json`，优先使用证据卡片作为主上下文。
   - 阶段 B（证据补强）：仅在必要处注入少量关键原文片段（带来源标识），避免整库全文拼接导致噪声与成本上升。
2. **结构化生成**：通过精心设计的 System Prompt 约束输出格式与学术风格：
   ```
   必须包含的章节：
   1. Abstract（摘要，200-300 词）
   2. Introduction（引言，含研究背景、研究缺口、本文贡献）
   3. Literature Review（文献综述，按主题而非时间线组织）
   4. Data Description（数据描述，含数据来源、变量定义、空间/时间范围）
   5. Methodology（研究方法，含模型选择的理论依据）
   ```
3. **质量约束（System Prompt 要素）**：
   - 使用客观、正式的学术语气，避免主观判断词
   - 所有引用必须使用 `references.bib` 中已有的文献，禁止编造引用
   - 生成后执行“引用一致性校验”（citation_key 必须存在、DOI 可回溯、关键事实可定位到证据片段）
   - 明确标注"此为 AI 辅助生成的初稿，需研究者审校"水印

**重要提示**：AI 生成的初稿仅作为写作起点，研究者必须对内容进行全面审校、补充原创见解、并确保引用的准确性。

### 3.4 Module 4: 数据采集与熔断模块 (Data Fetcher & Circuit Breaker)

| 项目 | 内容 |
|------|------|
| **核心工具** | **Anthropic API (Claude 3.5 Sonnet) + 受限代码执行沙箱** (如 e2b_code_interpreter 或是本地 Docker 工具) |
| **输入** | `research_plan.json` 中的 `data_sources` 字段 |
| **输出** | `data/raw/` 目录下的原始数据集（强制转换为 `.parquet`, `.geoparquet` 等降维压缩格式，并保留 EPSG 元数据） |

**工作流：**

0. **数据长期记忆与归档查重（Data Vault & Deduplication）**：在启动任何下载动作前，系统首先通过历史 `manifest.json` 哈希/元数据记录及本地的全局数据归档库（Data Vault，包含如 `vault_index.sqlite` 获取索引）进行查重检索。
   - 根据 `research_plan.json` 里的参数，如果在归档库中发现一致的数据（同一空间尺度、时间窗口、来源端点，如先前抓取过的庞大 Census 地图边界或遥感底图），系统会直接通过软链接（Symlink）或硬拷贝映射至当前任务的 `data/raw/`。
   - **核心目的**：将高价值、高体量的公开原始数据转化为**可复用的持久化资产**。大幅节省硬盘存储空间并避免无意义的网络 I/O，同时也极大减轻 Claude Agent 在此步骤的复杂度和运行成本。
1. **Agent接管策略**（状态机驱动的 Tool Calling / Code Interpreter）：
   - 弃用庞大且脆弱的本地 `pexpect` / `subprocess` 包装 Claude Code CLI 的逻辑（容易因人类终端 UI 的变动挂起或死锁）。
   - 在 LangGraph 节点中直接调用 Anthropic API，并配备一个受限的执行沙箱（如 `e2b_code_interpreter` 或是仅挂载 `data/raw/` 的本地 Docker）。
   - 让 Agent 循环执行“查阅要求 -> 编写下载提取代码 -> 运行沙箱 -> 读取报错 -> 修正代码”的逻辑循环，极其符合可预测的状态机设计模式。
2. **空间与普查数据格式降维审查（强制化标准）**：
   - 考虑到偏好普查以及地理信息数据，强制 Agent 在下载臃肿的 GeoJSON、大型 CSV 文件后，立刻通过沙箱将其转换为压缩率超过 70% 的 **Parquet 或 GeoParquet 格式**后落盘并清理零碎文件。
   - CRS 统一审查：要求强制提取当前数据的**空间参考系统 (CRS)**，并在 `manifest.json` 中明确写入其对应的 **EPSG 坐标系代码**，杜绝坐标系不匹配灾难。
3. **预探测与熔断机制（Dry-run 容量防爆）**：
   - 在 Prompt 中严格强制 Agent：**在执行真实数据下载前，必须编写带 `requests.head()` 的探针或利用 API 分页查询，预估下载总容量**。
   - 若发现单源数据包预测超过极限或触发大规模拉取，系统主动挂起请求确认。
4. **安全与日志记录**：
   - 依赖专门的沙箱以隔离本地文件系统风险，无需暴露主控机器环境。下载完毕后 Agent 需自述文件内容完整性检查（行列与哈希）。
5. **获取与数据处理指引**：
   - Agent 除了将文件清洗为规范格式产出，还应当基于对下载数据的初步侦察，编写配套的**“数据处理指引 (Data Processing Guideline)”**文本。
4. **熔断停机**：
   ```
   ════════════════════════════════════════════════════
   [SYSTEM HALT] 公开数据采集完成，系统停机（不做分析）

   ✅ 研究计划 ─── config/research_plan.json
   ✅ 选题筛选 ─── output/topic_screening.json / .md（含证据与淘汰原因）
   ✅ 文献资料 ─── data/literature/ (大概 32 篇, 大概 18 全文 + 14 摘要)
   ✅ 论文初稿 ─── output/Draft_v1.md（可选）
   ✅ 原始数据 ─── data/raw/ (3 文件, 共 24.7 MB)
   ✅ 运行索引 ─── output/run_index.json（所有结果的统一入口）

   系统已自动停机，等待研究者介入进行后续分析（本系统不做分析）。
   ════════════════════════════════════════════════════
   ```
   随后进程自动退出（`sys.exit(0)`），绝不越界进行任何数据清洗或分析操作。

---

## 四、技术栈与工具分配矩阵（智能与成本的分层调度）

### 4.1 编排层状态机设计 (LangGraph State)

在 LangGraph 中，各节点围绕统一的 `ResearchState` (TypedDict) 进行读写更新，核心定义如下：
```python
class ResearchState(TypedDict):
    # 【引用传递(Pass-by-Reference)防止内存状态膨胀及上下文溢出】
    domain_input: str               # 用户的宏观领域描述
    field_scan_path: str            # M1 领域牵引扫描内容路径
    candidate_topics_path: str      # M1 批量生成的候选题列表路径
    current_plan_path: str          # M1 最终选定的 research_plan.json 路径
    literature_inventory_path: str  # M2 文献索引 index.json 路径
    draft_content_path: str         # M3 初稿正文的本地路径
    raw_data_manifest_path: str     # M4 数据采集清单 manifest.json 路径
    execution_status: str           # e.g., "ideation" -> "harvesting" -> "drafting" -> "fetching"
```

### 4.2 工具与角色分配矩阵

| 功能层 | 工具 / 框架 | 角色 / 路由策略 | 预期输出 |
|--------|-------------|------|----------|
| **系统监控与UI交互** | Streamlit / LangGraph Studio | 实时监听 Agent 节点流转、耗费 Token 及日志，并在决策断点请求人类干预。 | 直观可视化仪表盘 |
| **长期记忆与知识网络**| SQLite / ChromaDB | 构筑历史沉淀记忆系统，存储过往方向尝试结果、剔除废案原因和有效引用图谱。 | `.db` 或向量存储 |
| **项目编排** | LangGraph (Python) | 定义 DAG 工作流、管理模块执行顺序与并行、提供断点恢复。 | 可运行的工作流引擎 |
| **选题广度扫描 (快筛)** | 廉价轻量级模型 (`gemini-3.1-flash-image-preview` 等) | 【广度扫描】以极低成本生成大量候选题草案，并基于规则进行硬门槛过滤。 | 落盘大量候选题、刷掉 80% 劣质想法 |
| **中度评估与方案制定** | 中端主干模型 (`gemini-3.1-pro-preview` 等) | 【方案定型】仅针对初筛存活的 Top-N 选题，设计严密的定量研究结构、变量组和推断验证方法。 | `research_plan.json` + `topic_ranking.csv` |
| **文献检索（元数据主链）** | OpenAlex API（主链）/ arXiv / Unpaywall API | 避免让大模型去广海捞针。基于纯数据流获取文献本体和高价值结构化引用图谱；OpenAlex 提供引用数、概念标签与开放获取状态。 | `data/literature/*.pdf` + `output/references.bib` |
| **内容合成与深度综述** | 深度推理模型 (Gemini Deep Research API) | 【深层解析】在**人工确认最终的唯一 1 个选题后**，进行耗费算力的大跨度横向对比阅读，攻克最难的综述撰写与理论争论。 | `output/Draft_v1.md` |
| **数据采集** | Anthropic API + 代码执行沙箱 (如 `e2b`) | 基于状态机构建循环驱动 Agent 编写脚本、抓取数据并纠错，最终转换 Parquet / 留存 EPSG 信息。 | `data/raw/*.parquet`, `*.geoparquet` |
| **PDF 预解析（降本）** | GROBID（优先）+ pymupdf | 免费拆除 PDF 废料（提取正文与方法，丢弃图表和致谢等），大幅节省 Token 上下文。 | 结构化片段提取 (`.json`) |
| **环境管理** | `uv` + `dotenv`（保留 venv 兼容） | API Key 隔离、锁定依赖版本、提升跨环境复现性。 | `.env` 文件、`pyproject.toml` |

---

## 五、实施计划（执行步骤）

### 第一阶段：基础设施搭建（第 1–2 天）

- [ ] 初始化项目目录结构（见第九节）
- [ ] 启动 GROBID 本地解析服务（可选但强烈推荐，也可降级至 `pymupdf4llm` 此处作为容错）：
  ```bash
  docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.0
  ```
- [ ] 配置 Python 环境（推荐 `uv`；兼容 venv），安装并锁定依赖：
  ```bash
  uv init
  uv add langgraph langgraph-checkpoint-sqlite langchain-google-genai langchain-anthropic \
         arxiv requests python-dotenv pyalex grobid-client e2b_code_interpreter \
         pandas geopandas pyarrow
  uv lock
  ```
- [ ] 配置好 Agent 代码执行沙箱（如配置 `E2B_API_KEY` 环境变量或拉取本地受限 Docker 镜像用于安全作业）。
- [ ] 配置 `.env` 文件，填入各 API Key：
  ```
  GEMINI_API_KEY=xxx
  OPENALEX_API_KEY=xxx             # 在 openalex.org/settings/api 免费申请
  OPENALEX_EMAIL=you@example.com   # 可选，Polite Pool 优化用
  UNPAYWALL_EMAIL=you@example.com  # Unpaywall 查询建议填写联系邮箱
  ```
- [ ] 搭建 LangGraph 工作流骨架，配置 `SqliteSaver` 检查点以便支持任意节点中断和断点重试恢复。
- [ ] **增加人机流转交互点（HITL）**：配置 `interrupt_before`，使得在 Module 1 产出候选题、生成 `topic_ranking.csv` 之后工作流挂起，等待研究者确认编号再启动后续检索。

### 第二阶段：核心模块开发（第 3–6 天）

**Module 1 — 选题引擎（第 3 天）**
- [ ] 编写 Gemini API 调用逻辑，使用 Structured Outputs 强制返回 JSON
- [ ] 实现数据源 API 可达性预检查（HTTP HEAD 请求验证端点存活）
- [ ] 编写单元测试

**Module 2 — 文献挖掘（第 4 天）**
- [ ] 实现 OpenAlex（主链）+ arXiv + Crossref 的并行检索链；配置 `OPENALEX_API_KEY` 环境变量
- [ ] 集成 Unpaywall 降级策略与 DOI 去重逻辑
- [ ] 实现 BibTeX 自动生成逻辑
- [ ] 添加速率限制（Rate Limiter）

**Module 3 — 学术起草（第 5 天，可选）**
- [ ] 编写 System Prompt（学术风格约束、章节结构要求、引用一致性校验）
- [ ] 实现“证据卡片优先 + 关键原文补强”的两段式上下文组装
- [ ] 调用 Gemini 3.1 Pro API 生成初稿

**Module 4 — 数据采集（第 6 天，必须最后执行）**
- [ ] 弃用 pexpect/Claude Code CLI，引入 Anthropic API Tool Calling 结合本地或 e2b 代码沙箱作为底层抓取引擎。
- [ ] 开发执行空间和普查数据格式降维（强制降维存 Parquet 与提取写入 EPSG 到 manifest）的功能链设计。
- [ ] 在核心 System Prompt 内强化对“容量预探（HEAD size/分页侦察）”的指令声明及容错中断机制逻辑设计。
- [ ] 落地整体熔断逻辑，待采集沙箱完成所有的降维清洗工作之后，系统自动清退一切后端环境并中止节点流程。

### 第三阶段：集成测试与工作流闭环（第 7–8 天）

- [ ] 将四个模块接入 LangGraph DAG，明确执行顺序：**Module 4（数据采集）必须为最后一步**；数据采集完成即停机
- [ ] 输入一个熟悉的测试领域，执行端到端测试
- [ ] 验证核心检查点：
  - `research_plan.json` 格式正确且数据源可达
  - `data/literature/` 中有 PDF/摘要文件，并生成 `data/literature/cards/` 证据卡片
  - `output/Draft_v1.md` 格式完整且引用有据可查（可选）
  - `data/raw/` 中有非空数据文件
  - 系统在数据下载完成后**立即停止**，无任何越界分析行为
- [ ] 修复集成问题，确保模块间数据传递无误

---

## 六、成本估算与资源需求

### 6.1 AI 请求成本分配与节省机制（Cost-aware Optimization）

在调用大量大模型接口时，“把好钢用在刀刃上”极为核心。本系统主要通过以下措施削减浪费：

1. **模型分片路由（Model Routing）**
   - **海量“浅层过滤”**：长段落抽取、格式转换、初筛阶段批量生成上百个“半成品选题”，交由 **Gemini Flash (`gemini-3.1-flash-image-preview`)** 级别模型处理，速度快且成本近乎忽略不计。
   - **深水区“压轴”**：仅在最后的 `Module 3` 阶段，为了读透数十篇核心长文、对比细微实证差别，才触发 **Gemini Deep Research API**。此时选题已定，投入再多 Token 也属于有效转化。
2. **上下文精准剪枝（Context Pruning）**
   - 不把下载的完整 PDF 盲目塞进模型。依靠开源解析工具（如 GROBID），强制剃除文献末尾几十页的参考文献列表（References）、作者致谢（Acknowledgments）和繁杂附录。提炼出的 `Abstract, Methodology, Conclusion` 有效降低 60%~70% 的 Token 上下文冗余。
3. **免费学术 API 拦截搜索需求**
   - 用大语言模型直接在网上无头苍蝇般去“寻找相关研究”是极度不划算的。让模型只充当“决策流控器”，实际对领域文献图谱的探索、引用树（Citation Tree）的遍历交由 OpenAlex API 这类免费元数据库完成。
4. **意念坟场复用（Idea Vault 落盘）**
   - 一次扫描产生 50 个 Idea，最终只用 1 个，剩下 49 个的计算成本不能付诸东流。强制将它们落盘作为数据资产保留在 JSON 中，下次需要做相近领域时，无需从 0 开始请求，直接去过往的废弃堆里翻找并重新审视。

### 6.2 近似 API 调用成本（单次完整运行估算）

| API | 预估用量 | 单价参考 | 估算费用 |
|-----|----------|----------|----------|
| Gemini Deep Research | 1–2 次深度调用 | 按 token 计费 | ~$0.50 |
| OpenAlex API | ~100 次请求（需免费 API Key）| 免费（遵守速率限制） | $0 |
| Crossref + Unpaywall | 元数据回填与开放获取落地 | 免费（遵守速率限制） | $0 |
| arXiv API | ~50 次请求 | 免费 | $0 |
| Gemini 3.1 Pro API | 1 次长上下文调用 (~100K tokens) | $1.25/M input, $5.00/M output | ~$0.50–1.00 |
| Census / OSM API | 若干次请求 | 免费 | $0 |
| **合计** | | | **~$1.50–4.50 / 次运行** |

### 6.2 硬件需求

- 最低配置：8GB RAM、10GB 可用磁盘空间、稳定的互联网连接
- 推荐配置：16GB RAM（便于加载大量 PDF 文本进行上下文组装）

---

## 七、风险分析与缓解策略

| 风险 | 影响 | 概率 | 缓解策略 |
|------|------|------|----------|
| API Key 泄露 | 高 | 中 | 使用 `.env` + `.gitignore` 严格隔离；不在代码中硬编码密钥 |
| 数据源 API 不可用或变更 | 中 | 中 | 依靠 Claude Code 动态抓取的强鲁棒性；Agent 会当场改变脚本应对 API 变更或解析死胡同 |
| 多源元数据检索结果口径不一致 | 中 | 中 | 以 DOI 为主键去重；Crossref 回填标准字段并保留来源优先级规则 |
| AI 生成内容包含幻觉引用 | 高 | 中 | System Prompt 严格限制仅引用 `references.bib` 中已有文献；输出后校验引用键是否存在 |
| 文献 PDF 下载率低（大量付费墙） | 低 | 高 | 降级策略确保至少获取 Abstract；依赖开放获取源 |
| 单次运行 API 成本超预期 | 低 | 低 | 设置 token 用量上限；分模块执行可随时中断 |
| 动态代码执行的安全风险 | 高 | 中 | **控制性委托**：移交 Claude Code 在自身安全外壳内处理作业，对于涉及重型环境操作的任务走人机确认流程即可 |

---

## 八、学术伦理与合规声明

### 8.1 AI 辅助研究的边界

本系统的定位是**研究辅助工具**，而非研究成果的替代品：

- **初稿声明**：所有 AI 生成的内容（`Draft_v1.md`）均在文件头部包含显著的 AI 辅助生成标注，明确提示研究者该内容需要全面审校。
- **引用诚信**：系统仅引用已实际检索并下载的文献，不编造虚假引用。研究者须逐条核实引用内容的准确性。
- **原创性要求**：AI 生成的初稿仅作为写作框架与素材整理，研究者必须注入原创分析见解，不可将系统输出直接作为最终学术成果提交。
- **透明性**：使用本系统辅助的研究成果，应在论文的致谢或方法论部分如实披露 AI 工具的使用情况。

### 8.2 数据合规

- 仅访问公开、免费、无需特殊授权的数据源
- 严格遵守各 API 的使用条款与速率限制
- 不存储、传输或处理任何个人隐私数据
- 下载的文献仅限于开放获取（Open Access）版本，不尝试绕过任何付费墙

---

## 九、项目目录结构

```
auto-pi/
├── .env                         # API Keys（已加入 .gitignore）
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
│
├── ui/
│   └── app.py                   # 实时监控仪表盘 (Streamlit / Gradio)
│
├── config/
│   └── research_plan.json       # Module 1 输出：研究计划配置
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py          # LangGraph DAG 工作流定义
│   ├── memory_retriever.py      # 本地历史记忆检索模块
│   ├── ideation_agent.py        # Module 1: 选题与研究设计
│   ├── literature_agent.py      # Module 2: 文献挖掘
│   ├── drafter_agent.py         # Module 3: 学术起草
│   └── data_fetcher_agent.py    # Module 4: 数据采集与熔断 (内部封装 Claude CLI 桥接)
│
├── data/
│   ├── literature/              # Module 2 输出：PDF + 摘要文本
│   │   ├── index.json           # 文献索引（获取状态记录）
│   │   └── cards/               # 文献证据卡片（结构化摘要与可追溯片段）
│   └── raw/                     # Module 4 输出：当前任务引用的原始数据（含全局归档库的软链接）
│       └── manifest.json        # 数据清单（文件元数据）
│
├── data_vault/                  # 全局长效数据归档库（脱离单任务生命周期）
│   ├── census/                  # 积累复用的普查数据包
│   ├── osm/                     # 积累复用的 GeoJSON 底图
│   └── vault_index.sqlite       # 全局数据集核心元数据与哈希查重索引
│
├── output/
│   ├── Draft_v1.md              # Module 3 输出：论文初稿（可选，但不包含任何实证分析结果）
│   ├── references.bib           # Module 2 输出：BibTeX 引用库
│   ├── topic_screening.json     # Module 1 输出：候选题筛选结果（含证据/淘汰原因/打分）
│   ├── topic_screening.md       # Module 1 输出：便于人工阅读的筛选报告
│   ├── literature_index.json    # Module 2 输出：文献索引摘要（便于快速浏览）
│   └── run_index.json           # 每次运行的统一索引入口（指向所有关键产物）
│
├── prompts/
│   └── academic_drafter.txt     # Gemini System Prompt 模板
│
└── tests/
    ├── test_ideation.py
    ├── test_literature.py
    ├── test_drafter.py
    └── test_data_fetcher.py
```

---

## 附录：与原方案的主要修订对照

| 原方案内容 | 问题 | 修订方案 |
|-----------|------|----------|
| 使用 "Google Antigravity" 作为主控台 | 该产品不存在 | 改用 **LangGraph** 作为工作流编排框架 |
| MCP 协议在 Antigravity 中管理模块 | 场景不适配 | 改用 LangGraph 的 DAG 节点管理模块通信 |
| 通过 `pexpect` / `subprocess` 控制 Claude Code CLI | 极度脆弱，遇到CLI提示词、授权或更新提示等UI交互极易挂起造成工作流死锁 | **重构为 Anthropic API + 工具调用沙箱**，运用状态机(State Machine)闭环跑通"写代码->测试->报错修复"，健壮性和可预测性大幅提高。 |
| 在 LangGraph 的 ResearchState 里传递满载数据的全量字典 | 在文献较多的状况下，易引发 SQLite Checkpoint 剧烈性能衰减或造成 LLM Context 溢出 | **系统化改为了引用传递 (Pass-by-Reference)**，在 State 内全转存为文件的路径指针，各节点依靠指向按需将缓存文件加载入内存处理。 |
| 未提及 API 速率限制 | 可能触发封禁 | 为每个外部 API 添加速率限制器 |
| 未提及成本估算 | 无法评估可行性 | 新增第六节成本估算 |
| 未提及学术伦理 | AI 生成内容的合规风险 | 新增第八节学术伦理声明 |
| 6 天完成全部开发 | 时间过于紧凑 | 调整为 **8 天**，增加集成测试阶段 |
| 技术栈表格格式混乱 | 不可读 | 重新组织为标准 Markdown 表格 |
