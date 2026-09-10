# results_clean/ 评测结果目录说明

本目录存放 LifeBench 记忆评测的**清洗后最终结果**，每个子目录对应一次独立实验。

结果目录已做清理：删除了 `.bak` 备份、`report.txt`、中间产物 `*_fix.json` / `*_fixed.json` / `recall_checkpoint*.json`，只保留最终（含修复后）的成果文件。

## 命名约定

目录名格式：`lifebench[-_]<变体>-<系统>[-<模型>]`

| 组成 | 含义 | 取值 |
|---|---|---|
| 数据集 | LifeBench 记忆评测基准 | `lifebench`（标准）/ `lifebench_offline`（离线变体）/ `lifebench_locomo_1people`（LOCOMO 单人变体） |
| 系统 | 被测记忆系统 | `cognee` / `graphiti_local` / `hindsight` / `mem0` / `direct_evidence` / `evermemos` / `evermemos_native` / `graphrag` / `memos_cloud` / `memu_cloud` / `mindmemos` |
| 模型 | 系统内的模型变体（消融） | `8b` / `14b` / `32b` / `qwen3.8MAX` / `glm5.2`（hindsight 系列）；`qwen8` / `qwen14` / `qwen32` / `qwen3.8MAX` / `glm5.2`（evermemos 系列，glm5.2 对应目录内 `glm52`） |

示例：

- `lifebench-hindsight-8b` = LifeBench 数据集 × hindsight 系统 × 8b 模型变体
- `lifebench_offline-hindsight` = 离线数据集 × hindsight 系统

## 各实验目录

### lifebench-direct_evidence
- **系统**：直接证据基线（Direct Evidence）
- **说明**：不经过任何检索系统，直接把该问题的 golden 证据拼进 prompt 交给 LLM 作答，作为**上界参考**（upper bound）。因此只有 `answer_results.json` 与 `eval_results.json`，没有检索 / 召回 / 时延文件。

### lifebench-cognee
- **系统**：cognee（图记忆系统）
- **检索**：`retriever_type = graph_completion`

### lifebench-graphiti_local
- **系统**：graphiti 本地图记忆系统
- **检索**：分层检索 `layers = {entity, edge, episode}`

### lifebench-mem0
- **系统**：mem0（`mode = oss`）

### lifebench-hindsight
- **系统**：hindsight 记忆系统（默认配置）

### lifebench-hindsight-{8b,14b,32b,qwen3.8MAX,glm5.2}
- **系统**：hindsight
- **模型变体**：后缀表示不同的模型配置，用于模型消融对比。

### lifebench_offline-hindsight
- **数据集**：离线（offline）数据集变体
- **系统**：hindsight

### lifebench-evermemos
- **系统**：evermemos 记忆系统（默认配置）

### lifebench-evermemos-{qwen8,qwen14,qwen32,qwen3.8MAX,glm52}
- **系统**：evermemos
- **模型变体**：后缀表示不同的系统内模型配置（qwen 系列按参数规模命名，glm52 为 GLM-5.2），用于模型消融对比。

### lifebench-graphrag
- **系统**：graphrag 图记忆系统

### lifebench-memos_cloud
- **系统**：memos（云端部署）

### lifebench-memu_cloud
- **系统**：memu（云端部署）
- **召回评测**：采用 text-match 方式，召回结果文件为 `recall_results_text_match.json`

### lifebench-mindmemos-schema
- **系统**：mindmemos
- **变体**：schema 结构化记忆变体

### lifebench_offline-evermemos_native
- **数据集**：离线（offline）数据集变体
- **系统**：evermemos native（本地部署，无外部依赖）

### lifebench_offline-graphrag
- **数据集**：离线（offline）数据集变体
- **系统**：graphrag

### lifebench_locomo_1people-{cognee,graphiti_local,hindsight,mem0,graphrag,mindmemos,evermemos}
- **数据集**：LOCOMO 单人（`locomo_1people`）变体
- **系统**：cognee / graphiti_local / hindsight / mem0 / graphrag / mindmemos / evermemos
- **说明**：单人 LOCOMO 数据集上的七系统对比，与标准 `lifebench` 数据集相比仅题目来源不同；该变体未运行召回评测，故无 `recall_results.json`。


## 文件说明

每个实验目录内文件命名统一，含义如下：

| 文件 | 含义 |
|---|---|
| `search_results.json` | 检索结果：每个问题从记忆系统检索到的上下文（context），含 `retrieval_metadata`（adapter、total_results 等） |
| `answer_results.json` | 问答结果：基于检索上下文生成的 `answer`、`golden_answer`、题目元信息（`question_type`、`score_points`、`conversation_id` 等） |
| `eval_results.json` | 问答评测：LLM judge 对答案打分（`accuracy`、`weighted_score`），metadata 含 judge 模型与修复信息 |
| `recall_results.json` | 召回评测：证据覆盖率 / 可回答性 / 精确率（`recall`、`recall@k`、`precision`、`answerable_rate` 等） |
| `add_latency.json` | 写入时延：每个 session 导入记忆系统的耗时（`latency_seconds`） |
| `search_latency.json` | 检索时延：每个问题的检索耗时（`latency_seconds`） |
| `checkpoint_default.json` | 流水线阶段检查点（`run_name`、`completed_stages` 等） |
| `llm_token_stats.json` | LLM token 用量统计（`prompt_tokens` / `completion_tokens` / `total_tokens` / `request_count` / `records`），仅较新的实验目录包含 |

### 各文件详细说明

**`search_results.json`**（列表，3380 条）

每个问题一条检索记录，字段：`question_id` / `query` / `conversation_id` / `results`（检索到的 top-k 上下文）/ `retrieval_metadata`（`adapter`、`total_results`、系统特有字段如 cognee 的 `retriever_type`、graphiti 的 `layers`、hindsight 的 `budget`/`entities`、mem0 的 `mode`）。

**`answer_results.json`**（列表，3380 条）

每个问题一条问答记录，字段：`question_id` / `question` / `answer`（生成答案）/ `golden_answer` / `category` / `conversation_id` / `formatted_context` / `metadata`（`ask_time`、`question_type`、`score_points` 等）。

**`eval_results.json`**（对象）

- 顶层：`total_questions` / `correct` / `accuracy` / `weighted_score` / `detailed_results` / `metadata`
- `metadata`：judge 模型 `deepseek-v4-flash`，`num_runs=3` + `aggregation=majority_vote`（三次判定取多数），以及 `fix_info` / `rejudged` 字段——记录了对 judge 空响应失败题的修复情况（修复后 accuracy 更高）。

**`recall_results.json`**（对象）

- 顶层标量指标：`recall`（= macro coverage）、`recall_at_5` / `recall_at_20`、`precision` / `precision_at_5` / `precision_at_20`、`answerable_rate`、`covered_and_answerable_rate`、`coverage_rate_micro`、`redundancy`、`avg_results_per_question`、`avg_tokens_per_*` 等
- `per_question`：每问题明细（2795 个有证据映射的问题）
- `by_source`：按证据来源（sms/note/calendar/photo/call/push/agent_chat）分组的覆盖率
- `rejudge_metadata`：召回 judge（`glm-5.2`）失败题的修复统计（targeted/recovered/still_failed）

**`add_latency.json`**（列表，约 3650 条 session）

字段：`date` / `session_id` / `num_chunks` / `num_messages` / `latency_seconds` / `added` / `failed` / `metadata`。

**`search_latency.json`**（列表，3380 条）

字段：`question_id` / `conversation_id` / `latency_seconds`。

**`checkpoint_default.json`**（对象）

流水线执行进度，字段：`run_name` / `completed_stages` / `answered_qa_ids` / `sample_add_completed` / `sample_search_completed` / `date_add_completed` / `date_search_completed` / `last_updated`。

## 共享模型配置（全局，非目录内区分项）

这些配置在所有实验目录中一致，由 `.env` / `config/systems/*.yaml` 统一控制，不作为目录名区分项：

| 用途 | 模型 |
|---|---|
| 答案生成 LLM（answer LLM） | `deepseek-v4-flash` |
| 问答评测 judge | `deepseek-v4-flash` |
| 召回评测 judge（recall judge） | `glm-5.2` |
| 默认向量化（embedding） | `Qwen/Qwen3-Embedding-4B` |
| 默认重排（reranker） | `Qwen/Qwen3-Reranker-4B` |

hindsight 系列的 `-8b/-14b/-32b/-qwen3.8MAX/-glm5.2` 后缀即是对 hindsight 系统内模型（embedding / 检索模型）的消融变体。

## 压缩归档说明

原始数据子目录（`results_clean/*/`）体积过大（单个 `search_results.json` 可达 400+ MB，超过 GitHub 100 MB 单文件上限），故已整体 gzip/zip 压缩成 `.zip` 归档后提交，原始子目录在 `.gitignore` 中被忽略（`results_clean/*/`）。

归档文件与本 README 同级，命名规则：

- 每个实验一个 `.zip`：`<目录名>.zip`（解压后还原为 `results_clean/<目录名>/`）
- `lifebench-cognee` 因体积最大，拆成两个包：
  - `lifebench-cognee_search_results.zip`（仅 `search_results.json`）
  - `lifebench-cognee_rest.zip`（其余文件）
  - 两者解压到同一目录即可还原完整 `lifebench-cognee/`

压缩为**无损**（zip DEFLATE / LZMA），解压后与原文件逐字节一致。

注意：`lifebench-evermemos-glm52.zip`、`lifebench-evermemos-qwen14.zip` 与 `lifebench-mindmemos-schema.zip` 的单个 JSON 即使 DEFLATE 后仍超过 100 MB 上限，故改用 **LZMA 压缩**（体积分别约 13 MB / 12 MB / 47 MB）。LZMA 归档只能用**方式一（Python zipfile）**解压，`unzip` 与 PowerShell `Expand-Archive` 不支持。

### 解压还原

所有归档都内嵌了顶层目录名（`<目录名>/<文件>`），把全部 `.zip` 解压到 `results_clean/` 即自动还原原始目录结构；cognee 的两个包解压到同一位置会自动合并回完整的 `lifebench-cognee/`。

在**项目根目录**（`LifeBench_eval/`）下，任选一种方式：

**方式一：Python（跨平台，推荐，本仓库环境必装）**

```bash
python -c "import zipfile, glob; [zipfile.ZipFile(f).extractall('results_clean') for f in glob.glob('results_clean/*.zip')]"
```

**方式二：Git Bash / Linux / macOS（`unzip`）**

```bash
cd results_clean && unzip -o '*.zip' && cd ..
```

**方式三：Windows PowerShell（`Expand-Archive`）**

```powershell
Get-ChildItem results_clean\*.zip | ForEach-Object { Expand-Archive $_.FullName -DestinationPath results_clean -Force }
```

解压完成后，`results_clean/<目录名>/` 会重新出现（这些子目录仍被 `.gitignore` 忽略，不会进入版本控制）。