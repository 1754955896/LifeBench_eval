# LifeBench_eval 记忆系统评估框架

用于评估 Mem0、Hindsight、Cognee、EverMemos 等记忆系统在多源生活场景数据（LifeBench）上的表现。

## 功能概述

- **多系统评测**：Mem0、Hindsight、Cognee、EverMemos、MindMemos 等系统
- **四阶段流水线**：ADD → SEARCH → ANSWER → EVALUATE，可整体运行，也可只跑指定阶段
- **按日期交叉执行**：ADD 与 SEARCH 按日期推进，先摄入当日数据，再检索当日问题
- **断点续跑**：结果与 checkpoint 增量落盘，重跑同一条命令会自动跳过已完成部分
- **资源追踪**：单次操作耗时（PerOpTracker）+ CPU/内存/存储周期采样（GlobalMonitor）+ LLM token 记账
- **LLM 代理**：统一的 LLM 转发与 token 计量服务，可加载系统特定变换修正请求格式
- **LLM Judge 评估**：由大模型对答案打分，支持多次判定取多数

## 目录结构

```
LifeBench_eval/
├── cli.py              # 入口命令行工具
├── src/                # 核心源码
│   ├── adapters/       # 适配器：连接框架与各记忆系统
│   ├── builders/       # 构建器：启动被测系统的运行环境
│   ├── pipeline/       # 流水线：四阶段调度与断点管理
│   ├── evaluators/     # 评估器：LLM Judge
│   ├── loaders/        # 数据加载器
│   ├── formatters/     # 检索结果格式化
│   ├── models/         # 数据模型
│   ├── trackers/       # 资源追踪（per-op + 全局监控 + LLM token）
│   └── utils/          # 工具函数（含 LLM 代理服务、recall评测）
├── config/
│   ├── systems/        # 被测系统配置（每系统一个 yaml）
│   └── datasets/       # 数据集元数据
├── datasets/           # 评测数据集
├── systems/            # 被测系统源码与部署配置
├── results_clean/      # 归档的历史评测结果（.zip）
├── results/            # 评测输出目录（运行时生成，不入库）
├── env.template        # 环境变量模板
└── requirements.txt
```

## 快速开始

### 1. 配置环境

```bash
pip install -r requirements.txt

# 复制环境变量模板，填入 LLM / Embedding / Rerank 的 API Key
cp env.template .env
```

### 2. 运行评测

```bash
# 完整评测（ADD + SEARCH + ANSWER + EVALUATE）
python cli.py --dataset lifebench --system mem0

# 只跑部分阶段（answer/evaluate 会从输出目录读取上一阶段的结果文件）
python cli.py --dataset lifebench --system mem0 --stages answer evaluate

# 强制串行 + 开启资源追踪
python cli.py --dataset lifebench --system mem0 --serial --enable-tracker

# 分段长跑（按对话下标切片，结果落在同一输出目录，可续跑）
python cli.py --dataset lifebench --system mem0 --from-conv 0 --to-conv 4
```

按对话分段运行时，各段结果会自动合并进同一输出目录，重跑已完成的片段会被自动跳过。

### 3. 查看结果

输出目录默认为 `results/{dataset}-{system}`（可用 `--output-dir` 指定）：

```
results/lifebench-mem0/
├── checkpoint_default.json     # 断点：已完成的阶段 / 日期 / QA
├── add_latency.json            # 每个 session 的写入耗时
├── search_latency.json         # 每个问题的检索耗时
├── search_results.json         # 检索结果（增量写入）
├── answer_results.json         # 回答结果
├── eval_results.json           # LLM Judge 评分
├── report.txt                  # 文本报告
├── tracker/                    # --enable-tracker 时生成
│   ├── tracker_records.json
│   └── global_resource_timeline.json
└── debug/                      # --debug 时生成
```

历史评测结果已归档在 `results_clean/`（压缩包 + 文件字段说明，见 `results_clean/README.md`）。

## CLI 参数

| 参数 | 说明 |
|------|------|
| `--dataset TEXT` | 数据集名称，对应 `config/datasets/{name}.yaml`（必需） |
| `--system TEXT` | 系统名称，对应 `config/systems/{name}.yaml`（必需） |
| `--stages STAGE...` | 要执行的阶段，可多选：`add` `search` `answer` `evaluate`，默认全部 |
| `--output-dir PATH` | 输出目录，默认 `results/{dataset}-{system}` |
| `--run-name TEXT` | 运行名，默认输出目录变为 `results/{dataset}-{system}-{run_name}` |
| `--from-conv N` | 起始对话下标（含），0-based，默认 0 |
| `--to-conv N` | 结束对话下标（不含），默认全部 |
| `--serial` | 强制串行执行，覆盖 runner 的并发设置 |
| `--smoke` | 冒烟测试模式 |
| `--smoke-messages N` | 冒烟测试消息数（默认 10） |
| `--smoke-questions N` | 冒烟测试问题数（默认 3） |
| `--enable-tracker` | 开启资源追踪 |
| `--tracker-interval N` | GlobalMonitor 采样间隔秒数（默认 5.0） |
| `--debug` | 输出详细的 ingestion 日志到 `debug/` |

## 核心概念

### Builder（构建器）

负责启动被测记忆系统的运行环境（Docker 容器、进程内实例或云服务配置）。每个系统对应一个 Builder，可参考对应config、system文件了解具体启动方式与关键配置：

| Builder | 系统 | 启动方式 |
|---------|------|---------|
| `Mem0Builder` | Mem0 | docker-compose (PostgreSQL + Mem0 Server) |
| `GraphitiBuilder` | Graphiti | docker-compose (Neo4j) |
| `HindsightBuilder` | Hindsight | docker-compose (AlloyDB) |
| ... | ... | ... |

### Adapter（适配器）

定义框架与记忆系统的统一接口：

```python
class BaseAdapter:
    async def add_chunks(self, chunks: List[MessageChunk]) -> None
    async def search(self, query: str, user_id: str, top_k: int = 5) -> List[SearchResult]
    async def answer(self, query: str, context: List[str]) -> str
    async def close(self) -> None
```

### Pipeline（流水线）

四阶段评测流程：

```
ADD + SEARCH → ANSWER → EVALUATE
  (按日期交叉)    (生成答案)   (LLM Judge)
```

- **ADD / SEARCH**：按日期推进，先写入当日 session，再检索当日问题，缓存检索上下文
- **ANSWER**：基于检索到的上下文生成答案
- **EVALUATE**：LLM Judge 按题目类型打分（Single_hop、Multi_hop、Temporal、Conflict、Unanswerable、Pattern_recognition、Causal、Knowledge_update、Hidden_info 等）

单独执行某个阶段时，会自动从输出目录读取上一阶段的结果文件，无需重跑前置阶段。

## 资源追踪（Tracker）

加 `--enable-tracker` 后启动三层追踪，全部写入 `{output_dir}/tracker/`：

| 组件 | 作用 | 产出 |
|------|------|------|
| PerOpTracker | 记录每次 add/search 的耗时与元数据 | `tracker_records.json` |
| GlobalMonitor | 后台线程周期性采样 CPU / 内存 / 存储，并在操作边界插入 op_start / op_end 标记 | `global_resource_timeline.json` |
| SystemTracker | 采集被测系统特有的指标（容器负载、数据库大小、LLM token 等） | 汇总进时间线，结束时打印 summary |

SystemTracker 按系统名注册，未注册的系统会自动回退到 `default`（进程级 CPU/内存 + LLM token）。

其中 LLM token 指标来自 LLM 代理：系统配置里的 `llm_proxy_url` 指向代理地址，tracker 通过 `/token-stats` 轮询累计用量。

新增系统 tracker：继承 `DefaultTracker`，加 `@register_tracker("系统名")` ，并在 `src/trackers/system_trackers/__init__.py` 中导入该模块。

## LLM 代理

`src/utils/llm_proxy.py` 是一个 OpenAI 兼容的转发服务：接收被测系统的 LLM 请求 → 转发到真实上游 → 记录 token 用量。可选地加载某个系统的 tracker 变换，对请求/响应做归一化。

```bash
# 默认透传
python -m src.utils.llm_proxy

# 加载 cognee tracker 的请求/响应变换
python -m src.utils.llm_proxy -s cognee
```

- 监听端口默认 18443，可用环境变量 `LLM_PROXY_PORT` 修改
- 上游地址与密钥取自 `.env` 的 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`
- token 统计落盘到 `results/llm_token_stats.json`（累计值 + 逐请求明细）
- 端点：`POST /v1/chat/completions`、`GET /token-stats`、`DELETE /token-stats`（重置）、`GET /health`

**让被测系统走代理**，两种方式任选：

```bash
# 方式一：系统配置里声明（builder 会把它注入被测系统）
# config/systems/*.yaml
llm_proxy_url: "http://localhost:18443"

# 方式二：环境变量覆盖
LLM_BASE_URL='http://localhost:18443' python cli.py --dataset lifebench --system graphrag
```

注意事项：

- 必须用 `python -m src.utils.llm_proxy` 启动；直接 `python src/utils/llm_proxy.py` 会因 `src/utils/logging.py` 遮蔽标准库 `logging` 而报循环导入
- 代理进程不要在 `LLM_BASE_URL` 指向 18443 的终端里启动，否则请求会转发给自己
- 代理是独立进程，先启动代理再跑评测

## 数据集与系统配置

### 数据集（`config/datasets/`）

| 名称 | 说明 |
|------|------|
| `lifebench` | 标准基准：多源生活数据，10 人 / 3380 QA |
| `lifebench_offline` | 离线变体：所有 QA 的 ask_time 固定为 2025-12-31 |
| `lifebench_dense` | 密集子集：3 人 / 453 QA，全部上下文集中在 04~09 月 |
| `lifebench_sparse` | 稀疏子集：3 人 / 453 QA，干扰项均匀分布在 01~12 月 |
| `lifebench_event` | 事件粒度版本：3 人 / 1008 QA 
| `lifebench_locomo_1people` | 单人子集：1 人 / 328 QA |
| `lifebench_locomo_3people` | 三人子集：1008 QA |
| `locomo` | 原始 LoCoMo 格式，10 人 |
| `smoke` / `locomo_smoke` | 冒烟测试用的小规模数据 |

### 被测系统（`config/systems/`）

已完整接入（配置 + adapter 齐备）：`mem0`、`hindsight`、`cognee`、`graphiti_local`、`graphrag`、`evermemos`、`mindmemos`、`memos_cloud`、`memu_cloud`等。

配置中声明的模型、检索 top_k、并发度等参数会透传给 builder 与 adapter；`llm_proxy_url` 用于把该系统的 LLM 流量导向代理。

## 添加新系统

1. **实现 Builder**：在 `src/builders/` 中创建 `{system}_builder.py`，并注册到 `_BUILDER_MODULES`
2. **实现 Adapter**：在 `src/adapters/` 中创建 `{system}_adapter.py`，用 `@register_adapter("{system}")` 注册
3. **添加配置**：在 `config/systems/` 中创建 `{system}.yaml`（可选：`{system}_builder`、`{system}_adapter` 分离命名）
4. **（可选）实现 Tracker**：需要系统特有资源指标时，在 `src/trackers/system_trackers/` 中添加并在 `__init__.py` 注册

## 相关文档

| 文档 | 内容 |
|------|------|
| `src/README.md` | 核心源码结构与接口 |
| `config/README.md` | 系统配置与数据集配置格式 |
| `datasets/README.md` | 数据集目录与数据格式 |
| `systems/README.md` | 各被测系统的部署与接入细节 |
| `results_clean/README.md` | 归档评测结果的文件说明与解压方式 |
