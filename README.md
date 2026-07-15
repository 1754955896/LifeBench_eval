# LifeBench_eval 记忆系统评估框架

用于评估 Mem0、LifeMem 等记忆系统在实际生活场景中的表现。

## 功能概述

LifeBench_eval 是一个通用的记忆系统评测框架，支持：

- **多系统评测**：Mem0、Cognee、Graphiti、Hindsight、EverMemos、MindMemos 等
- **三阶段流水线**：ADD + SEARCH → ANSWER → EVALUATE
- **断点续跑**：支持按 Sample、Date、QA 粒度的断点恢复
- **LLM Judge 评估**：基于大模型对答案质量进行评分
- **Docker 化部署**：被测系统以 Docker 容器方式运行

## 目录结构

```
LifeBench_eval/
├── cli.py              # 入口命令行工具
├── src/                # 核心源码
│   ├── adapters/       # 适配器：连接框架与各记忆系统
│   ├── builders/       # 构建器：启动被测系统的运行环境
│   ├── pipeline/       # 流水线：四阶段调度
│   ├── evaluators/     # 评估器：LLM Judge 等
│   ├── loaders/        # 数据加载器
│   ├── formatters/     # 检索结果格式化
│   ├── models/         # 数据模型
│   └── utils/          # 工具函数
├── config/             # 配置文件
│   ├── systems/       # 被测系统配置
│   └── datasets/       # 数据集元数据
├── datasets/           # 评测数据集
├── systems/            # 被测系统源码（不参与版本控制）
├── tests/             # 集成测试
├── results/            # 评测结果输出
├── output/             # 临时输出
├── .env                # 环境变量（含 API Key，不提交）
└── env.template        # 环境变量模板
```

## 快速开始

### 1. 配置环境

```bash
# 复制环境变量模板
cp env.template .env

# 编辑 .env，填入你的 API Key
vim .env
```

### 2. 运行评测

```bash
# 完整评测
python cli.py --dataset lifebench_locomo_format --system mem0

# 指定输出目录
python cli.py --dataset lifebench_locomo_format --system mem0 --output results/my-run

# 断点续跑
python cli.py --dataset lifebench_locomo_format --system mem0 --resume

# Debug 模式（生成详细日志）
python cli.py --dataset lifebench_locomo_format --system mem0 --debug
```

### 3. 查看结果

评测结果保存在 `results/{dataset}-{system}/` 目录：

```
results/lifebench_locomo_format-mem0/
├── checkpoint_default.json     # 断点记录
├── add_latency.json            # ADD 延迟统计
├── search_lat latency.json         # SEARCH 延迟统计
├── search_results.json         # 检索结果
├── answer_results.json         # 回答结果
├── eval_results.json           # 评估结果
└── report.txt                  # 文本报告
```

## 核心概念

### Builder（构建器）

负责启动被测记忆系统的运行环境（Docker）。每个系统对应一个 Builder：

| Builder | 系统 | 启动方式 |
|---------|------|---------|
| `Mem0Builder` | Mem0 | docker-compose (PostgreSQL + Mem0 Server) |
| `CogneeBuilder` | Cognee | docker-compose |
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
    async def cleanup(self) -> None
```

### Pipeline（流水线）

四阶段评测流程：

```
ADD + SEARCH → ANSWER → EVALUATE
   (按日期交叉)    (生成答案)  (LLM Judge)
```

**ADD + SEARCH**：按日期顺序，交叉执行数据摄入与检索
**ANSWER**：基于检索到的记忆生成答案
**EVALUATE**：使用 LLM Judge 评估答案质量，支持以下类型：

- Single_hop、Multi_hop、Temporal、Conflict
- Unanswerable、Pattern_recognition、Causal
- Knowledge_update、Hidden_info

## 添加新系统

1. **实现 Builder**：在 `src/builders/` 中创建 `{system}_builder.py`
2. **实现 Adapter**：在 `src/adapters/` 中创建 `{system}_adapter.py`
3. **添加配置**：在 `config/systems/` 中创建 `{system}.yaml`
4. **注册系统**：在 Builder 和 Adapter 的 registry 中注册

## 与 LifeMem/evaluation 的区别

| 方面 | LifeMem/evaluation | LifeBench_eval |
|------|-------------------|----------------|
| 项目定位 | 内部评估工具，深度耦合 LifeMem | 通用评估框架，支持多系统 |
| 适配器接口 | adapter add 接收整个 conversation | add 接收分割好的 message chunk |
| 运行方式 | 统一 ADD 后再 SEARCH | 边 ADD 边 SEARCH，按日期交叉 |
| 记忆系统部署 | 基于代码 | 基于 Docker |
| 断点续跑 | Stage 级别 | Date + QA 级别（更细粒度） |
| 结果保存 | 最后一次性保存 | 增量保存 |
| Debug 模式 | 无 | 生成 debug 日志文件 |

## CLI 参数

```
--dataset TEXT         数据集名称（必需）
--system TEXT         系统名称（必需）
--output PATH         输出目录（默认: results/{dataset}-{system}）
--resume              从断点恢复运行
--debug               开启 Debug 模式
--max-workers N       并发线程数（默认: 10）
--rerank-model MODEL  Rerank 模型名称
--rerank-provider PROVIDER  Rerank 提供商
```