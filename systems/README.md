# 被测系统目录

存放各被测记忆系统的源码和部署配置。

## MindMemOS

**运行方式**：Docker 基础设施 + FastAPI 服务器（HTTP 通信）

### 需要配置的文件

| 文件 | 说明 | 操作 |
|------|------|------|
| `LifeBench_eval/.env` | 全局 API Key | 填写 LLM_API_KEY, VECTORIZE_API_KEY 等 |
| `systems/MindMemOS/.env` | Docker 容器连接参数 | 从 `.env.example` 复制，按需修改 |
| `systems/MindMemOS/config/mindmemos/dev.yaml` | 服务端配置 | 从 `dev.example.yaml` 复制，按需修改 |
| `config/systems/mindmemos.yaml` | 流水线参数 | 一般无需修改；可选调整 memory_algorithm、search_top_k 等 |

Builder 会自动生成 API key 写入 `config/mindmemos/api_keys.yaml`，无需手动处理。

### 需要启动的服务

Builder（[src/builders/mindmemos_builder.py](../src/builders/mindmemos_builder.py)）自动处理以下启动流程：

1. **Docker 基础设施** — Qdrant（向量库，端口 6333）、Neo4j（图数据库，端口 7474/7687）、Kafka（消息队列，端口 9092）
2. **NLP 资源安装** — 通过 `scripts/install_nlp_assets.py` 安装
3. **API Server** — `uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000`

### 手动启动（Builder 失败时备用）

```bash
# 1. 启动 Docker 服务
docker compose --env-file systems/MindMemOS/.env \
  -f systems/MindMemOS/dockers/docker-compose.memory.yml \
  up -d --wait qdrant neo4j kafka kafka-ui kafka-exporter

# 2. 启动 API Server
cd systems/MindMemOS
.venv/Scripts/python.exe -m uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### 核心流程

Adapter（[src/adapters/mindmemos_adapter.py](../src/adapters/mindmemos_adapter.py)）通过 `mindmemos_sdk` 与 API Server 通信：

- **ADD** → `POST /v1/memory/add`，按 session 批量写入
- **SEARCH** → `POST /v1/memory/search`，支持 agentic（多跳图谱搜索）和 fast（向量搜索）两种策略
- **ANSWER** → Adapter 内置的 LLM 调用（prompt 与 mindmemos_eval 一致），不做第二次检索

内存算法在 `config/systems/mindmemos.yaml` 中通过 `memory_algorithm` 切换：
- `vanilla` — 扁平记忆 + 向量搜索
- `schema` — 实体建模 + 图谱 + Agentic 多跳搜索

---

## Hindsight

**运行方式**：纯本地进程内（无 Docker，使用嵌入式 pg0 数据库）

### 需要配置的文件

| 文件 | 说明 | 操作 |
|------|------|------|
| `LifeBench_eval/.env` | 全局 API Key | 填写 LLM_API_KEY, VECTORIZE_API_KEY 等 |
| `systems/hindsight/.env` | Hindsight 环境变量 | 可为空；Builder 从 YAML 注入所有环境变量 |
| `config/systems/hindsight.yaml` | 流水线参数 | 按需修改 LLM provider/model、embedding provider 等 |

Hindsight 的环境变量由 Builder（[src/builders/hindsight_builder.py](../src/builders/hindsight_builder.py)）从 YAML 配置中注入，不走 `.env` 文件。如需手动运行 Hindsight 自身（不通过流水线），则需配置 `systems/hindsight/.env`（参考 `.env.example`）。

### 需要启动的服务

**无需启动任何外部服务。** Builder 会在进程内创建 `MemoryEngine` 实例，自动管理嵌入式 pg0 数据库的生命周期。

### 核心流程

Adapter（[src/adapters/hindsight_adapter.py](../src/adapters/hindsight_adapter.py)）直接调用 `MemoryEngine`：

- **ADD** → `memory.retain_batch_async()` — 按 conversation 分组批量写入
- **SEARCH** → `memory.recall_async()` — 多策略检索（语义 + BM25 + 图谱 + 时序）+ 重排序
- **ANSWER** → 独立 LLM 调用，使用结构化输出（`QuestionAnswer` pydantic model）

关键配置项（在 `config/systems/hindsight.yaml` 中）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `budget` | `mid` | 检索候选量级：low(~100) / mid(~300) / high(~1000) |
| `db_url` | `pg0` | 数据库：pg0 为嵌入式，也可配置外部 PostgreSQL |
| `memory_llm_*` | deepseek-v4-flash | 记忆提取/巩固用的 LLM |
| `answer_llm_*` | 同 memory | 答案生成用的 LLM |
| `HINDSIGHT_API_EMBEDDINGS_*` | SiliconFlow | 向量化服务配置 |
| `HINDSIGHT_API_RERANKER_*` | SiliconFlow | 重排序服务配置 |

---

## Cognee

**运行方式**：纯本地进程内（无 Docker，使用 SQLite + LanceDB + Ladybug 文件存储）

### 需要配置的文件

| 文件 | 说明 | 操作 |
|------|------|------|
| `LifeBench_eval/.env` | 全局 API Key | 填写 LLM_API_KEY, VECTORIZE_API_KEY 等 |
| `systems/cognee/.env` | Cognee 环境变量 | Builder 自动生成，无需手动创建 |
| `config/systems/cognee.yaml` | 流水线参数 | 按需修改 LLM、Embedding、检索策略等 |

Builder（[src/builders/cognee_builder.py](../src/builders/cognee_builder.py)）在每次运行时将 YAML 配置翻译为 `systems/cognee/.env`，cognee SDK 加载该文件获取配置。**不需要手动编辑 `systems/cognee/.env`**。

### 需要启动的服务

**无需启动任何外部服务。** Cognee 使用文件存储：
- **SQLite** — 元数据和状态存储
- **LanceDB** — 向量嵌入存储
- **Ladybug** — 知识图谱存储

所有数据默认存放在 `D:/cg/data/` 和 `D:/cg/system/`（可通过 `config/systems/cognee.yaml` 中的 `data_root_directory` 和 `system_root_directory` 修改）。

### 核心流程

Adapter（[src/adapters/cognee_adapter.py](../src/adapters/cognee_adapter.py)）直接调用 cognee SDK，复刻 BEAM eval 流程：

- **ADD** → `cognee.add()` + `cognee.cognify()` — 先写入原始数据，再运行 5 任务流水线（分类 → 分块 → 提取图谱 → 摘要 → 存储）
- **SEARCH** → `cognee.search(query_type=GRAPH_COMPLETION, only_context=True)` — 图谱遍历 + 上下文检索（不做 LLM 补全）
- **ANSWER** → `cognee.generate_completion()` — 对缓存的图谱上下文做 LLM 补全

**性能提示**：cognify 阶段包含 LLM 实体/关系提取，耗时较长。对于不会被搜索的 session-only 日期（无 QA），pipeline 会自动跳过 cognify，仅执行 add。

关键配置项（在 `config/systems/cognee.yaml` 中）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `retriever_type` | `graph_completion` | 检索策略：支持 graph_completion / graph_completion_cot / graph_summary_completion 等 |
| `top_k` | 40 | 最终返回的记忆数 |
| `wide_search_top_k` | 100 | 宽搜索候选池大小 |
| `chunk_size` | 1024 | cognify 分块 token 预算 |
| `enable_backend_access_control` | true | 多租户数据隔离（每用户独立 Ladybug 图文件） |
| `prune_on_init` | false | 每次运行前清空状态（基准测试应设为 true） |
| `sqlite_timeout` | 300 | SQLite 并发写超时（秒） |

---

## Mem0

**运行方式**：Docker 基础设施 + HTTP API（OSS Server）

### 需要配置的文件

| 文件 | 说明 | 操作 |
|------|------|------|
| `LifeBench_eval/.env` | 全局 API Key | 填写 LLM_API_KEY, VECTORIZE_API_KEY, RERANK_API_KEY 等 |
| `systems/mem0/server/.env` | Docker 容器环境变量 | Builder 从 `.env` 和 YAML 自动生成，无需手动创建 |
| `config/systems/mem0.yaml` | 流水线参数 + runtime_config | 按需修改 LLM/Embedder provider、model 等 |

Builder（[src/builders/mem0_builder.py](../src/builders/mem0_builder.py)）从 `LifeBench_eval/.env` 和 YAML 的 `runtime_config` 块自动生成 `systems/mem0/server/.env`，并启动容器后通过 `POST /configure` 推送运行时 LLM/Embedder 配置到 mem0 server。**不需要手动编辑 `systems/mem0/server/.env`**。

### 需要启动的服务

Builder 自动处理以下启动流程：

1. **PostgreSQL** — Docker 容器，pgvector 扩展用于向量存储（端口 5432）
2. **Mem0 OSS Server** — Docker 容器，FastAPI 服务（端口 8888）
3. **Mem0 Dashboard** — Docker 容器，可选 Web UI

### 手动启动（Builder 失败时备用）

```bash
cd systems/mem0/server

# 1. 准备 .env（从 LifeBench_eval/.env 和 config/systems/mem0.yaml 参考生成）
cp .env.example .env
# 编辑 .env，填入 LLM/Embedding API Key

# 2. 启动服务
docker compose -f docker-compose.yaml up -d --wait

# 3. 验证服务
curl http://localhost:8888/configure
```

### 核心流程

Adapter（[src/adapters/mem0_adapter.py](../src/adapters/mem0_adapter.py)）通过 HTTP 与 Mem0 OSS Server 通信：

- **ADD** → `POST /memories` — 按 batch（默认 4 条消息一组）并发写入，支持 `infer` 开关控制是否由 LLM 提取事实
- **SEARCH** → `POST /search` — 向量搜索 + Cohere 兼容 reranker（SiliconFlow），按 user_id 隔离
- **ANSWER** → Adapter 内置的独立 LLM 调用（7 步推理 prompt），不做第二次检索

关键配置项（在 `config/systems/mem0.yaml` 中）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `oss` | 运行模式：oss（Docker 本地）/ cloud（Mem0 Cloud API） |
| `infer` | `true` | 是否由 LLM 提取事实（false 则原样存储文本） |
| `add_batch_size` | 4 | 每次 API 调用合并的消息数（让 LLM 看到多轮对话上下文） |
| `add_concurrency` | 5 | ADD 阶段并发度 |
| `runtime_config.llm` | deepseek | LLM provider / model / base_url（通过 /configure 推送） |
| `runtime_config.embedder` | openai_compatible + SiliconFlow | Embedder provider / model / dimensions |
| `search.top_k` | 40 | 检索返回的记忆数 |
| `search.rerank` | false | 是否启用服务端 rerank（mem0 server 内配置） |

**性能提示**：Mem0 使用 Docker 容器运行，首次冷启动需拉取镜像 + PostgreSQL 初始化（约 30-60s）。pgvector 数据卷在容器重启后保留，旧数据需要手动清理或开启 `prune` 功能。

---

## 四个系统对比

| 维度 | MindMemOS | Mem0 | Hindsight | Cognee |
|------|-----------|------|-----------|--------|
| 运行模式 | Docker + HTTP API | Docker + HTTP API | 进程内直调 | 进程内直调 |
| 外部依赖 | Qdrant, Neo4j, Kafka | PostgreSQL(pgvector) | 无 | 无 |
| 存储 | Qdrant(向量) + Neo4j(图谱) | pgvector(向量) + PostgreSQL | pg0(嵌入式PG) | SQLite + LanceDB + Ladybug |
| ADD 开销 | 低（写 API） | 低（写 API，可选 LLM 提取） | 中（LLM 提取事实） | 高（cognify 5 任务流水线） |
| 搜索策略 | fast / agentic 多跳 | 向量搜索 + rerank | 语义+BM25+图谱+时序 | GRAPH_COMPLETION 图谱遍历 |
| 启动时间 | ~30-90s（Docker + 服务） | ~30-60s（Docker） | <5s | <5s |
