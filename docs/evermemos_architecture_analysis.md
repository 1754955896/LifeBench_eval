# EverMemOS 架构分析与 Adapter 对照

## 目录

1. [两个 Adapter 的记忆处理流程差异（核心）](#1-两个-adapter-的记忆处理流程差异核心)
2. [完整数据流](#2-完整数据流)
3. [EverMemOS_bz 涉及的源码文件及作用](#3-evermemos_bz-涉及的源码文件及作用)
4. [YAML 配置文件对比](#4-yaml-配置文件对比)
5. [Builder 实现对比](#5-builder-实现对比)
6. [存储架构](#6-存储架构)
7. [边界检测机制](#7-边界检测机制)
8. [Adapter 方法对照分析](#8-adapter-方法对照分析)
9. [全维度对比总结](#9-全维度对比总结)
10. [已知问题](#10-已知问题)
---

## 1. 两个 Adapter 的记忆处理流程差异（核心）

两个 adapter 都处理"消息摄入 → 记忆提取 → 索引构建 → 检索 → 回答"这一链路，但在每个环节的实现路径截然不同。

### 1.1 HTTP API 版的处理链路

```
cli.py → HTTP POST → FastAPI Controller → 生产代码全链路 → MongoDB + ES + Milvus + Redis
```

每个环节使用的是 **EverMemOS 生产系统**（`src/` 目录）：

| 环节 | 生产代码路径 | 说明 |
|------|-------------|------|
| 消息接收 | `agentic_v3_controller.py` → `group_chat_converter.py` | HTTP 反序列化 + 格式转换 |
| 边界检测 | `memory_layer/memory_manager.py::extract_memcell()` | LLM 判断是否构成完整事件 |
| 消息缓存 | Redis（`conversation_data_repo_impl.py`） | 未达边界时暂存消息 |
| MemCell 提取 | `memory_layer/memcell_extractor/conv_memcell_extractor.py` | 使用生产提取器 |
| 聚类 | `memory_layer/cluster_manager.py`（MongoDB 持久化） | 提取用户画像 |
| 多维记忆提取 | `memory_layer/memory_extractor/`（episode + semantic + event_log） | 三个维度分别提取 |
| 索引存储 | `infra_layer/.../repository/`（MongoDB + ES + Milvus） | 分布式存储 |
| 检索 | `agentic_layer/memory_manager.py`（ES BM25 + Milvus Vector + RRF + Rerank） | 完整检索流水线 |

### 1.2 Native 版的处理链路

```
cli.py → 直接 import → evaluation 编排脚本 + 生产提取组件 → 本地 pickle 文件
```

Native 版的代码来自 **两个目录的混合**：

- `evaluation/src/adapters/evermemos/stage*_*.py` — 编排脚本（消息拼接、索引构建逻辑、检索流程控制）
- `src/memory_layer/*` — 生产提取组件（MemCellExtractor、ClusterManager、LLMProvider 等）
- `src/agentic_layer/*` — 生产服务组件（vectorize_service、rerank_service）

| 环节 | 实际代码来源 | 说明 |
|------|-------------|------|
| 消息拼接 | 自实现（`_convert_message_to_raw`） | 转为 stage1 需要的 dict 格式 |
| **边界检测** | **无** | 所有消息一次性喂给 extractor，无累积过程 |
| MemCell 提取 | `stage1_memcells_extraction.py` → 调用生产 `ConvMemCellExtractor` | 调用了生产代码的提取器 |
| 聚类 | `stage1_memcells_extraction.py` → 调用生产 `ClusterManager` + `ProfileManager` | **但使用 InMemory 存储**，不持久化 |
| 索引构建 | `stage2_index_building.py` 自实现 | 本地 rank_bm25 建 BM25 + 调用生产 vectorize_service 建 Embedding → 存 `.pkl` 文件 |
| 检索 | `stage3_memory_retrivel.py` 自实现 | 加载 `.pkl` → 本地 BM25 / Embedding 搜索，或用 LLM 做简化版 Agentic |
| 回答 | `stage4_response.py` 自实现 prompt 模板 | 调 LLM API |

### 1.3 Native 版调用的生产代码

stage1 导入了这些生产模块：

```python
from memory_layer.llm.llm_provider import LLMProvider          # src/memory_layer/
from memory_layer.memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor  # src/memory_layer/
from memory_layer.memory_extractor.episode_memory_extractor import EpisodeMemoryExtractor  # src/memory_layer/
from memory_layer.memory_extractor.event_log_extractor import EventLogExtractor          # src/memory_layer/
from memory_layer.cluster_manager import ClusterManager          # src/memory_layer/（InMemory 模式）
from memory_layer.profile_manager import ProfileManager          # src/memory_layer/（InMemory 模式）
from api_specs.memory_types import RawDataType                   # src/api_specs/
```

stage3 导入了这些生产模块：

```python
from agentic_layer import vectorize_service                     # src/agentic_layer/
from agentic_layer import rerank_service                        # src/agentic_layer/
from memory_layer.llm.llm_provider import LLMProvider           # src/memory_layer/
```

**Native 版不是"独立实现"**，它的核心提取（MemCell、聚类、Profile）和外部服务（向量化、Rerank）**确实调用了生产代码**。差异在于编排方式、中间数据存储和持久化策略。

---

## 2. 完整数据流

### 2.1 环境启动

**HTTP API 版：**

```
EverMemOSBuilder.build()
  1. docker-compose up -d
     启动: MongoDB :27017 | ES :19200 | Milvus :19530 | Redis :6379
  2. 等待 Docker 45 秒
  3. .venv/Scripts/python.exe src/run.py --port 8001 --env-file .env
     - setup_environment() → 加载 .env，设置 Python 路径
     - setup_all() → 依赖注入，注册所有 Bean
     - uvicorn.run(app) → FastAPI 监听 :8001
  4. 轮询 /docs 端点直到就绪
```

**Native 版：**

```
EverMemOSNativeBuilder.build()
  1. 检查是否在 .venv 环境中（否则自动重启）
  2. 无需 Docker，直接完成
```

### 2.2 ADD 阶段：HTTP API 版

```
adapter.add_chunks(chunks)
  │
  ├── 对每个 conversation_id，先保存元数据:
  │     POST /conversation-meta → MongoDB
  │
  └── 按 dia_id 分组消息，同组拼接
        POST /memorize（逐条发送）
        │
        └── 后端 memoriz() 完整流程:
              │
              ├── 1. preprocess_conv_request()
              │     从 Redis 读取该 group_id 的历史消息
              │     拼接到 history_raw_data_list
              │
              ├── 2. extract_memcell(history + new)
              │     │
              │     ├── 非边界: 新消息存 Redis → 返回 "accumulated"
              │     │
              │     └── 是边界:
              │           ├── 清空 + 重新保存 Redis
              │           ├── 存 MemCell → MongoDB
              │           ├── 聚类 → ClusterManager（MongoDB 持久化）
              │           ├── 提取 Episode → MongoDB + ES + Milvus
              │           ├── 提取 Semantic → MongoDB + ES + Milvus
              │           └── 提取 EventLog → MongoDB + ES + Milvus
              │
              └── 返回 saved_memories / None
```

**关键特点：**
- 消息先到 Redis 排队
- LLM 判断"边界"后才真正提取
- 提取后自动三库同步（MongoDB + ES + Milvus）
- 每次 POST 只发一条拼接后的消息

### 2.3 ADD 阶段：Native 版

```
adapter.add_chunks(chunks)
  │
  └── 对每个 chunk:
        ├── 1. 所有消息追加到内存缓冲区 _conv_buffers
        │      同时更新 _conv_latest_dates[conv_id]（跟踪最新消息日期）
        │
        └── 2. 如果 chunk 的 dia_id 命中 final_date（2025-12-31）:
              │
              └── _build_index_for_conv()
                    │
                    ├── stage1_memcells_extraction.process_single_conversation()
                    │     一次传入所有消息 → ConvMemCellExtractor 批量提取 MemCell
                    │     使用生产 ClusterManager（InMemory）进行聚类
                    │     使用生产 ProfileManager（InMemory）提取画像
                    │     memcells 保存到本地文件 memcells/ 目录
                    │
                    └── stage2_index_building
                          ├── build_bm25_index()
                          │    rank_bm25 库 → 本地 .pkl 文件
                          └── build_emb_index()
                               调用生产 vectorize_service → 本地 .pkl 文件
```

**关键特点：**
- 所有消息在内存累积，无边界检测
- final_date 触发一次性批量处理（但有自动检测 fallback，见 2.5）
- 聚类使用 InMemory 模式，不持久化到 MongoDB
- 索引存本地文件，不写入 ES/Milvus
- 存储结构:
  ```
  output/
  ├── memcells/
  │   └── conv_0_memcells.json
  ├── bm25_index/
  │   └── bm25_index_conv_0.pkl
  └── vectors/
      └── embedding_index_conv_0.pkl
  ```

### 2.4 SEARCH 阶段：HTTP API 版

```
adapter.search(query, conv_id, top_k=20, mode="agentic")
  │
  └── POST /retrieve_agentic
        │
        └── memory_manager.retrieve_agentic()
              │
              ├── Round 1: RRF（Milvus Vector + ES BM25 → RRF 融合）
              ├── Rerank Top 20 → Top 5
              ├── LLM 判断充分性
              │     ├── 充分 → 返回
              │     └── 不充分 → Round 2
              │
              └── Round 2:
                    ├── LLM 生成多个改进查询
                    ├── 并行执行多个 RRF 检索
                    ├── 去重 + 融合 Round1 + Round2
                    └── Final Rerank → Top N
```

### 2.5 SEARCH 阶段：Native 版（已修复）

修复前的问题：Native 版原本只响应 `final_date: "2025-12-31"` 配置的日期，对于日期不匹配的数据集（如 locomo，数据在 2023 年），索引永不构建、搜索全部被缓冲返回空结果 → 3% accuracy。

修复后的逻辑：

```
adapter.search(query, conv_id, ask_time, ...)
  │
  ├── 从 kwargs 或 query 文本提取 ask_time
  │   （runner 现在通过 kwargs 传入 qa.metadata["ask_time"]）
  │
  ├── is_final_date = _is_final_qa(ask_time)  // 检查是否匹配 final_date
  │
  ├── Fallback 1: ask_time 有值且 >= 该 conv 的最新消息日期 → is_final_date = True
  │   （处理数据集有 ask_time 但日期 ≠ final_date 的情况）
  │
  ├── Fallback 2: 无 ask_time 但缓存有消息 → 首次 search 触发索引构建
  │   （处理 locomo 等无 ask_time 的英文数据集）
  │
  ├── 如果索引已构建（buf["index_built"]）→ 强制设为 is_final_date = True
  │   （修复之前后续 search 被继续缓冲的 bug）
  │
  ├── Case 1: NOT final_date → 缓冲搜索请求，返回空结果
  │   （仅当确实未到最终日期时触发）
  │
  └── Case 2: IS final_date
        │
        ├── 构建索引（如果尚未构建）
        ├── 执行所有缓冲的搜索请求
        └── stage3_memory_retrivel
              ├── bm25_search() → 加载本地 .pkl, rank_bm25 搜索
              ├── embedding_search() → 加载本地 .pkl, numpy 余弦相似度
              ├── agentic_retrieval() → 简化版（无多轮 + 无 Rerank）
              └── hybrid_search_with_rrf() → 本地 RRF 融合
```

**修复效果：** 以 locomo 数据集为例，Accuracy 从 **3% → 84.55%**，233 条搜索全部有结果、0 条被缓冲。

### 2.6 ANSWER 阶段

两者基本相同：使用 LLM API 生成答案。

**HTTP 版：** 自实现，拼接 `ANSWER_PROMPT` 模板调 `/chat/completions`

**Native 版：** 调用 `stage4_response.locomo_response()`，内部也是 LLM API 调用，prompt 模板不同

---

## 3. EverMemOS_bz 涉及的源码文件及作用

### 3.1 HTTP API 版涉及的服务端文件

**服务启动与 DI 容器：**

| 文件 | 作用 |
|------|------|
| `src/run.py` | FastAPI 进程入口。解析 `--port 8001`，执行 DI，启动 uvicorn |
| `src/application_startup.py` | `setup_all()` → 加载 Addons → 扫描组件 → DI 容器初始化 |
| `src/app.py` | 创建 FastAPI 实例 + 注册 Controllers + 中间件 |
| `src/base_app.py` | 基础 FastAPI 应用工厂 |
| `src/bootstrap.py` | 应用引导初始化 |

**API 控制器：**

| 文件 | 作用 |
|------|------|
| `src/infra_layer/adapters/input/api/v3/agentic_v3_controller.py` | 4 个核心端点：`/memorize`、`/retrieve_lightweight`、`/retrieve_agentic`、`/conversation-meta` |

**业务逻辑层：**

| 文件 | 作用 |
|------|------|
| `src/agentic_layer/memory_manager.py` | 统一接口：`memorize()`、`retrieve_lightweight()`（ES + Milvus + RRF）、`retrieve_agentic()`（多轮） |
| `src/agentic_layer/agentic_utils.py` | `check_sufficiency()`、`generate_multi_queries()` |
| `src/agentic_layer/rerank_service.py` | 对接外部 rerank 模型 |
| `src/agentic_layer/vectorize_service.py` | 对接外部 embedding 模型 |
| `src/agentic_layer/retrieval_utils.py` | `reciprocal_rank_fusion()` |
| `src/biz_layer/mem_memorize.py` | 记忆提取核心：边界检测 → MemCell → 聚类 → Episode/Semantic/EventLog → 多库同步 |
| `src/biz_layer/mem_db_operations.py` | MongoDB 存储操作 |
| `src/biz_layer/mem_sync.py` | MongoDB → ES + Milvus 同步 |
| `src/biz_layer/conversation_data_repo.py` + `_impl.py` | Redis 消息缓存 |

**数据访问层：**

| 文件 | 作用 |
|------|------|
| `src/infra_layer/.../repository/episodic_memory_raw_repository.py` | MongoDB EpisodicMemory CRUD |
| `src/infra_layer/.../repository/semantic_memory_record_raw_repository.py` | MongoDB SemanticMemory CRUD |
| `src/infra_layer/.../repository/event_log_record_raw_repository.py` | MongoDB EventLog CRUD |
| `src/infra_layer/.../repository/memcell_raw_repository.py` | MongoDB MemCell CRUD |
| `src/infra_layer/.../repository/conversation_meta_raw_repository.py` | MongoDB ConversationMeta CRUD |
| `src/infra_layer/.../repository/group_user_profile_memory_raw_repository.py` | MongoDB Profile CRUD |
| `src/infra_layer/.../search/repository/episodic_memory_es_repository.py` | ES 全文搜索 |

**记忆提取层：**

| 文件 | 作用 |
|------|------|
| `src/memory_layer/memory_manager.py` | `extract_memcell()` 边界检测 |
| `src/memory_layer/memcell_extractor/conv_memcell_extractor.py` | MemCell 提取器 |
| `src/memory_layer/memory_extractor/episode_memory_extractor.py` | Episode 提取 |
| `src/memory_layer/memory_extractor/event_log_extractor.py` | 原子事实提取 |
| `src/memory_layer/cluster_manager.py` | 聚类（MongoDB 持久化） |
| `src/memory_layer/profile_manager.py` | 用户画像（MongoDB 持久化） |
| `src/memory_layer/llm/llm_provider.py` | 统一 LLM 调用 |

**DTO 与数据模型：**

| 文件 | 作用 |
|------|------|
| `src/api_specs/dtos/memory_command.py` | `MemorizeRequest` 等 |
| `src/api_specs/memory_types.py` | `Memory`、`MemCell`、`MemoryType` |
| `src/api_specs/request_converter.py` | 外部格式转 `MemorizeRequest` |
| `src/infra_layer/.../mapper/group_chat_converter.py` | 简单消息 → 内部格式 |

### 3.2 Native 版涉及的 EverMemOS_bz 文件

**编排脚本（evaluation 目录）：**

| 文件 | 作用 |
|------|------|
| `evaluation/src/adapters/evermemos/stage1_memcells_extraction.py` | 消息→MemCell 提取 + 聚类的编排，内部调用生产 `ConvMemCellExtractor`、`ClusterManager`、`ProfileManager` |
| `evaluation/src/adapters/evermemos/stage2_index_building.py` | BM25 + Embedding 索引构建，本地 `.pkl` 持久化 |
| `evaluation/src/adapters/evermemos/stage3_memory_retrivel.py` | BM25 / Embedding / Hybrid / Agentic 检索，加载 `.pkl` 搜索 |
| `evaluation/src/adapters/evermemos/stage4_response.py` | LLM 答案生成，自实现 prompt |
| `evaluation/src/adapters/evermemos/config.py` | `ExperimentConfig` 配置类 |
| `evaluation/src/adapters/evermemos/tools/agentic_utils.py` | 简化版 agentic 检索工具 |

**被调用的生产模块（src 目录）：**

| 生产代码 | 被哪个 stage 使用 |
|---------|-----------------|
| `memory_layer/llm/llm_provider.py` | stage1, stage3 |
| `memory_layer/memcell_extractor/conv_memcell_extractor.py` | stage1 |
| `memory_layer/memory_extractor/episode_memory_extractor.py` | stage1 |
| `memory_layer/memory_extractor/event_log_extractor.py` | stage1 |
| `memory_layer/cluster_manager.py` | stage1（InMemory 模式） |
| `memory_layer/profile_manager.py` | stage1（InMemory 模式） |
| `agentic_layer/vectorize_service.py` | stage2, stage3 |
| `agentic_layer/rerank_service.py` | stage3 |
| `api_specs/memory_types.py` | stage1 |
| `common_utils/datetime_utils.py` | stage1 |

---

## 4. YAML 配置文件对比

**`evermemos.yaml`**（HTTP API 版）和 **`evermemos_native.yaml`**（Native 版）位于 `config/systems/` 目录。

### 4.1 核心字段对比

| 配置项 | `evermemos.yaml` | `evermemos_native.yaml` |
|--------|-----------------|------------------------|
| `name` | `"evermemos"` | `"evermemos_native"` |
| `adapter` | `"evermemos"` → `evermemos_adapter.py` | `"evermemos_native"` → `evermemos_native_adapter.py` |
| `builder` | `"evermemos"` → `evermemos_builder.py` | `"evermemos_native"` → `evermemos_native_builder.py` |

### 4.2 通信与启动配置

| 配置项 | `evermemos.yaml` | `evermemos_native.yaml` |
|--------|-----------------|------------------------|
| `api_url` | `"http://localhost:8001"` | ❌ 无（直接 import 不需要 HTTP） |
| `api_wait` | `180` 秒（等待 HTTP 服务就绪） | ❌ 无 |
| `env_file` | `".env"`（传给 FastAPI 进程） | ❌ 无 |
| `docker_compose` | `"systems/EverMemOS_bz/docker-compose.yaml"`（builder 内部拼接） | ❌ 已移除（无需 Docker） |
| `buffer_mode` | ❌ 无（实时处理） | ✅ `true`（缓冲模式） |
| `final_date` | ❌ 无 | ✅ `"2025-12-31"`（触发批量处理的日期，有自动检测 fallback） |

> **注意：** `final_date` 有自动检测 fallback。当数据集日期与 `final_date` 不匹配时，search 阶段会自动检测该 conversation 的最后消息日期作为触发点，或索引未构建时首次 search 强制触发构建。不需要为每个数据集手动配置 `final_date`。

### 4.3 LLM 配置

| 配置项 | `evermemos.yaml` | `evermemos_native.yaml` |
|--------|-----------------|------------------------|
| 模型 | `${LLM_MODEL}` | `${LLM_MODEL}` |
| API Key | `${LLM_API_KEY}` | `${LLM_API_KEY}` |
| Base URL | `${LLM_BASE_URL:https://openrouter.ai/api/v1}` | `${LLM_BASE_URL:https://openrouter.ai/api/v1}` |
| temperature | `0.3` | `0.3` |
| max_tokens | `32768` | `32768` |

### 4.4 Embedding / Rerank 配置

| 配置项 | `evermemos.yaml` | `evermemos_native.yaml` |
|--------|-----------------|------------------------|
| vectorize.provider | `${VECTORIZE_PROVIDER}` | `${VECTORIZE_PROVIDER:deepinfra}` |
| vectorize.model | `${VECTORIZE_MODEL}` | `${VECTORIZE_MODEL:Qwen/Qwen3-Embedding-4B}` |
| rerank.enabled | `true` | `true` |
| rerank.model | `${RERANK_MODEL}` | `${RERANK_MODEL:Qwen/Qwen3-Reranker-4B}` |

### 4.5 ADD 配置

| 配置项 | `evermemos.yaml` | `evermemos_native.yaml` |
|--------|-----------------|------------------------|
| `add.num_workers` | `15`（并发发送消息） | ❌ 无（native 在本地串行处理） |
| `add.enable_semantic_extraction` | `false` | `false` |
| `add.enable_clustering` | `true` | `true` |
| `add.enable_profile_extraction` | `false` | `false` |

### 4.6 SEARCH 配置

| 配置项 | `evermemos.yaml` | `evermemos_native.yaml` |
|--------|-----------------|------------------------|
| `search.mode` | `"agentic"` | `"agentic"` |
| `search.use_hybrid_search` | `true` | `true` |
| `search.use_reranker` | `true` | `true` |
| `search.hybrid_emb_candidates` | ❌ **无** | `50` |
| `search.hybrid_bm25_candidates` | ❌ **无** | `50` |
| `search.hybrid_rrf_k` | ❌ **无** | `40` |
| `search.lightweight_search_mode` | `"hybrid"` | `"bm25_only"`（默认值不同） |
| `search.num_workers` | `1`（查询并发数） | ❌ 无 |

**注意**：`hybrid_emb_candidates`、`hybrid_bm25_candidates`、`hybrid_rrf_k` 这些参数在 HTTP 版中不需要在 yaml 配置，因为检索在生产服务端（`memory_manager.py`）内部逻辑中控制；而在 native 版中，这些参数会传递给 `stage3` 的搜索函数用于控制本地检索行为。

### 4.7 ANSWER 配置

| 配置项 | `evermemos.yaml` | `evermemos_native.yaml` |
|--------|-----------------|------------------------|
| `answer.temperature` | `0` | `0` |
| `answer.response_top_k` | `30` | `30` |

### 4.8 附加配置

| 配置项 | `evermemos.yaml` | `evermemos_native.yaml` |
|--------|-----------------|------------------------|
| `stats.enabled` | `true`（统计 token 用量） | ❌ 无 |
| `stats.output_filename` | `token_stats.json` | ❌ 无 |
| `sample_parallel` | `true` | `true` |

### 4.9 配置差异总结

| 差异类型 | 说明 |
|----------|------|
| **通信方式** | HTTP 版需要 `api_url/api_wait` 指向 FastAPI 服务；native 版不需要 |
| **工作模式** | HTTP 版实时逐条处理；native 版有 `buffer_mode` + `final_date` 批量处理（含自动检测 fallback） |
| **配置位置** | native 版的 docker 路径直接在 yaml 中；HTTP 版由 builder 内部拼接 |
| **参数默认值** | native 版的 LLM/Embedding/Rerank 模型都有硬编码默认值，HTTP 版完全依赖环境变量 |
| **搜索参数** | native 版有 `hybrid_emb_candidates` 等细粒度参数（传递给 stage3），HTTP 版由服务端内部控制 |

---

## 5. Builder 实现对比

**`evermemos_builder.py`**（HTTP API 版）和 **`evermemos_native_builder.py`**（Native 版）位于 `src/builders/` 目录。

### 5.1 核心职责对比

| 职责 | `evermemos_builder.py` | `evermemos_native_builder.py` |
|------|----------------------|-----------------------------|
| 启动 Docker | ✅ `docker-compose up -d` | ❌ 无（已移除） |
| 启动 API 服务 | ✅ 启动 FastAPI 子进程 | ❌ 无（直接 import） |
| 服务就绪检测 | ✅ 轮询 HTTP `/docs` 端点 | ❌ 无 |
| venv 自动激活 | ❌ 无 | ✅ 检查并自动重启 |
| 清理 | ✅ stop API + stop Docker | ❌ 无操作 |

### 5.2 启动流程对比

**HTTP API 版 Builder：**

```
EverMemOSBuilder.build()
  │
  ├── 1. _start_docker()
  │     docker compose -f docker-compose.yaml up -d
  │     启动: MongoDB | ES | Milvus | Redis
  │     返回值区分"容器已存在"(True)和"刚启动"(False)
  │     → 刚启动时等待 120 秒（原 45 秒不够，容器约需 65-100 秒才全部健康）
  │
  ├── 2. 服务端编码修复
  │     设置 PYTHONUTF8=1 环境变量
  │     stdout/stderr → DEVNULL（避免 Windows GBK 编码下 emoji 崩溃）
  │
  ├── 3. _start_api_server()
  │     找到 EverMemOS_bz/.venv 的 Python
  │     检查 :8001/docs 是否已在运行
  │     如未运行: subprocess.Popen([venv_python, run.py, --port 8001])
  │     传递 env_file 指向 systems/EverMemOS_bz/.env
  │
  └── 4. _wait_for_api()
         轮询 http://localhost:8001/docs
         超时: api_wait = 180 秒
         成功后额外等 3 秒确保 lifespan 完成
```

**Native 版 Builder（已移除 Docker）：**

```
EverMemOSNativeBuilder.build()
  │
  ├── 1. _is_venv_activated() 检查
  │     │
  │     ├── 未激活:
  │     │     ├── 找 .venv/Scripts/python.exe
  │     │     ├── 如果有 → _restart_with_venv()
  │     │     │     subprocess.run([venv_python, cli.py, ...args])
  │     │     │     sys.exit(result.returncode)  ← 当前进程退出
  │     │     │
  │     │     └── 如果没有 → 打印激活说明，返回 False
  │     │
  │     └── 已激活 → 返回 True（无需 Docker）
  │
  └── [结束] Native 版不需要 Docker
```

### 5.3 关键差异详解

**差异 1：venv 自动激活（仅 native 版有）**

- HTTP 版的 FastAPI 运行在子进程中，通过 `subprocess.Popen([venv_python, run.py])` 直接指定了 venv Python，不依赖调用者的环境
- Native 版直接在调用进程内 `import` 生产模块，必须确保调用者本身就在 `.venv` 中。如果不在，它会：
  1. 用 `.venv` 的 Python 重新启动整个评测脚本（含 `cli.py`）
  2. 原进程 `sys.exit()`
  3. 新进程的 builder 检测到 `.venv` 已激活，继续执行

> **修复记录：** `_restart_with_venv()` 原代码 `subprocess.run([venv_python] + cmd_args[1:])` 遗漏了 `cmd_args[0]`（即 `cli.py`），导致执行 `.venv\python.exe --dataset smoke ...` 报 `unknown option --dataset`。

**差异 2：API 服务管理（仅 HTTP 版有）**

HTTP 版 builder 管理子进程生命周期（`terminate()` / `kill()` / `wait()`），启动前先检查端口是否已被占用，需要将 `.env` 路径传给 FastAPI 进程。Native 版无此逻辑。

**差异 3：就绪检测方式**

| 维度 | HTTP 版 | Native 版 |
|------|---------|-----------|
| 检测对象 | HTTP 端点 `/docs` | ❌ 无（不需要，无 Docker 无 API） |
| 检测工具 | `aiohttp.ClientSession` | ❌ 无 |
| 超时 | `api_wait = 180` 秒 | ❌ 无 |
| 等待策略 | 循环 GET 请求，成功后额外等 3 秒 | ❌ 无 |

**差异 4：配置读取方式**

| 配置项 | HTTP 版 | Native 版 |
|--------|---------|-----------|
| `docker_compose` | `config.get("docker_compose", ...)` | ❌ 已移除 |
| `api_url` | `config.get("api_url", "http://localhost:8001")` | ❌ 无 |
| `api_wait` | `config.get("api_wait", 60)` | ❌ 无 |
| `env_file` | `config.get("env_file", ".env")` | ❌ 无 |

### 5.4 清理流程对比

**HTTP 版：**
```
cleanup()
  ├── _stop_api_server()
  │     terminate() → wait(10s) → kill() → wait()
  └── _stop_docker()
        docker compose down
```

**Native 版：**
```
cleanup()
  └── 无操作（无需停止 Docker）
```

### 5.5 Builder 差异总结

| 维度 | `evermemos_builder.py` | `evermemos_native_builder.py` |
|------|----------------------|-----------------------------|
| **启动内容** | Docker + FastAPI 子进程 | 仅 venv 检查（无需 Docker 或 API） |
| **venv 策略** | 子进程自动使用 venv Python | 必须已在 venv 中，否则自动重启 |
| **服务就绪** | HTTP 端点健康检查（180s 超时） | ❌ 无（无 Docker 无 API） |
| **进程管理** | 有（Popen + terminate/kill） | 无 |
| **aiohttp 依赖** | ✅ 需要（HTTP 健康检查） | ❌ 不需要 |
| **额外等待** | 45s Docker + 180s API = ~225s | 约 1s（仅 venv 检查） |
| **清理** | stop API → stop Docker | 无操作 |
| **env 传递** | `.env` 文件路径传给 FastAPI | ❌ 不需要 |
| **适用场景** | 正式评测（需要完整系统） | 调试/开发（零基础设施依赖） |

---

## 6. 存储架构

### 6.1 HTTP API 版

使用 4 个 Docker 数据服务 + 1 个外部嵌入服务：

| 存储 | 用途 | HTTP API 版 | Native 版 |
|------|------|------------|-----------|
| **MongoDB** | 主存储：MemCell、Episode、Semantic、EventLog、Profile、Meta | ✅ | ❌（InMemory 替代） |
| **Elasticsearch** | BM25 全文搜索 | ✅ | ❌（本地 `.pkl` 替代） |
| **Milvus** | 向量相似度搜索 | ✅ | ❌（本地 `.pkl` 替代） |
| **Redis** | 消息缓存、分布式锁 | ✅ | ❌ |
| 外部 Embedding API | 文本→向量 | ✅ | ✅（同一服务） |
| 外部 Rerank API | 检索结果重排序 | ✅ | ✅（同一服务） |
| 本地 `.pkl` 文件 | BM25/Embedding 索引 | ❌ | ✅ |

### 6.2 HTTP API 版的数据存储映射

| 数据类型 | MongoDB 集合 | ES 索引 | Milvus 集合 |
|----------|-------------|---------|-------------|
| MemCell | `memcells` | — | — |
| EpisodicMemory | `episodic_memories` | `episodic_memories` | `episodic_memory` |
| SemanticMemoryRecord | `semantic_memory_records` | `semantic_memory_records` | `semantic_memory_record` |
| EventLogRecord | `event_log_records` | `event_log_records` | `event_log_record` |
| ConversationMeta | `conversation_meta` | — | — |
| UserProfile | `user_profiles` | — | — |
| Cluster | `clusters` | — | — |
| 对话缓存 | — | — | (Redis) |

### 6.3 Native 版的数据存储映射

```
output/{dataset}-{system}/
├── memcells/
│   ├── conv_0_memcells.json
│   └── conv_1_memcells.json ...
├── bm25_index/
│   ├── bm25_index_conv_0.pkl    (rank_bm25 模型 + 文档)
│   └── bm25_index_conv_1.pkl ...
├── vectors/
│   ├── embedding_index_conv_0.pkl (numpy 向量矩阵)
│   └── embedding_index_conv_1.pkl ...
├── results/
│   └── {question_id}.json          (单个搜索结果，由 _save_search_result 写入)
├── search_results.json             (增量合并的搜索结果)
├── search_latency.json             (搜索延迟)
├── answer_results.json             (回答结果)
├── eval_results.json               (评测结果)
├── add_latency.json                (添加延迟)
├── checkpoint_default.json         (断点续跑状态)
└── report.txt                      (摘要报告)
```

### 6.4 Docker 使用对照

| Docker 服务 | HTTP API 版 | Native 版 |
|-------------|------------|-----------|
| MongoDB `memsys-mongodb` | ✅ builder 启动 + adapter 使用 | ❌ **不需要**（已移除） |
| ES `memsys-elasticsearch` | ✅ builder 启动 + adapter 使用 | ❌ **不需要**（已移除） |
| Milvus `memsys-milvus-standalone` | ✅ builder 启动 + adapter 使用 | ❌ **不需要**（已移除） |
| Redis `memsys-redis` | ✅ builder 启动 + adapter 使用 | ❌ **不需要**（已移除） |

**结论：Native 版完全不需要 Docker。** adapter 的内存提取用 InMemory 聚类、索引用本地 `.pkl`、检索从本地文件读取，无需任何容器。Builder 已移除所有 Docker 相关代码。

---

## 7. 边界检测机制

**仅 HTTP API 版有。** Native 版无此机制。

当 adapter 调用 `POST /memorize` 时，后端执行：

```
preprocess_conv_request()
  → 从 Redis 读取该 group_id 的历史消息
  → 拼接到 history_raw_data_list

extract_memcell(history + new messages)
  → LLM 判断这批消息是否构成完整的"事件"
  │
  ├── 非边界 (memcell is None):
  │     新消息追加到 Redis 缓存
  │     更新状态表为"继续累积"
  │     HTTP 返回: {count: 0, status_info: "accumulated"}
  │
  └── 是边界 (memcell 提取成功):
        清空 Redis 旧历史
        将新消息保存到 Redis（作为下次的起点）
        保存 MemCell → MongoDB
        触发聚类
        提取 Episode/Semantic/EventLog → MongoDB + ES + Milvus
        更新状态表
        HTTP 返回: {count: N, status_info: "extracted"}
```

这就是 adapter 日志中 "accumulated" 和 "extracted" 的含义：
- **accumulated**：消息还在 Redis 排队，未到达边界阈值
- **extracted**：触发了边界，已提取为结构化记忆

**Native 版无此机制**，所有消息一次传入直接提取，不存在"排队等待边界"的过程。

---

## 8. Adapter 方法对照分析

### 8.1 ADD 相关方法

| 方法 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| `add_chunks()` | 按日期前缀分组（所有同日消息合并为一条）→ GBK 安全过滤（`_sanitize_content`）→ POST `/memorize`。后端用 Redis 累积 + 边界检测跨日时间边界触发提取 | 缓冲到内存 → 记录 `_conv_latest_dates` → final_date 触发或 search 自动触发批量 `stage1` 提取 + `stage2` 建索引 |

### 8.2 SEARCH 相关方法

| 方法 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| `search()` lightweight | POST `/retrieve_lightweight` → ES BM25 + Milvus Vector + RRF | 加载 `.pkl` → 本地 BM25 / Embedding 搜索 |
| `search()` agentic | POST `/retrieve_agentic` → Round1 RRF → Rerank → LLM 判断 → Round2 多查询 → Final Rerank | 加载 `.pkl` → 简化 agentic（无多轮/Rerank） |
| 缓冲机制 | 无（每次调用立即搜索） | 有：非 final_date 的搜索先缓冲，到达最终日期时批量执行。支持自动检测（ask_time 匹配 / 首次搜索强制触发 / 索引已构建则直接执行） |

### 8.3 ANSWER 方法

| 方法 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| `answer()` | 自实现，类似 Native 版的 CoT prompt（`evaluation/.../answer_prompts.py`） | `stage4_response.locomo_response()`= |

### 8.4 辅助方法

| 方法 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| 消息拼接 | 同 dia_id 拼接为 "speaker: xxx\ntext: xxx\n---\nspeaker: yyy\ntext: yyy" | 逐条转为 dict，追加到缓冲区列表 |
| 会话映射 | 直接用字符串 conversation_id | 需要映射为数字索引 `conv_0`, `conv_1` |

---

## 9. 全维度对比总结

### 9.1 实现方式

| 维度 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| 代码架构 | HTTP 客户端 → FastAPI 服务 | 直接 import Python 模块 |
| 调用链路 | cli.py → httpx → FastAPI → DI 容器 → Biz → Repository → DB | cli.py → evaluation 编排脚本 → 生产提取组件 → 本地文件 |
| 环境需求 | 需要 Docker + .venv + 网络 | 需要 .venv + Embedding/Rerank API 网络 |

### 9.2 消息处理

| 维度 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| 消息输入方式 | 逐条 HTTP POST 发送 | 缓冲到内存，最后一次性处理 |
| 消息累积 | Redis（有持久化） | 内存 dict（无持久化） |
| 边界检测 | ✅ LLM 判断，消息排队等待 | ❌ 无，全量传入 |
| 消息拼接 | 同 dia_id 拼接为一段文本 | 逐条转为结构化 dict |

### 9.3 记忆提取

| 维度 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| MemCell 提取 | 生产 `MemoryManager.extract_memcell()` | stage1 → 生产 `ConvMemCellExtractor` |
| Episode 提取 | 生产 `EpisodeMemoryExtractor` | stage1 → 生产 `EpisodeMemoryExtractor` ✅ 相同 |
| Semantic 提取 | 生产代码 | stage1 → 生产代码 ✅ 相同 |
| EventLog 提取 | 生产 `EventLogExtractor` | stage1 → 生产 `EventLogExtractor` ✅ 相同 |
| 聚类 | 生产 `ClusterManager`（MongoDB 持久化） | stage1 → 生产 `ClusterManager`（**InMemory**） |
| Profile 提取 | 生产 `ProfileManager`（MongoDB 持久化） | stage1 → 生产 `ProfileManager`（**InMemory**） |

### 9.4 索引构建

| 维度 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| 触发时机 | 每次边界检测后自动同步 | final_date 时批量构建，或 search 自动检测触发 |
| BM25 索引 | ES 自动索引（`episodic_memory_es_repository`） | stage2 → rank_bm25 → 本地 `.pkl` |
| Embedding 索引 | Milvus 自动索引（`episodic_memory_milvus_repository`） | stage2 → 生产 `vectorize_service` → 本地 `.pkl` |
| 三重同步 | MongoDB + ES + Milvus 自动同步 | 无 |

### 9.5 检索

| 维度 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| BM25 检索 | ES multi_search（jieba + 停用词） | 本地 `rank_bm25` |
| Embedding 检索 | Milvus vector_search（COSINE） | 本地 numpy 余弦相似度 |
| RRF 融合 | 生产 `retrieval_utils.reciprocal_rank_fusion()` | stage3 自实现简化版 |
| Rerank | 生产 `rerank_service`（完整） | stage3 → 生产 `rerank_service` ✅ 相同 |
| Agentic 检索 | 完整：Round1 → Rerank → LLM判断 → Round2多查询 → Final Rerank | 简化版：无多轮、无充分性判断 |

### 9.6 基础设施依赖

| 依赖 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| Docker (MongoDB) | ✅ 必需 | ❌ 不需要（已移除） |
| Docker (ES) | ✅ 必需 | ❌ 不需要（已移除） |
| Docker (Milvus) | ✅ 必需 | ❌ 不需要（已移除） |
| Docker (Redis) | ✅ 必需 | ❌ 不需要（已移除） |
| 外部 Embedding API | ✅ 必需 | ✅ 必需 |
| 外部 Rerank API | 可选 | 可选 |
| 外部 LLM API | ✅ 必需 | ✅ 必需 |
| `.venv` (EverMemOS_bz) | ✅ 必需（运行 FastAPI 服务） | ✅ 必需（import 生产代码） |
| venv 自动激活 | ❌ 无 | ✅ 有（`_restart_with_venv`） |

### 9.7 实验行为差异

| 行为 | HTTP API 版 | Native 版 |
|------|------------|-----------|
| 搜索请求时序 | 随 ADD 过程交叉进行（当日 ADD + 当日 SEARCH） | 非 final_date 的搜索被缓冲，到达最终日期才执行 |
| 搜索结果一致性 | 实时反映当前已提取的记忆 | 一次性获取全部记忆（无时间渐进） |
| 断点续跑 | 支持（checkpoint 记录已处理的位置） | 支持但因缓冲模式行为不同 |
| 运行时日志 | adapter 日志 + 服务端日志 | adapter 日志 + stage 日志 |
| 错误隔离 | HTTP 超时/重试机制 | Python 异常直接传播 |

### 9.8 实际评测结果（locomo 数据集）

#### Native 版

| 指标| 修复后 |
|------|--------|
| **Accuracy** | **84.55%** |
| 正确数| **197/233** |
| 总时间| ~5243s（87min） |
| Add+Search  | 4623s（正常处理） |

#### HTTP API 版（Server 版）

| Prompt 版本 | Accuracy | 正确数 |  时间 | 
|-------------|----------|--------|-------------|
| **CoT prompt** | **80.69%** | **188/233** | 5197s | 

存在 context 格式差异：server 版用 agentic search 返回的 episode 摘要，native 版用结构化 speaker 模板

---

## 10. 已知问题

### 10.1 Answer 端上下文格式差异
- Server 版使用 agentic search 返回的 episode 摘要作为 answer context
- Native 版使用结构化模板（`TEMPLATE` with `speaker_1`/`speaker_2`）

### 10.2 LLM 调用耗时过长
- `deepseek-v4-flash` 通过 DashScope API 调用单次 30-50s
- `openai_provider.py:180` 阈值已从 30s 调整为 60s
- 根本原因是 API 本身延迟

### 10.3 Agentic 检索简化
- Native 版使用简化版 agentic 检索（`stage3_memory_retrivel.py`），缺少多轮查询和充分性判断
- HTTP API 版的生产 `memory_manager.py` 有完整的 Round1→Rerank→LLM判断→Round2 流程
