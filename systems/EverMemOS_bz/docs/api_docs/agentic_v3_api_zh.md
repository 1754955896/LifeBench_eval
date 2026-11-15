# Agentic API 文档

## 概述

Agentic API 是 MemSys 的智能记忆系统接口，提供记忆存储和智能检索功能。该 API 支持简单直接的消息格式输入，以及两种不同复杂度的检索模式：轻量级检索和 Agentic 智能检索。

## 主要特性

- ✅ **简单存储**：采用最简单的单条消息格式，无需复杂的数据结构
- ✅ **智能检索**：支持轻量级和 Agentic 两种检索模式
- ✅ **多源融合**：支持向量检索、BM25 关键词检索和 RRF 融合
- ✅ **LLM 增强**：Agentic 模式使用 LLM 进行多轮智能检索
- ✅ **灵活过滤**：支持按用户、群组、时间范围等多维度过滤
- ✅ **元数据管理**：支持保存和管理对话元数据

## 接口说明

### POST `/api/v3/agentic/memorize`

存储单条群聊消息记忆

#### 请求格式

**Content-Type**: `application/json`

**请求体**：简单直接的单条消息格式（无需预转换）

```json
{
  "group_id": "group_123",
  "group_name": "项目讨论组",
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+08:00",
  "sender": "user_001",
  "sender_name": "张三",
  "content": "今天讨论下新功能的技术方案",
  "refer_list": ["msg_000"]
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| group_id | string | 否 | 群组ID |
| group_name | string | 否 | 群组名称 |
| message_id | string | 是 | 消息唯一标识 |
| create_time | string | 是 | 消息创建时间（ISO 8601格式） |
| sender | string | 是 | 发送者用户ID |
| sender_name | string | 否 | 发送者名称（默认使用 sender） |
| content | string | 是 | 消息内容 |
| refer_list | array | 否 | 引用的消息ID列表 |

#### 响应格式

**成功响应 (200 OK)**

```json
{
  "status": "ok",
  "message": "Extracted 1 memories",
  "result": {
    "saved_memories": [
      {
        "memory_type": "episode_summary",
        "user_id": "user_001",
        "group_id": "group_123",
        "timestamp": "2025-01-15T10:00:00",
        "content": "用户讨论了新功能的技术方案"
      }
    ],
    "count": 1,
    "status_info": "extracted"
  }
}
```

**注意**：如果消息被累积但尚未提取为记忆（等待边界检测），则返回：

```json
{
  "status": "ok",
  "message": "Message queued, awaiting boundary detection",
  "result": {
    "saved_memories": [],
    "count": 0,
    "status_info": "accumulated"
  }
}
```

**错误响应 (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "数据格式错误：缺少必需字段 message_id",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/memorize"
}
```

**错误响应 (500 Internal Server Error)**

```json
{
  "status": "failed",
  "code": "SYSTEM_ERROR",
  "message": "存储记忆失败，请稍后重试",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/memorize"
}
```

---

### POST `/api/v3/agentic/retrieve_lightweight`

轻量级记忆检索（Embedding + BM25 + RRF 融合）

#### 功能说明

- 并行执行向量检索和关键词检索
- 使用 RRF（Reciprocal Rank Fusion）融合结果
- 速度快，适合实时场景
- 支持多种数据源和检索模式

#### 请求格式

**Content-Type**: `application/json`

**请求体**：

```json
{
  "query": "北京旅游美食",
  "user_id": "default",
  "group_id": "assistant",
  "time_range_days": 365,
  "top_k": 20,
  "retrieval_mode": "rrf",
  "data_source": "episode",
  "current_time": "2025-01-15",
  "radius": 0.6
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| query | string | 条件必需 | - | 用户查询（data_source=profile 时可为空） |
| user_id | string | 否 | - | 用户ID（用于过滤） |
| group_id | string | 否 | - | 群组ID（用于过滤） |
| time_range_days | integer | 否 | 365 | 时间范围天数 |
| top_k | integer | 否 | 20 | 返回结果数量 |
| retrieval_mode | string | 否 | rrf | 检索模式：rrf/embedding/bm25 |
| data_source | string | 否 | episode | 数据源：episode/event_log/semantic_memory/profile |
| current_time | string | 否 | - | 当前时间（YYYY-MM-DD格式，用于语义记忆有效期过滤） |
| radius | float | 否 | 0.6 | COSINE 相似度阈值，范围 [-1, 1] |

**检索模式说明**：

- `rrf`: RRF 融合（默认，推荐）- 结合向量检索和关键词检索的优点
- `embedding`: 纯向量检索 - 适合语义相似性搜索
- `bm25`: 纯关键词检索 - 适合精确词匹配

**数据源说明**：

- `episode`: 从 MemCell.episode 检索（默认）- 情景记忆
- `event_log`: 从 event_log.atomic_fact 检索 - 原子事实
- `semantic_memory`: 从语义记忆检索 - 抽象化的长期记忆
- `profile`: 档案检索（仅需 user_id + group_id，query 可空）

**记忆范围说明**：

- `all`: 所有记忆（默认）- 同时使用 user_id 和 group_id 参数过滤
- `personal`: 仅个人记忆 - 只使用 user_id 参数过滤，不使用 group_id
- `group`: 仅群组记忆 - 只使用 group_id 参数过滤，不使用 user_id

**相似度阈值说明**：

- `radius` 参数控制 COSINE 相似度的最低阈值
- 范围为 [-1, 1]，默认 0.6
- 只返回相似度 >= radius 的结果
- 影响向量检索部分（embedding/rrf 模式）的结果质量
- 对语义记忆和情景记忆有效（semantic_memory/episode）
- 事件日志使用 L2 距离暂不支持

#### 响应格式

**成功响应 (200 OK)**

```json
{
  "status": "ok",
  "message": "检索成功，找到 10 条记忆",
  "result": {
    "memories": [
      {
        "content": "用户喜欢吃北京烤鸭",
        "score": 0.85,
        "timestamp": "2025-01-10T15:30:00",
        "user_id": "default",
        "group_id": "assistant"
      }
    ],
    "count": 10,
    "metadata": {
      "retrieval_mode": "lightweight",
      "emb_count": 15,
      "bm25_count": 12,
      "final_count": 10,
      "total_latency_ms": 123.45
    }
  }
}
```

**错误响应 (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "缺少必需参数：query",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/retrieve_lightweight"
}
```

---

### POST `/api/v3/agentic/retrieve_agentic`

Agentic 记忆检索（LLM 引导的多轮智能检索）

#### 功能说明

- 使用 LLM 判断检索充分性
- 自动进行多轮检索和查询改进
- 使用 Rerank 提升结果质量
- 适合需要深度理解的复杂查询

#### 检索流程

1. **Round 1**: RRF 混合检索（Embedding + BM25）
2. **Rerank 优化**: 使用重排序模型优化结果
3. **LLM 判断**: 判断检索结果是否充分
4. **如果不充分**: 生成多个改进查询
5. **Round 2**: 多查询并行检索
6. **融合返回**: 融合结果并 Rerank 返回最终结果

#### 请求格式

**Content-Type**: `application/json`

**请求体**：

```json
{
  "query": "用户喜欢吃什么？",
  "user_id": "default",
  "group_id": "assistant",
  "time_range_days": 365,
  "top_k": 20,
  "llm_config": {
    "api_key": "your_api_key",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "qwen/qwen3-235b-a22b-2507"
  }
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| query | string | 是 | - | 用户查询 |
| user_id | string | 否 | - | 用户ID（用于过滤） |
| group_id | string | 否 | - | 群组ID（用于过滤） |
| time_range_days | integer | 否 | 365 | 时间范围天数 |
| top_k | integer | 否 | 20 | 返回结果数量 |
| llm_config | object | 否 | - | LLM 配置 |

**llm_config 字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| api_key | string | 否 | LLM API Key（未提供时使用环境变量 OPENROUTER_API_KEY/OPENAI_API_KEY） |
| base_url | string | 否 | LLM API 地址（默认 https://openrouter.ai/api/v1） |
| model | string | 否 | LLM 模型（默认 qwen/qwen3-235b-a22b-2507） |

#### 响应格式

**成功响应 (200 OK)**

```json
{
  "status": "ok",
  "message": "Agentic 检索成功，找到 15 条记忆",
  "result": {
    "memories": [
      {
        "content": "用户喜欢吃川菜，尤其是麻辣火锅",
        "score": 0.92,
        "timestamp": "2025-01-10T15:30:00",
        "user_id": "default",
        "group_id": "assistant"
      }
    ],
    "count": 15,
    "metadata": {
      "retrieval_mode": "agentic",
      "is_multi_round": true,
      "round1_count": 20,
      "is_sufficient": false,
      "reasoning": "需要更多关于饮食偏好的具体信息",
      "refined_queries": ["用户最喜欢的菜系？", "用户不喜欢吃什么？"],
      "round2_count": 40,
      "final_count": 15,
      "total_latency_ms": 2345.67
    }
  }
}
```

**错误响应 (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "缺少必需参数：query",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/retrieve_agentic"
}
```

**错误响应 (500 Internal Server Error)**

```json
{
  "status": "failed",
  "code": "SYSTEM_ERROR",
  "message": "Agentic 检索失败，请稍后重试",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/retrieve_agentic"
}
```

---

### POST `/api/v3/agentic/conversation-meta`

保存对话元数据

#### 功能说明

保存对话的元数据信息，包括场景、参与者、标签等。使用 upsert 行为，如果 `group_id` 已存在则更新整个记录。

#### 请求格式

**Content-Type**: `application/json`

**请求体**：

```json
{
  "version": "1.0",
  "scene": "group_chat",
  "scene_desc": "项目团队讨论",
  "name": "项目讨论组",
  "description": "新功能开发的技术讨论",
  "group_id": "group_123",
  "created_at": "2025-01-15T10:00:00+08:00",
  "default_timezone": "Asia/Shanghai",
  "user_details": {
    "user_001": {
      "full_name": "张三",
      "role": "developer",
      "extra": {"department": "工程部"}
    },
    "user_002": {
      "full_name": "李四",
      "role": "designer",
      "extra": {"department": "设计部"}
    }
  },
  "tags": ["工作", "技术"]
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| version | string | 是 | 元数据版本 |
| scene | string | 是 | 场景标识（如 "group_chat"） |
| scene_desc | string | 是 | 场景描述 |
| name | string | 是 | 对话名称 |
| description | string | 是 | 对话描述 |
| group_id | string | 是 | 群组唯一标识 |
| created_at | string | 是 | 对话创建时间（ISO 8601 格式） |
| default_timezone | string | 是 | 默认时区 |
| user_details | object | 是 | 参与者详情 |
| tags | array | 否 | 标签列表 |

#### 响应格式

**成功响应 (200 OK)**

```json
{
  "status": "ok",
  "message": "对话元数据保存成功",
  "result": {
    "id": "507f1f77bcf86cd799439011",
    "group_id": "group_123",
    "scene": "group_chat",
    "name": "项目讨论组",
    "version": "1.0",
    "created_at": "2025-01-15T10:00:00+08:00",
    "updated_at": "2025-01-15T10:00:00+08:00"
  }
}
```

**错误响应 (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "缺少必需字段: version",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/conversation-meta"
}
```

---

## 使用场景

### 1. 实时消息流处理 + 智能问答

适用于处理来自聊天应用的实时消息流，每收到一条消息就存储，然后根据用户查询智能检索相关记忆。

**存储示例**：

```bash
curl -X POST http://localhost:1995/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "group_name": "项目讨论组",
    "message_id": "msg_001",
    "create_time": "2025-01-15T10:00:00+08:00",
    "sender": "user_001",
    "sender_name": "张三",
    "content": "我们项目下周要发布新功能",
    "refer_list": []
  }'
```

**轻量级检索示例**：

```bash
curl -X POST http://localhost:1995/api/v3/agentic/retrieve_lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "query": "项目发布时间",
    "group_id": "group_123",
    "top_k": 10
  }'
```

### 2. 智能客服系统

使用 Agentic 检索模式，根据用户问题自动改进查询，提供更准确的答案。

**示例**：

```python
import httpx
import asyncio

async def intelligent_customer_service(user_query: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/retrieve_agentic",
            json={
                "query": user_query,
                "user_id": "customer_001",
                "group_id": "support",
                "top_k": 20,
                "llm_config": {
                    "model": "qwen/qwen3-235b-a22b-2507"
                }
            }
        )
        result = response.json()
        return result["result"]["memories"]

# 使用示例
memories = asyncio.run(intelligent_customer_service("如何退款？"))
```

### 3. 个人知识库管理

存储个人笔记和想法，使用轻量级检索快速查找相关内容。

**示例**：

```python
import httpx
import asyncio

async def save_note(content: str, user_id: str):
    """保存笔记"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/memorize",
            json={
                "message_id": f"note_{int(time.time())}",
                "create_time": datetime.now().isoformat(),
                "sender": user_id,
                "content": content,
                "group_id": f"personal_{user_id}"
            }
        )
        return response.json()

async def search_notes(query: str, user_id: str):
    """搜索笔记"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/retrieve_lightweight",
            json={
                "query": query,
                "user_id": user_id,
                "group_id": f"personal_{user_id}",
                "retrieval_mode": "rrf",
                "top_k": 20
            }
        )
        return response.json()
```

---

## 使用示例

### 使用 curl 调用

#### 存储记忆

```bash
curl -X POST http://localhost:1995/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "group_name": "项目讨论组",
    "message_id": "msg_001",
    "create_time": "2025-01-15T10:00:00+08:00",
    "sender": "user_001",
    "sender_name": "张三",
    "content": "今天讨论下新功能的技术方案",
    "refer_list": []
  }'
```

#### 轻量级检索

```bash
curl -X POST http://localhost:1995/api/v3/agentic/retrieve_lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "query": "技术方案",
    "group_id": "group_123",
    "top_k": 10,
    "retrieval_mode": "rrf"
  }'
```

#### Agentic 智能检索

```bash
curl -X POST http://localhost:1995/api/v3/agentic/retrieve_agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "我们讨论过哪些技术问题？",
    "group_id": "group_123",
    "top_k": 20,
    "llm_config": {
      "model": "qwen/qwen3-235b-a22b-2507"
    }
  }'
```

### 使用 Python 代码调用

#### 存储记忆

```python
import httpx
import asyncio
from datetime import datetime

async def memorize_message():
    message_data = {
        "group_id": "group_123",
        "group_name": "项目讨论组",
        "message_id": "msg_001",
        "create_time": datetime.now().isoformat(),
        "sender": "user_001",
        "sender_name": "张三",
        "content": "今天讨论下新功能的技术方案",
        "refer_list": []
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/memorize",
            json=message_data
        )
        result = response.json()
        print(f"保存结果: {result['message']}")
        print(f"记忆数量: {result['result']['count']}")

asyncio.run(memorize_message())
```

#### 轻量级检索

```python
import httpx
import asyncio

async def lightweight_retrieve():
    query_data = {
        "query": "技术方案",
        "group_id": "group_123",
        "top_k": 10,
        "retrieval_mode": "rrf"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/retrieve_lightweight",
            json=query_data
        )
        result = response.json()
        print(f"找到 {result['result']['count']} 条记忆")
        for memory in result['result']['memories']:
            print(f"- {memory['content']} (score: {memory.get('score', 'N/A')})")

asyncio.run(lightweight_retrieve())
```

#### Agentic 智能检索

```python
import httpx
import asyncio

async def agentic_retrieve():
    query_data = {
        "query": "我们讨论过哪些技术问题？",
        "group_id": "group_123",
        "top_k": 20,
        "llm_config": {
            "model": "qwen/qwen3-235b-a22b-2507"
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/retrieve_agentic",
            json=query_data,
            timeout=30.0  # Agentic 检索可能需要更长时间
        )
        result = response.json()
        
        print(f"找到 {result['result']['count']} 条记忆")
        metadata = result['result']['metadata']
        print(f"检索模式: {metadata['retrieval_mode']}")
        print(f"是否多轮: {metadata.get('is_multi_round', False)}")
        print(f"总耗时: {metadata['total_latency_ms']:.2f}ms")
        
        for memory in result['result']['memories']:
            print(f"- {memory['content']} (score: {memory.get('score', 'N/A')})")

asyncio.run(agentic_retrieve())
```

---

## 常见问题

### 1. 轻量级检索和 Agentic 检索有什么区别？

**轻量级检索**：
- 速度快（通常 100-500ms）
- 适合实时场景
- 使用固定的检索策略
- 无 LLM 调用成本

**Agentic 检索**：
- 速度较慢（通常 2-5s）
- 适合复杂查询
- 使用 LLM 智能改进查询
- 有 LLM API 调用成本

**建议**：
- 实时对话、快速搜索 → 使用轻量级检索
- 复杂问题、深度挖掘 → 使用 Agentic 检索

### 2. 如何选择检索模式（rrf/embedding/bm25）？

- `rrf`（推荐）：结合向量和关键词的优点，适合大多数场景
- `embedding`：更注重语义相似性，适合概念性查询
- `bm25`：更注重精确词匹配，适合查找特定关键词

### 3. data_source 参数如何选择？

- `episode`（默认）：适合检索具体的对话内容和情景
- `event_log`：适合检索原子级别的事实信息
- `semantic_memory`：适合检索抽象的长期记忆
- `profile`：适合获取用户或群组的档案信息


### 4. radius 参数如何调整？

`radius` 是 COSINE 相似度阈值，范围 [-1, 1]：

- **0.8-1.0**：非常严格，只返回高度相关的结果，可能结果较少
- **0.6-0.8**（推荐）：平衡准确性和召回率
- **0.4-0.6**：较宽松，返回更多结果，但相关性可能降低
- **< 0.4**：非常宽松，可能包含大量不相关结果

**建议**：
- 精确搜索：使用 0.7-0.8
- 常规搜索：使用 0.6（默认）
- 广泛搜索：使用 0.5

### 5. 如何配置 LLM for Agentic 检索？

有两种方式配置 LLM：

**方式一：通过环境变量**

```bash
export OPENROUTER_API_KEY="your_api_key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="qwen/qwen3-235b-a22b-2507"
```

**方式二：通过请求参数**

```json
{
  "query": "用户喜欢什么？",
  "llm_config": {
    "api_key": "your_api_key",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "qwen/qwen3-235b-a22b-2507"
  }
}
```

### 6. 如何处理消息时间？

`create_time` 必须使用 ISO 8601 格式，支持带时区：

```json
{
  "create_time": "2025-01-15T10:00:00+08:00"  // 带时区
}
```

或不带时区（默认使用 UTC）：

```json
{
  "create_time": "2025-01-15T10:00:00"  // UTC
}
```

### 7. 检索时如何过滤时间范围？

使用 `time_range_days` 参数指定最近几天的记忆：

```json
{
  "query": "技术讨论",
  "time_range_days": 7  // 只检索最近 7 天的记忆
}
```

### 8. 如何处理 profile 检索？

档案检索（data_source=profile）不需要 query 参数，只需提供 user_id 和 group_id：

```json
{
  "user_id": "user_001",
  "group_id": "group_123",
  "data_source": "profile"
}
```

### 9. 接口调用频率有限制吗？

目前没有硬性限制，但建议：
- **存储接口**：每秒不超过 100 次请求
- **轻量级检索**：每秒不超过 50 次请求
- **Agentic 检索**：每秒不超过 10 次请求（受 LLM API 限制）

---

## 架构说明

### 数据流

#### 存储流程

```
客户端
  ↓
  │ 简单直接的单条消息格式
  ↓
Agentic V3 Controller (agentic_v3_controller.py)
  ↓
  │ 调用 group_chat_converter.py
  ↓
格式转换 (convert_simple_message_to_memorize_input)
  ↓
  │ 内部格式
  ↓
Memory Manager (memory_manager.py)
  ↓
  │ 记忆提取和存储
  ↓
数据库 / 向量库
```

#### 检索流程（轻量级）

```
客户端
  ↓
  │ 检索请求
  ↓
Agentic V3 Controller
  ↓
Memory Manager (retrieve_lightweight)
  ↓
  ├─→ Embedding 检索 ──┐
  │                     ├─→ RRF 融合
  └─→ BM25 检索 ────────┘
  ↓
返回结果
```

#### 检索流程（Agentic）

```
客户端
  ↓
  │ 检索请求
  ↓
Agentic V3 Controller
  ↓
Memory Manager (retrieve_agentic)
  ↓
Round 1: RRF 混合检索
  ↓
Rerank 优化
  ↓
LLM 判断充分性
  ↓
  ├─→ 充分 → 返回结果
  │
  └─→ 不充分
       ↓
     生成改进查询
       ↓
     Round 2: 多查询并行检索
       ↓
     融合 + Rerank
       ↓
     返回结果
```

### 核心组件

1. **Agentic V3 Controller** (`agentic_v3_controller.py`)
   - 接收 HTTP 请求
   - 参数验证和转换
   - 调用 Memory Manager
   - 统一响应格式

2. **Memory Manager** (`memory_manager.py`)
   - 记忆提取和存储
   - 轻量级检索（RRF 融合）
   - Agentic 智能检索
   - 向量化和持久化

3. **Group Chat Converter** (`group_chat_converter.py`)
   - 格式适配层
   - 消息格式转换
   - 保持单一职责

4. **Conversation Meta Repository** (`conversation_meta_raw_repository.py`)
   - 对话元数据存储
   - MongoDB ODM 操作
   - Upsert 支持

---

## 性能优化建议

### 1. 批量存储优化

对于批量消息存储，建议：
- 控制并发数（建议 10-20 并发）
- 添加适当延迟（每条消息间隔 50-100ms）
- 使用异步客户端

```python
import httpx
import asyncio

async def batch_memorize(messages: list):
    async with httpx.AsyncClient() as client:
        tasks = []
        for msg in messages:
            task = client.post(
                "http://localhost:1995/api/v3/agentic/memorize",
                json=msg
            )
            tasks.append(task)
            
            # 每 10 个消息暂停一下
            if len(tasks) % 10 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(0.1)
        
        # 处理剩余消息
        if tasks:
            await asyncio.gather(*tasks)
```

### 2. 检索优化

- 使用合适的 `top_k` 值（建议 10-30）
- 使用 `time_range_days` 限制时间范围
- 使用 `radius` 过滤低相关结果
- 轻量级检索优先，复杂查询才用 Agentic

### 3. 缓存策略

对于频繁查询，建议：
- 在应用层缓存检索结果
- 设置合理的缓存过期时间（5-30 分钟）
- 使用 Redis 等缓存系统

---

## 相关文档

- [Memory API 文档](./memory_api_zh.md)
- [Agentic Retrieval 开发指南](../dev_docs/agentic_retrieval_guide.md)
- [Agentic Retrieve 测试指南](../dev_docs/agentic_retrieve_testing.md)
- [GroupChatFormat 格式规范](../../data_format/group_chat/group_chat_format.md)

