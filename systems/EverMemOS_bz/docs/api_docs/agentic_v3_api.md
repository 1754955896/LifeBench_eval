# Agentic API Documentation

## Overview

The Agentic API is MemSys's intelligent memory system interface, providing memory storage and smart retrieval capabilities. This API supports simple direct message format input, along with two retrieval modes of different complexity: lightweight retrieval and Agentic intelligent retrieval.

## Key Features

- ✅ **Simple Storage**: Uses the simplest single message format, no complex data structures required
- ✅ **Smart Retrieval**: Supports both lightweight and Agentic retrieval modes
- ✅ **Multi-Source Fusion**: Supports vector retrieval, BM25 keyword retrieval, and RRF fusion
- ✅ **LLM Enhancement**: Agentic mode uses LLM for multi-round intelligent retrieval
- ✅ **Flexible Filtering**: Supports filtering by user, group, time range, and other dimensions
- ✅ **Metadata Management**: Supports saving and managing conversation metadata

## Interface Specification

### POST `/api/v3/agentic/memorize`

Store a single group chat message memory

#### Request Format

**Content-Type**: `application/json`

**Request Body**: Simple direct single message format (no pre-conversion needed)

```json
{
  "group_id": "group_123",
  "group_name": "Project Discussion Group",
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+08:00",
  "sender": "user_001",
  "sender_name": "Zhang San",
  "content": "Let's discuss the technical approach for the new feature today",
  "refer_list": ["msg_000"]
}
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| group_id | string | No | Group ID |
| group_name | string | No | Group name |
| message_id | string | Yes | Unique message identifier |
| create_time | string | Yes | Message creation time (ISO 8601 format) |
| sender | string | Yes | Sender user ID |
| sender_name | string | No | Sender name (defaults to sender) |
| content | string | Yes | Message content |
| refer_list | array | No | List of referenced message IDs |

#### Response Format

**Success Response (200 OK)**

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
        "content": "User discussed technical approach for the new feature"
      }
    ],
    "count": 1,
    "status_info": "extracted"
  }
}
```

**Note**: If the message is accumulated but not yet extracted as memory (awaiting boundary detection), the response will be:

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

**Error Response (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Data format error: missing required field message_id",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/memorize"
}
```

**Error Response (500 Internal Server Error)**

```json
{
  "status": "failed",
  "code": "SYSTEM_ERROR",
  "message": "Failed to store memory, please try again later",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/memorize"
}
```

---

### POST `/api/v3/agentic/retrieve_lightweight`

Lightweight memory retrieval (Embedding + BM25 + RRF fusion)

#### Features

- Parallel execution of vector and keyword retrieval
- Uses RRF (Reciprocal Rank Fusion) to merge results
- Fast, suitable for real-time scenarios
- Supports multiple data sources and retrieval modes

#### Request Format

**Content-Type**: `application/json`

**Request Body**:

```json
{
  "query": "Beijing travel and food",
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

**Field Descriptions**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | Conditional | - | User query (can be empty when data_source=profile) |
| user_id | string | No | - | User ID (for filtering) |
| group_id | string | No | - | Group ID (for filtering) |
| time_range_days | integer | No | 365 | Time range in days |
| top_k | integer | No | 20 | Number of results to return |
| retrieval_mode | string | No | rrf | Retrieval mode: rrf/embedding/bm25 |
| data_source | string | No | episode | Data source: episode/event_log/semantic_memory/profile |
| current_time | string | No | - | Current time (YYYY-MM-DD format, for semantic memory validity filtering) |
| radius | float | No | 0.6 | COSINE similarity threshold, range [-1, 1] |

**Retrieval Mode Descriptions**:

- `rrf`: RRF fusion (default, recommended) - combines advantages of vector and keyword retrieval
- `embedding`: Pure vector retrieval - suitable for semantic similarity search
- `bm25`: Pure keyword retrieval - suitable for exact word matching

**Data Source Descriptions**:

- `episode`: Retrieve from MemCell.episode (default) - episodic memory
- `event_log`: Retrieve from event_log.atomic_fact - atomic facts
- `semantic_memory`: Retrieve from semantic memory - abstracted long-term memory
- `profile`: Profile retrieval (only requires user_id + group_id, query can be empty)

**Memory Scope Descriptions**:

- `all`: All memories (default) - uses both user_id and group_id parameters for filtering
- `personal`: Personal memories only - uses only user_id parameter, not group_id
- `group`: Group memories only - uses only group_id parameter, not user_id

**Similarity Threshold Description**:

- `radius` parameter controls the minimum COSINE similarity threshold
- Range is [-1, 1], default 0.6
- Only returns results with similarity >= radius
- Affects result quality of vector retrieval part (embedding/rrf modes)
- Effective for semantic memory and episodic memory (semantic_memory/episode)
- Event log uses L2 distance, currently not supported

#### Response Format

**Success Response (200 OK)**

```json
{
  "status": "ok",
  "message": "Retrieval successful, found 10 memories",
  "result": {
    "memories": [
      {
        "content": "User likes Beijing roast duck",
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

**Error Response (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Missing required parameter: query",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/retrieve_lightweight"
}
```

---

### POST `/api/v3/agentic/retrieve_agentic`

Agentic memory retrieval (LLM-guided multi-round intelligent retrieval)

#### Features

- Uses LLM to judge retrieval sufficiency
- Automatically performs multi-round retrieval and query refinement
- Uses Rerank to improve result quality
- Suitable for complex queries requiring deep understanding

#### Retrieval Process

1. **Round 1**: RRF hybrid retrieval (Embedding + BM25)
2. **Rerank Optimization**: Optimize results using reranking model
3. **LLM Judgment**: Determine if retrieval results are sufficient
4. **If Insufficient**: Generate multiple refined queries
5. **Round 2**: Multi-query parallel retrieval
6. **Merge and Return**: Merge results and Rerank before returning final results

#### Request Format

**Content-Type**: `application/json`

**Request Body**:

```json
{
  "query": "What does the user like to eat?",
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

**Field Descriptions**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | Yes | - | User query |
| user_id | string | No | - | User ID (for filtering) |
| group_id | string | No | - | Group ID (for filtering) |
| time_range_days | integer | No | 365 | Time range in days |
| top_k | integer | No | 20 | Number of results to return |
| llm_config | object | No | - | LLM configuration |

**llm_config Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| api_key | string | No | LLM API Key (uses environment variable OPENROUTER_API_KEY/OPENAI_API_KEY if not provided) |
| base_url | string | No | LLM API address (default https://openrouter.ai/api/v1) |
| model | string | No | LLM model (default qwen/qwen3-235b-a22b-2507) |

#### Response Format

**Success Response (200 OK)**

```json
{
  "status": "ok",
  "message": "Agentic retrieval successful, found 15 memories",
  "result": {
    "memories": [
      {
        "content": "User likes Sichuan cuisine, especially spicy hot pot",
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
      "reasoning": "Need more specific information about dietary preferences",
      "refined_queries": ["What is user's favorite cuisine?", "What does user not like to eat?"],
      "round2_count": 40,
      "final_count": 15,
      "total_latency_ms": 2345.67
    }
  }
}
```

**Error Response (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Missing required parameter: query",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/retrieve_agentic"
}
```

**Error Response (500 Internal Server Error)**

```json
{
  "status": "failed",
  "code": "SYSTEM_ERROR",
  "message": "Agentic retrieval failed, please try again later",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/retrieve_agentic"
}
```

---

### POST `/api/v3/agentic/conversation-meta`

Save conversation metadata

#### Features

Saves conversation metadata information, including scene, participants, tags, etc. Uses upsert behavior - updates the entire record if `group_id` already exists.

#### Request Format

**Content-Type**: `application/json`

**Request Body**:

```json
{
  "version": "1.0",
  "scene": "group_chat",
  "scene_desc": "Project team discussion",
  "name": "Project Discussion Group",
  "description": "Technical discussion for new feature development",
  "group_id": "group_123",
  "created_at": "2025-01-15T10:00:00+08:00",
  "default_timezone": "Asia/Shanghai",
  "user_details": {
    "user_001": {
      "full_name": "Zhang San",
      "role": "developer",
      "extra": {"department": "Engineering"}
    },
    "user_002": {
      "full_name": "Li Si",
      "role": "designer",
      "extra": {"department": "Design"}
    }
  },
  "tags": ["work", "technical"]
}
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| version | string | Yes | Metadata version |
| scene | string | Yes | Scene identifier (e.g., "group_chat") |
| scene_desc | string | Yes | Scene description |
| name | string | Yes | Conversation name |
| description | string | Yes | Conversation description |
| group_id | string | Yes | Unique group identifier |
| created_at | string | Yes | Conversation creation time (ISO 8601 format) |
| default_timezone | string | Yes | Default timezone |
| user_details | object | Yes | Participant details |
| tags | array | No | Tag list |

#### Response Format

**Success Response (200 OK)**

```json
{
  "status": "ok",
  "message": "Conversation metadata saved successfully",
  "result": {
    "id": "507f1f77bcf86cd799439011",
    "group_id": "group_123",
    "scene": "group_chat",
    "name": "Project Discussion Group",
    "version": "1.0",
    "created_at": "2025-01-15T10:00:00+08:00",
    "updated_at": "2025-01-15T10:00:00+08:00"
  }
}
```

**Error Response (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Missing required field: version",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/conversation-meta"
}
```

---

## Use Cases

### 1. Real-time Message Stream Processing + Smart Q&A

Suitable for processing real-time message streams from chat applications, storing each message as it arrives, then intelligently retrieving relevant memories based on user queries.

**Storage Example**:

```bash
curl -X POST http://localhost:1995/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "group_name": "Project Discussion Group",
    "message_id": "msg_001",
    "create_time": "2025-01-15T10:00:00+08:00",
    "sender": "user_001",
    "sender_name": "Zhang San",
    "content": "Our project will release new features next week",
    "refer_list": []
  }'
```

**Lightweight Retrieval Example**:

```bash
curl -X POST http://localhost:1995/api/v3/agentic/retrieve_lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "query": "project release date",
    "group_id": "group_123",
    "top_k": 10
  }'
```

### 2. Intelligent Customer Service System

Uses Agentic retrieval mode to automatically refine queries based on user questions, providing more accurate answers.

**Example**:

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

# Usage example
memories = asyncio.run(intelligent_customer_service("How to get a refund?"))
```

### 3. Personal Knowledge Base Management

Store personal notes and thoughts, use lightweight retrieval to quickly find relevant content.

**Example**:

```python
import httpx
import asyncio

async def save_note(content: str, user_id: str):
    """Save note"""
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
    """Search notes"""
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

## Usage Examples

### Using curl

#### Store Memory

```bash
curl -X POST http://localhost:1995/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "group_name": "Project Discussion Group",
    "message_id": "msg_001",
    "create_time": "2025-01-15T10:00:00+08:00",
    "sender": "user_001",
    "sender_name": "Zhang San",
    "content": "Let'\''s discuss the technical approach for the new feature today",
    "refer_list": []
  }'
```

#### Lightweight Retrieval

```bash
curl -X POST http://localhost:1995/api/v3/agentic/retrieve_lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "query": "technical approach",
    "group_id": "group_123",
    "top_k": 10,
    "retrieval_mode": "rrf"
  }'
```

#### Agentic Intelligent Retrieval

```bash
curl -X POST http://localhost:1995/api/v3/agentic/retrieve_agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What technical issues have we discussed?",
    "group_id": "group_123",
    "top_k": 20,
    "llm_config": {
      "model": "qwen/qwen3-235b-a22b-2507"
    }
  }'
```

### Using Python Code

#### Store Memory

```python
import httpx
import asyncio
from datetime import datetime

async def memorize_message():
    message_data = {
        "group_id": "group_123",
        "group_name": "Project Discussion Group",
        "message_id": "msg_001",
        "create_time": datetime.now().isoformat(),
        "sender": "user_001",
        "sender_name": "Zhang San",
        "content": "Let's discuss the technical approach for the new feature today",
        "refer_list": []
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/memorize",
            json=message_data
        )
        result = response.json()
        print(f"Save result: {result['message']}")
        print(f"Memory count: {result['result']['count']}")

asyncio.run(memorize_message())
```

#### Lightweight Retrieval

```python
import httpx
import asyncio

async def lightweight_retrieve():
    query_data = {
        "query": "technical approach",
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
        print(f"Found {result['result']['count']} memories")
        for memory in result['result']['memories']:
            print(f"- {memory['content']} (score: {memory.get('score', 'N/A')})")

asyncio.run(lightweight_retrieve())
```

#### Agentic Intelligent Retrieval

```python
import httpx
import asyncio

async def agentic_retrieve():
    query_data = {
        "query": "What technical issues have we discussed?",
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
            timeout=30.0  # Agentic retrieval may take longer
        )
        result = response.json()
        
        print(f"Found {result['result']['count']} memories")
        metadata = result['result']['metadata']
        print(f"Retrieval mode: {metadata['retrieval_mode']}")
        print(f"Multi-round: {metadata.get('is_multi_round', False)}")
        print(f"Total latency: {metadata['total_latency_ms']:.2f}ms")
        
        for memory in result['result']['memories']:
            print(f"- {memory['content']} (score: {memory.get('score', 'N/A')})")

asyncio.run(agentic_retrieve())
```

---

## FAQ

### 1. What's the difference between lightweight and Agentic retrieval?

**Lightweight Retrieval**:
- Fast (typically 100-500ms)
- Suitable for real-time scenarios
- Uses fixed retrieval strategy
- No LLM call cost

**Agentic Retrieval**:
- Slower (typically 2-5s)
- Suitable for complex queries
- Uses LLM to intelligently refine queries
- Has LLM API call cost

**Recommendation**:
- Real-time conversation, quick search → Use lightweight retrieval
- Complex questions, deep mining → Use Agentic retrieval

### 2. How to choose retrieval mode (rrf/embedding/bm25)?

- `rrf` (recommended): Combines advantages of vector and keyword, suitable for most scenarios
- `embedding`: Focuses on semantic similarity, suitable for conceptual queries
- `bm25`: Focuses on exact word matching, suitable for finding specific keywords

### 3. How to choose data_source parameter?

- `episode` (default): Suitable for retrieving specific conversation content and scenarios
- `event_log`: Suitable for retrieving atomic-level fact information
- `semantic_memory`: Suitable for retrieving abstract long-term memories
- `profile`: Suitable for getting user or group profile information


### 4. How to adjust radius parameter?

`radius` is the COSINE similarity threshold, range [-1, 1]:

- **0.8-1.0**: Very strict, only returns highly relevant results, may have fewer results
- **0.6-0.8** (recommended): Balances accuracy and recall
- **0.4-0.6**: More lenient, returns more results, but relevance may decrease
- **< 0.4**: Very lenient, may include many irrelevant results

**Recommendation**:
- Precise search: Use 0.7-0.8
- Regular search: Use 0.6 (default)
- Broad search: Use 0.5

### 5. How to configure LLM for Agentic retrieval?

Two ways to configure LLM:

**Method 1: Via environment variables**

```bash
export OPENROUTER_API_KEY="your_api_key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="qwen/qwen3-235b-a22b-2507"
```

**Method 2: Via request parameters**

```json
{
  "query": "What does user like?",
  "llm_config": {
    "api_key": "your_api_key",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "qwen/qwen3-235b-a22b-2507"
  }
}
```

### 6. How to handle message timestamps?

`create_time` must use ISO 8601 format, supports timezone:

```json
{
  "create_time": "2025-01-15T10:00:00+08:00"  // with timezone
}
```

Or without timezone (defaults to UTC):

```json
{
  "create_time": "2025-01-15T10:00:00"  // UTC
}
```

### 7. How to filter by time range during retrieval?

Use `time_range_days` parameter to specify memories from recent days:

```json
{
  "query": "technical discussion",
  "time_range_days": 7  // Only retrieve memories from last 7 days
}
```

### 8. How to handle profile retrieval?

Profile retrieval (data_source=profile) doesn't require query parameter, only user_id and group_id:

```json
{
  "user_id": "user_001",
  "group_id": "group_123",
  "data_source": "profile"
}
```

### 9. Are there rate limits for API calls?

Currently no hard limits, but recommended:
- **Storage interface**: No more than 100 requests per second
- **Lightweight retrieval**: No more than 50 requests per second
- **Agentic retrieval**: No more than 10 requests per second (limited by LLM API)

---

## Architecture

### Data Flow

#### Storage Process

```
Client
  ↓
  │ Simple direct single message format
  ↓
Agentic V3 Controller (agentic_v3_controller.py)
  ↓
  │ Call group_chat_converter.py
  ↓
Format Conversion (convert_simple_message_to_memorize_input)
  ↓
  │ Internal format
  ↓
Memory Manager (memory_manager.py)
  ↓
  │ Memory extraction and storage
  ↓
Database / Vector Database
```

#### Retrieval Process (Lightweight)

```
Client
  ↓
  │ Retrieval request
  ↓
Agentic V3 Controller
  ↓
Memory Manager (retrieve_lightweight)
  ↓
  ├─→ Embedding retrieval ──┐
  │                          ├─→ RRF fusion
  └─→ BM25 retrieval ────────┘
  ↓
Return results
```

#### Retrieval Process (Agentic)

```
Client
  ↓
  │ Retrieval request
  ↓
Agentic V3 Controller
  ↓
Memory Manager (retrieve_agentic)
  ↓
Round 1: RRF hybrid retrieval
  ↓
Rerank optimization
  ↓
LLM judgment of sufficiency
  ↓
  ├─→ Sufficient → Return results
  │
  └─→ Insufficient
       ↓
     Generate refined queries
       ↓
     Round 2: Multi-query parallel retrieval
       ↓
     Merge + Rerank
       ↓
     Return results
```

### Core Components

1. **Agentic V3 Controller** (`agentic_v3_controller.py`)
   - Receives HTTP requests
   - Parameter validation and conversion
   - Calls Memory Manager
   - Unified response format

2. **Memory Manager** (`memory_manager.py`)
   - Memory extraction and storage
   - Lightweight retrieval (RRF fusion)
   - Agentic intelligent retrieval
   - Vectorization and persistence

3. **Group Chat Converter** (`group_chat_converter.py`)
   - Format adaptation layer
   - Message format conversion
   - Maintains single responsibility

4. **Conversation Meta Repository** (`conversation_meta_raw_repository.py`)
   - Conversation metadata storage
   - MongoDB ODM operations
   - Upsert support

---

## Performance Optimization Recommendations

### 1. Batch Storage Optimization

For batch message storage, recommend:
- Control concurrency (suggest 10-20 concurrent)
- Add appropriate delays (50-100ms between messages)
- Use async client

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
            
            # Pause after every 10 messages
            if len(tasks) % 10 == 0:
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(0.1)
        
        # Process remaining messages
        if tasks:
            await asyncio.gather(*tasks)
```

### 2. Retrieval Optimization

- Use appropriate `top_k` value (suggest 10-30)
- Use `time_range_days` to limit time range
- Use `radius` to filter low-relevance results
- Prioritize lightweight retrieval, use Agentic only for complex queries

### 3. Caching Strategy

For frequent queries, recommend:
- Cache retrieval results at application layer
- Set reasonable cache expiration time (5-30 minutes)
- Use cache systems like Redis

---

## Related Documentation

- [Memory API Documentation](./memory_api.md)
- [Agentic Retrieval Development Guide](../dev_docs/agentic_retrieval_guide.md)
- [Agentic Retrieve Testing Guide](../dev_docs/agentic_retrieve_testing.md)
- [GroupChatFormat Specification](../../data_format/group_chat/group_chat_format.md)

