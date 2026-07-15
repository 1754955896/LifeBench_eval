"""
MindMemOS Adapter for LifeBench_eval.

Provides add_chunks, search, and answer interfaces for MindMemOS memory system.
MindMemOS supports incremental writes with deduplication.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from rich.console import Console

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


@register_adapter("mindmemos")
class MindMemOSAdapter(BaseAdapter):
    """
    MindMemOS adapter for LifeBench_eval.

    MindMemOS provides:
    - Incremental memory writes with deduplication
    - Hybrid search (vector + keyword + graph)
    - Schema-based memory modeling

    API endpoints:
    - POST /v1/memory/add - Add memories
    - POST /v1/memory/search - Search memories
    """

    # Mapping from conversation_id (sample) to API key for project-level isolation
    # Each sample gets its own API key -> project_id mapping in MindMemOS
    # smoke dataset: sample_ids are [于晓薇, 于晓雯]
    # locomo_3people dataset: sample_ids are [yxw, yxw2, fhr]
    # locomo_format dataset: sample_ids are [于晓薇, 于晓雯, 冯浩然, 叶明轩, 孙雨薇, 宋雅静, 尹浩, 陆明强, 雷明轩, 马秀兰]
    _CONVERSATION_API_KEYS = {
        # smoke dataset (order: 于晓薇=001, 于晓雯=002)
        "于晓薇": "smoke-001-api-key",
        "于晓雯": "smoke-002-api-key",
        # locomo_3people dataset (order: yxw=001, yxw2=002, fhr=003)
        "yxw": "locomo-001-api-key",
        "yxw2": "locomo-002-api-key",
        "fhr": "locomo-003-api-key",
        # locomo_format dataset (10 samples)
        "于晓薇": "lifebench-001-api-key",
        "于晓雯": "lifebench-002-api-key",
        "冯浩然": "lifebench-003-api-key",
        "叶明轩": "lifebench-004-api-key",
        "孙雨薇": "lifebench-005-api-key",
        "宋雅静": "lifebench-006-api-key",
        "尹浩": "lifebench-007-api-key",
        "陆明强": "lifebench-008-api-key",
        "雷明轩": "lifebench-009-api-key",
        "马秀兰": "lifebench-010-api-key",
    }

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector
        self.console = Console()

        # MindMemOS API configuration
        self.api_base_url = config.get("api_base_url", "http://127.0.0.1:8001")
        self.api_key = config.get("api_key", "dev-api-key-001")
        self.api_key_002 = config.get("api_key_002", "dev-api-key-002")  # for schema memory
        self.memory_algorithm = config.get("memory_algorithm", "vanilla")  # vanilla or schema
        self.search_top_k = config.get("search_top_k", 10)
        # Ensure rerank is a boolean (YAML may parse "true" as string)
        rerank_val = config.get("rerank", True)
        if isinstance(rerank_val, str):
            self.rerank = rerank_val.lower() in ("true", "1", "yes")
        else:
            self.rerank = bool(rerank_val)

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Track added memories count
        self._total_memories_added = 0

        logger.info(f"MindMemOS Adapter initialized")
        logger.info(f"  API URL: {self.api_base_url}")
        logger.info(f"  Memory Algorithm: {self.memory_algorithm}")
        logger.info(f"  Search TopK: {self.search_top_k}")
        logger.info(f"  Per-sample project isolation: enabled ({len(self._CONVERSATION_API_KEYS)} samples)")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_api_key_for_conversation(self, conversation_id: str) -> str:
        """Get API key for a specific conversation (sample).

        Each conversation maps to a dedicated API key which corresponds to a unique
        project_id in MindMemOS, providing project-level memory isolation.
        """
        api_key = self._CONVERSATION_API_KEYS.get(conversation_id)
        if api_key:
            return api_key
        # Fallback to default key if conversation_id not found
        logger.warning(f"No dedicated API key for conversation, using default")
        return self.api_key

    def _get_ascii_id_for_conversation(self, conversation_id: str) -> str:
        """Get ASCII-safe user_id for a specific conversation.

        Maps Chinese conversation_id to ASCII id to avoid encoding issues.
        Falls back to the original id if no mapping exists.
        """
        ascii_id_map = {
            # smoke dataset
            "于晓薇": "smoke_user_001",
            "于晓雯": "smoke_user_002",
            # locomo_3people dataset
            "yxw": "locomo_user_001",
            "yxw2": "locomo_user_002",
            "fhr": "locomo_user_003",
            # locomo_format dataset
            "于晓薇": "lifebench_user_001",
            "于晓雯": "lifebench_user_002",
            "冯浩然": "lifebench_user_003",
            "叶明轩": "lifebench_user_004",
            "孙雨薇": "lifebench_user_005",
            "宋雅静": "lifebench_user_006",
            "尹浩": "lifebench_user_007",
            "陆明强": "lifebench_user_008",
            "雷明轩": "lifebench_user_009",
            "马秀兰": "lifebench_user_010",
        }
        return ascii_id_map.get(conversation_id, conversation_id)

    def _get_api_key(self) -> str:
        """Get default API key based on memory algorithm."""
        if self.memory_algorithm == "schema":
            return self.api_key_002
        return self.api_key

    def _build_headers(self, api_key: str) -> dict:
        """Build request headers with specific API key."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def add_chunks(self, chunks: List[ChunkedMessage], **kwargs) -> Dict[str, Any]:
        """
        Add chunks to MindMemOS memory.

        MindMemOS supports incremental writes - each add is merged with existing memories.
        """
        if not chunks:
            return {"added": 0, "memories": 0}

        start_time = time.time()

        # Group chunks by conversation_id
        conv_chunks: Dict[str, List[ChunkedMessage]] = {}
        for chunk in chunks:
            if chunk.conversation_id not in conv_chunks:
                conv_chunks[chunk.conversation_id] = []
            conv_chunks[chunk.conversation_id].append(chunk)

        total_added = 0
        total_memories = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            for conv_id, conv_chunk_list in conv_chunks.items():
                # Build messages for this conversation
                messages = []
                for chunk in conv_chunk_list:
                    for msg in chunk.messages:
                        # Convert message to MindMemOS DialogueMessage format
                        # Normalize role: use "user" for all speakers except known assistant patterns
                        speaker_name = msg.speaker_name or msg.speaker_id or "user"
                        is_assistant = "assistant" in speaker_name.lower() or "Assistant" in speaker_name

                        if is_assistant:
                            # Prefix assistant content to ensure it's stored as user's memory
                            role = "user"
                            content = f"我收到并决定采用【AI助手建议】：{msg.content}"
                        else:
                            role = "user"  # Always use "user" for MindMemOS storage
                            content = msg.content

                        timestamp_ms = None
                        if msg.timestamp:
                            # Convert datetime to millisecond timestamp
                            if isinstance(msg.timestamp, datetime):
                                timestamp_ms = int(msg.timestamp.timestamp() * 1000)
                            else:
                                timestamp_ms = msg.timestamp

                        messages.append({
                            "role": role,
                            "content": content,
                            "timestamp": timestamp_ms,
                        })

                if not messages:
                    continue

                # Get per-conversation API key and ASCII user_id for project-level isolation
                conv_api_key = self._get_api_key_for_conversation(conv_id)
                ascii_user_id = self._get_ascii_id_for_conversation(conv_id)

                # Call MindMemOS add API
                request_data = {
                    "user_id": ascii_user_id,  # Use ASCII user_id to avoid encoding issues
                    "messages": messages,
                    "mode": "sync",
                }

                try:
                    # Use ascii-safe logging to avoid encoding errors with Chinese characters
                    logger.debug(f"Calling MindMemOS add API with {len(messages)} messages")
                    response = await client.post(
                        f"{self.api_base_url}/v1/memory/add",
                        json=request_data,
                        headers=self._build_headers(conv_api_key),
                    )
                    logger.debug(f"Response status: {response.status_code}")
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            memories_count = len(result.get("data", {}).get("memories", []))
                            total_memories += memories_count
                            total_added += 1
                            logger.debug(f"Added {memories_count} memories")
                        except Exception as e:
                            logger.error(f"JSON parse error: {e}, response text: {response.text[:500]}")
                    else:
                        logger.error(f"Add failed: {response.status_code} - {response.text[:500]}")

                except Exception as e:
                    import traceback
                    logger.error(f"Error adding memories: {type(e).__name__}: {e}\n{traceback.format_exc()}")

        elapsed = time.time() - start_time
        self._total_memories_added += total_memories

        logger.info(f"ADD completed: {total_added} conversations, {total_memories} memories, {elapsed:.2f}s")

        return {
            "added": total_added,
            "memories": total_memories,
            "total_memories": self._total_memories_added,
        }

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """
        Search memories from MindMemOS.

        Uses conversation_id as user_id for filtering.
        """
        question_id = kwargs.get("question_id", "")

        start_time = time.time()

        # Get per-conversation API key and ASCII user_id for project-level isolation
        conv_api_key = self._get_api_key_for_conversation(conversation_id)
        ascii_user_id = self._get_ascii_id_for_conversation(conversation_id)

        # Build search request
        # Note: project-level isolation via API key -> project_id mapping
        request_data = {
            "user_id": ascii_user_id,  # Use ASCII user_id to avoid encoding issues
            "query": query,
            "top_k": self.search_top_k,
            "search_strategy": "agentic",
            "rerank": self.rerank,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/v1/memory/search",
                    json=request_data,
                    headers=self._build_headers(conv_api_key),
                )

                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    items = result.get("data", {}).get("memories", [])

                    results = []
                    for item in items:
                        memory_id = item.get("id", "")
                        memory_content = item.get("memory", "")
                        score = item.get("score", 0.0)
                        memory_type = item.get("memory_type", "fact")

                        results.append(
                            RetrievedMemory(
                                content=memory_content,
                                score=float(score),
                                metadata={
                                    "memory_id": memory_id,
                                    "memory_type": memory_type,
                                },
                            )
                        )

                    # Build formatted context
                    formatted_context = "\n\n".join([
                        f"[Memory {i+1}] {r.content}"
                        for i, r in enumerate(results)
                    ])

                    return SearchResult(
                        question_id=question_id,
                        query=query,
                        conversation_id=conversation_id,
                        results=results,
                        retrieval_metadata={
                            "retrieval_mode": "mindmemos",
                            "total_latency_ms": elapsed * 1000,
                            "total_results": len(results),
                            "formatted_context": formatted_context,
                        },
                    )
                else:
                    logger.error(f"Search failed: {response.status_code} - {response.text}")
                    return SearchResult(
                        question_id=question_id,
                        query=query,
                        conversation_id=conversation_id,
                        results=[],
                        retrieval_metadata={"error": f"API error: {response.status_code}"},
                    )

        except Exception as e:
            logger.error(f"Search error: {e}")
            return SearchResult(
                question_id=question_id,
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": str(e)},
            )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """
        Generate answer using LLM given query and retrieved context.
        """
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "openai")
        model = llm_config.get("model", "deepseek-v4-flash")
        api_key = llm_config.get("api_key", "")
        base_url = llm_config.get("base_url", "https://api.deepseek.com")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.warning("No LLM API key configured for answer generation")
            return context

        prompt = f"""Based on the following retrieved memories, answer the question.

Memories:
{context}

Question: {query}

Answer:"""

        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.error(f"LLM answer failed: {response.status_code} - {response.text}")
                    return context
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return context

    def build_lazy_index(self, conversations: List, output_dir: Any) -> Dict[str, Any]:
        """
        MindMemOS doesn't need lazy loading - it stores memories in Qdrant/Neo4j.
        Return empty index metadata.
        """
        return {}

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "MindMemOS",
            "version": "1.0",
            "description": "MindMemOS memory system with incremental writes and schema modeling",
            "features": [
                "incremental_writes",
                "deduplication",
                "hybrid_search",
                "schema_modeling",
            ],
            "api_base_url": self.api_base_url,
            "memory_algorithm": self.memory_algorithm,
        }


def get_mindmemos_adapter(config: dict) -> MindMemOSAdapter:
    """Factory function to create MindMemOS adapter."""
    return MindMemOSAdapter(config)
