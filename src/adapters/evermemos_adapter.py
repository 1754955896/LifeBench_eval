"""
EverMemOS Adapter for LifeBench_eval - HTTP API version using httpx.

Calls EverMemOS_bz HTTP API endpoints instead of direct function imports.
Uses httpx for HTTP client (same as demo for consistency).
"""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)

# Answer prompt template (same as memos_cloud for consistency)
ANSWER_PROMPT = """
You are a knowledgeable and helpful AI assistant.

# CONTEXT:
You have access to memories from a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories. Synthesize information across different entries if needed.
2. Pay close attention to the timestamps to determine the answer. If memories contain contradictory information, the **most recent memory** is the source of truth.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. Your answer must be grounded in the memories. However, you may use general world knowledge to interpret or complete information found within a memory.
5. If the question involves time references (like "last year", "two months ago", etc.), you **must** calculate the actual date based on the memory's timestamp.
6. Always convert relative time references to specific dates, months, or years in your final answer.
7. The answer must be brief (under 5-6 words) and direct, with no extra description.

{context}

Question: {question}

Answer:
"""


@register_adapter("evermemos")
class EverMemOSAdapter(BaseAdapter):
    """
    EverMemOS HTTP API adapter using httpx.

    Endpoints:
    - POST /api/v3/agentic/memorize - Store memory
    - POST /api/v3/agentic/retrieve_lightweight - Lightweight retrieval
    - POST /api/v3/agentic/retrieve_agentic - Agentic retrieval

    Configuration:
        api_url: Base URL for EverMemOS API (default: http://localhost:8001)
        max_retries: Maximum retry attempts (default 3)
        retry_delay: Base delay in seconds between retries (default 2.0)
        timeout: HTTP request timeout in seconds (default 500.0)
        rpm: Requests per minute rate limit (default 60)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        # Setup file logging if output_dir is provided
        if self.output_dir:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            log_file = output_path / "evermemos_add.log"
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)

        # API configuration
        self.api_url = config.get("api_url", "http://localhost:8001")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.timeout = config.get("timeout", 500.0)
        self.rpm = config.get("rpm", 60)

        # Endpoints
        self.memorize_url = f"{self.api_url}/api/v3/agentic/memorize"
        self.retrieve_lightweight_url = f"{self.api_url}/api/v3/agentic/retrieve_lightweight"
        self.retrieve_agentic_url = f"{self.api_url}/api/v3/agentic/retrieve_agentic"
        self.conversation_meta_url = f"{self.api_url}/api/v3/agentic/conversation-meta"

        # Conversation meta cache (to avoid duplicate calls)
        self._conversation_meta_saved: Dict[str, bool] = {}

        # Search mode
        self.search_mode = config.get("search", {}).get("mode", "lightweight")

        logger.info(f"✅ EverMemOS Adapter initialized with output_dir={self.output_dir}")
        logger.info(f"   API URL: {self.api_url}")
        logger.info(f"   Search mode: {self.search_mode}")

    async def close(self) -> None:
        """Cleanup resources (no-op for httpx, it handles connection management)."""
        pass

    def _format_timestamp(self, timestamp: Any) -> str:
        """Format timestamp to ISO format string."""
        if timestamp is None:
            return datetime.now().isoformat()
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp).isoformat()
        return str(timestamp)

    def _normalize_sender(self, speaker_id: str) -> str:
        """Normalize speaker_id to match user_details keys in conversation-meta.

        EverMemOS expects sender values that match the user_details keys
        ("User" or "Assistant"). This method normalizes various speaker_id
        formats to one of these two values.

        Args:
            speaker_id: Original speaker_id from dataset

        Returns:
            "User" or "Assistant"
        """
        if not speaker_id:
            return "User"
        sid = speaker_id.lower()
        # Match assistant-related identifiers
        if sid in ("assistant", "ai", "bot", "agent", "ai assistant", "gpt", "claude"):
            return "Assistant"
        return "User"

    async def _ensure_conversation_meta(self, conversation_id: str) -> bool:
        """Ensure conversation metadata is saved for this conversation."""
        if conversation_id in self._conversation_meta_saved:
            return True

        payload = {
            "version": "1.0.0",
            "scene": "assistant",
            "scene_desc": {},
            "name": conversation_id,
            "description": f"Evaluation conversation {conversation_id}",
            "group_id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "default_timezone": "Asia/Shanghai",
            "user_details": {
                "User": {"full_name": "User", "role": "user", "extra": {}},
                "Assistant": {"full_name": "Assistant", "role": "assistant", "extra": {}},
            },
            "tags": ["evaluation"],
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.conversation_meta_url, json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        if result.get("status") == "ok":
                            self._conversation_meta_saved[conversation_id] = True
                            return True
            except Exception as exc:
                logger.warning(
                    "conversation-meta attempt %d/%d failed (conv=%s): %s",
                    attempt + 1, self.max_retries, conversation_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        return False

    async def _memorize_message(self, message: Dict[str, Any], conversation_id: str) -> tuple:
        """Call memorize API for a single message.

        Returns:
            Tuple of (success: bool, status_info: str)
        """
        payload = {
            "message_id": message.get("message_id", f"{conversation_id}_{uuid.uuid4().hex[:8]}"),
            "create_time": message.get("create_time", datetime.now().isoformat()),
            "sender": message.get("sender", ""),
            "sender_name": message.get("sender_name", message.get("sender", "")),
            "type": "text",
            "content": message.get("content", ""),
            "group_id": conversation_id,
            "group_name": conversation_id,
            "scene": "assistant",
            "refer_list": message.get("refer_list", []),
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.memorize_url, json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        saved_count = result.get("result", {}).get("count", 0)
                        status_info = result.get("result", {}).get("status_info", "unknown")

                        if status_info == "accumulated":
                            logger.info(f"⏳ Queued: {payload['message_id']}")
                        elif status_info == "extracted":
                            logger.info(f"✓ Extracted {saved_count} memories: {payload['message_id']}")
                        else:
                            if saved_count > 0:
                                logger.info(f"✓ Extracted {saved_count} memories: {payload['message_id']}")
                            else:
                                logger.info(f"⏳ Queued: {payload['message_id']}")

                        return (result.get("status") == "ok", status_info)
                    logger.error(f"✗ Failed: HTTP {resp.status_code}: {resp.text[:200]}")
                    raise Exception(f"HTTP {resp.status_code}: {resp.text}")
            except httpx.ConnectError:
                logger.error(f"✗ Connection failed: Unable to connect to {self.api_url}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue
            except httpx.ReadTimeout:
                logger.warning(f"⚠ Timeout: Processing exceeded {self.timeout}s, skipping: {payload['message_id']}")
                continue
            except Exception as exc:
                error_detail = str(exc)[:200] if str(exc) else repr(exc)[:200]
                logger.warning(
                    "memorize attempt %d/%d failed (msg=%s): %s - %s",
                    attempt + 1, self.max_retries, payload["message_id"], type(exc).__name__, error_detail
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        return (False, "failed")

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via EverMemOS HTTP API.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        total_added = 0
        total_failed = 0
        total_messages = 0
        total_extracted = 0

        for chunk in chunks:
            if not chunk.messages:
                continue

            conversation_id = chunk.conversation_id

            # Ensure conversation metadata is saved first
            await self._ensure_conversation_meta(conversation_id)

            # Send each message to memorize API
            for msg in chunk.messages:
                total_messages += 1

                message_dict = {
                    "message_id": f"{conversation_id}_{uuid.uuid4().hex[:8]}",
                    "create_time": self._format_timestamp(msg.timestamp),
                    "sender": msg.speaker_id,
                    "sender_name": msg.speaker_name or msg.speaker_id,
                    "type": "text",
                    "content": msg.content,
                    "group_id": conversation_id,
                    "group_name": conversation_id,
                    "scene": "assistant",
                    "refer_list": [],
                }

                success, status_info = await self._memorize_message(message_dict, conversation_id)
                if success:
                    total_added += 1
                    if status_info == "extracted":
                        total_extracted += 1
                else:
                    total_failed += 1

        logger.info(f"Add chunks summary: total={total_messages}, added={total_added}, failed={total_failed}, extracted={total_extracted}")
        return {
            "type": "evermemos",
            "total_chunks": len(chunks),
            "total_messages": total_messages,
            "added": total_added,
            "failed": total_failed,
            "extracted": total_extracted,
        }

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via EverMemOS HTTP API.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as group_id)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories
        """
        top_k = kwargs.get("top_k", 20)

        # Choose retrieval endpoint based on mode
        if self.search_mode == "agentic":
            url = self.retrieve_agentic_url
            payload = {
                "query": query,
                "group_id": conversation_id,
                "time_range_days": 365,
                "top_k": top_k,
            }
            # Add LLM config if specified
            llm_config = self.config.get("llm", {})
            if llm_config.get("api_key"):
                payload["llm_config"] = {
                    "api_key": llm_config.get("api_key"),
                    "base_url": llm_config.get("base_url", "https://openrouter.ai/api/v1"),
                    "model": llm_config.get("model", "gpt-4o-mini"),
                }
        else:
            url = self.retrieve_lightweight_url
            search_config = self.config.get("search", {})
            payload = {
                "query": query,
                "group_id": conversation_id,
                "top_k": top_k,
                "retrieval_mode": search_config.get("lightweight_search_mode", "rrf"),
                "data_source": "episode",
            }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        if result.get("status") == "ok":
                            data = result.get("result", {})
                            memories = data.get("memories", [])
                            metadata = data.get("metadata", {})

                            retrieved = []
                            for mem in memories:
                                # EverMemOS returns episode/summary, not content
                                content = mem.get("episode") or mem.get("summary") or mem.get("content", "")
                                retrieved.append(RetrievedMemory(
                                    content=content,
                                    score=mem.get("score", 0.0),
                                    metadata={
                                        "timestamp": mem.get("timestamp", ""),
                                        "user_id": mem.get("user_id", ""),
                                        "group_id": mem.get("group_id", ""),
                                        "subject": mem.get("subject", ""),
                                    },
                                ))

                            return SearchResult(
                                question_id=kwargs.get("question_id", ""),
                                query=query,
                                conversation_id=conversation_id,
                                results=retrieved,
                                retrieval_metadata={
                                    "adapter": "evermemos",
                                    "mode": self.search_mode,
                                    "total_results": len(retrieved),
                                    "metadata": metadata,
                                },
                            )
                    raise Exception(f"HTTP {resp.status_code}: {resp.text}")

            except Exception as exc:
                logger.warning(
                    "search attempt %d/%d failed (conv=%s): %s",
                    attempt + 1, self.max_retries, conversation_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={"adapter": "evermemos", "mode": self.search_mode},
        )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """Generate answer using LLM given query and retrieved context.

        Args:
            query: Question text
            context: Formatted retrieved context
            conversation_id: Conversation ID
            **kwargs: Extra parameters

        Returns:
            Generated answer string
        """
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "openai")
        model = llm_config.get("model", "gpt-4o-mini")
        api_key = llm_config.get("api_key", "")
        base_url = llm_config.get("base_url", "https://openrouter.ai/api/v1")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        prompt = ANSWER_PROMPT.format(context=context, question=query)

        # OpenAI-compatible API format
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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"]
            return str(data)
        except Exception as exc:
            logger.error("Answer generation failed: %s", str(exc)[:200])
            return f"Error generating answer: {str(exc)[:100]}"

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "EverMemOS",
            "version": "1.0",
            "description": "EverMemOS memory system via HTTP API",
            "api_url": self.api_url,
        }