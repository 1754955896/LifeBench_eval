"""
Memos Cloud Adapter for LifeBench_eval.

API Reference:
- Base URL: https://memos.memtensor.cn/api/openmem/v1
- Auth: Token authorization header
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


ANSWER_PROMPT_MEMOS = """
You are a knowledgeable and helpful AI assistant.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories. Synthesize information across different entries if needed to form a complete answer.
2. Pay close attention to the timestamps to determine the answer. If memories contain contradictory information, the **most recent memory** is the source of truth.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. Your answer must be grounded in the memories. However, you may use general world knowledge to interpret or complete information found within a memory (e.g., identifying a landmark mentioned by description).
5. If the question involves time references (like "last year", "two months ago", etc.), you **must** calculate the actual date based on the memory's timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years in your final answer.
7. Do not confuse character names mentioned in memories with the actual users who created them.
8. The answer must be brief (under 5-6 words) and direct, with no extra description.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question.
2. Synthesize findings from multiple memories if a single entry is insufficient.
3. Examine timestamps and content carefully, looking for explicit dates, times, locations, or events.
4. If the answer requires calculation (e.g., converting relative time references), perform the calculation.
5. Formulate a precise, concise answer based on the evidence from the memories (and allowed world knowledge).
6. Double-check that your answer directly addresses the question asked and adheres to all instructions.
7. Ensure your final answer is specific and avoids vague time references.

{context}

Question: {question}

Answer:
"""


@register_adapter("memos_cloud")
class MemosCloudAdapter(BaseAdapter):
    """
    Memos Cloud API adapter (memtensor.cn).

    Endpoints:
    - POST /add/message - Add messages to memory
    - POST /search/memory - Search memories

    Configuration:
        api_url: Base URL for Memos Cloud API
        api_key: API key for authentication
        max_retries: Maximum retry attempts (default 3)
        retry_delay: Base delay in seconds between retries (default 2.0)
        rpm: Requests per minute rate limit (default 60)
        timeout: HTTP request timeout in seconds (default 60.0)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        self.api_url = config.get("api_url", "")
        if not self.api_url:
            raise ValueError("Memos Cloud API URL is required. Set 'api_url' in config.")

        self.api_key = config.get("api_key", "")
        if not self.api_key:
            raise ValueError("Memos Cloud API key is required. Set 'api_key' in config.")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.timeout = config.get("timeout", 60.0)
        self.rpm = config.get("rpm", 60)

        self.limiter = asyncio.Semaphore(self.rpm)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    async def close(self) -> None:
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via Memos Cloud API.

        Splits each session's messages into groups of 3 for add.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        total_added = 0
        total_failed = 0

        for chunk in chunks:
            if not chunk.messages:
                continue

            # Format all messages for this chunk, each with chat_time derived from dia_id
            all_messages = []
            for msg in chunk.messages:
                dia_id = msg.metadata.get("dia_id", "")
                if dia_id and "_" in dia_id:
                    date_part = dia_id.split("_")[0]
                    chat_time = f"{date_part} 23:39:00"
                else:
                    chat_time = "2025-01-01 23:39:00"
                all_messages.append({
                    "role": "assistant" if "assistant" in msg.speaker_name.lower() else "user",
                    "content": msg.content,
                    "chat_time": chat_time,
                })

            success = await self._add_messages(
                messages=all_messages,
                user_id=chunk.conversation_id,
                conversation_id=chunk.conversation_id,
            )

            if success:
                total_added += 1
            else:
                total_failed += 1

        return {
            "type": "memos_cloud",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
        }

    def _sanitize_user_id(self, user_id: str) -> str:
        """Sanitize user_id for API compatibility."""
        import hashlib
        # If user_id contains non-ASCII characters, create a safe hash-based ID
        try:
            user_id.encode('ascii')
            return user_id
        except UnicodeEncodeError:
            # Create a safe user_id from hash
            safe_id = f"user_{hashlib.md5(user_id.encode()).hexdigest()[:12]}"
            logger.info(f"Sanitized user_id: {user_id} -> {safe_id}")
            return safe_id

    async def _add_messages(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        conversation_id: str,
    ) -> bool:
        """Add messages to Memos Cloud.

        Args:
            messages: List of message dicts [{"role": ..., "content": ...}]
            user_id: User ID for this conversation
            conversation_id: Conversation ID

        Returns:
            True if successful, False otherwise
        """
        session = await self._get_session()
        url = f"{self.api_url}/add/message"

        # Sanitize user_id and conversation_id if they contain non-ASCII characters
        safe_user_id = self._sanitize_user_id(user_id)
        safe_conv_id = self._sanitize_user_id(conversation_id)

        logger.info(f"ADD: user_id={user_id} -> {safe_user_id}, conv_id={conversation_id} -> {safe_conv_id}, msg_count={len(messages)}")

        payload = {
            "user_id": safe_user_id,
            "conversation_id": safe_conv_id,
            "messages": messages,
        }

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()
                        if result.get("message") != "ok":
                            raise Exception(f"API error: {result}")

                        return True

            except Exception as exc:
                logger.warning(
                    "ADD attempt %d/%d failed (user=%s): %s",
                    attempt + 1, self.max_retries, user_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "ADD failed after %d attempts for user=%s",
                        self.max_retries, user_id
                    )
                    return False

        return False

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via Memos Cloud API.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as user_id)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories
        """
        top_k = kwargs.get("top_k", self.config.get("top_k", 40))
        memory_limit_number = top_k
        include_preference = kwargs.get("include_preference", True)

        session = await self._get_session()
        url = f"{self.api_url}/search/memory"

        # Sanitize user_id and conversation_id if they contain non-ASCII characters
        safe_user_id = self._sanitize_user_id(conversation_id)
        safe_conv_id = self._sanitize_user_id(conversation_id)

        payload = {
            "query": query,
            "user_id": safe_user_id,
            "conversation_id": safe_conv_id,
            "memory_limit_number": memory_limit_number,
            "include_preference": include_preference,
        }

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()
                        if result.get("message") != "ok":
                            raise Exception(f"API error: {result}")

                        data = result.get("data", {})
                        memory_list = data.get("memory_detail_list", [])
                        preference_list = data.get("preference_detail_list", [])

                        results = []
                        for item in memory_list:
                            results.append(RetrievedMemory(
                                content=item.get("memory_value", ""),
                                score=item.get("relativity", 0.0),
                                metadata={
                                    "memory_id": item.get("id", ""),
                                    "created_at": item.get("memory_time", ""),
                                    "memory_type": item.get("memory_type", ""),
                                },
                            ))

                        for item in preference_list:
                            results.append(RetrievedMemory(
                                content=item.get("preference", ""),
                                score=item.get("relativity", 0.0),
                                metadata={
                                    "preference_id": item.get("id", ""),
                                    "created_at": item.get("create_time", ""),
                                    "preference_type": item.get("preference_type", ""),
                                },
                            ))

                        results.sort(key=lambda x: x.score, reverse=True)
                        results = results[:top_k]

                        return SearchResult(
                            question_id=kwargs.get("question_id", ""),
                            query=query,
                            conversation_id=conversation_id,
                            results=results,
                            retrieval_metadata={
                                "adapter": "memos_cloud",
                                "total_results": len(results),
                            }
                        )

            except Exception as exc:
                logger.warning(
                    "SEARCH attempt %d/%d failed (user=%s): %s",
                    attempt + 1, self.max_retries, conversation_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "SEARCH failed after %d attempts for user=%s",
                        self.max_retries, conversation_id
                    )
                    return SearchResult(
                        question_id=kwargs.get("question_id", ""),
                        query=query,
                        conversation_id=conversation_id,
                        results=[],
                        retrieval_metadata={"adapter": "memos_cloud"}
                    )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """
        Generate answer using LLM given query and retrieved context.

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
        model = llm_config.get("model", "deepseek-chat")
        api_key = llm_config.get("api_key", "")
        base_url = llm_config.get("base_url", "https://openrouter.ai/api/v1")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        prompt = ANSWER_PROMPT_MEMOS.format(context=context, question=query)

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
            async with self.limiter:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        resp.raise_for_status()
                        data = await resp.json()

            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"]
            return str(data)
        except Exception as exc:
            logger.error("Answer generation failed: %s", str(exc)[:200])
            return f"Error generating answer: {str(exc)[:100]}"