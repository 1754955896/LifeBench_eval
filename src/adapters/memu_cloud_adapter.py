"""
memU Cloud Adapter for LifeBench_eval.

API Reference (v4):
- Base URL: https://api.memu.so/api/v4/memory/
- Auth: Bearer token authorization header
- commit_results: POST /api/v4/memory/  (memorize)
- progressive_retrieve: POST /api/v4/memory/search  (retrieve)

Data model:
- RecallFile: {name, track("memory"/"skill"), description, content}
- Each recall file's content is split into segments (one per non-empty line)
- Search returns {segments: [...], files: [...], resources: [...]}
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


ANSWER_PROMPT_MEMU = """
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
8. Keep the answer concise and direct, with no extra description. For list, commonality, or multi-item questions, include all distinct supported items even if this exceeds 5-6 words.

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


@register_adapter("memu_cloud")
class MemuCloudAdapter(BaseAdapter):
    """
    memU Cloud API adapter (memu.so).

    Uses the v4 cloud API:
    - commit_results (POST /api/v4/memory/) for memorization
    - progressive_retrieve (POST /api/v4/memory/search) for retrieval

    Simplification: stores all messages from a conversation into a single recall
    file per conversation (not per-speaker), since only the human user's
    perspective matters.  search queries use conversation_id as the user scope.

    Configuration:
        api_url: Base URL for memU API (default: https://api.memu.so)
        api_key: memU API key (from memu.so)
        max_retries: Maximum retry attempts (default 3)
        retry_delay: Base delay in seconds between retries (default 2.0)
        rpm: Requests per minute rate limit (default 60)
        timeout: HTTP request timeout in seconds (default 60.0)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        self.api_url = config.get("api_url", "https://api.memu.so").rstrip("/")
        self.api_key = config.get("api_key", "")
        if not self.api_key:
            raise ValueError("memU API key is required. Set 'api_key' in config or MEMU_API_KEY in .env.")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.timeout = config.get("timeout", 60.0)
        self.rpm = config.get("rpm", 60)

        self.limiter = asyncio.Semaphore(self.rpm)
        self._session: Optional[aiohttp.ClientSession] = None

    # ── session management ─────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_chat_time(chunk) -> str:
        """Extract and format chat_time from a chunk."""
        session_time_str = getattr(chunk, 'session_time_str', '') or ''
        if session_time_str:
            for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
                try:
                    dt = datetime.strptime(session_time_str.strip(), fmt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
            if "on " in session_time_str:
                date_only = session_time_str.split("on ", 1)[-1].strip()
                try:
                    dt = datetime.strptime(date_only, "%d %B, %Y")
                    return dt.strftime("%Y-%m-%d 23:59:59")
                except ValueError:
                    pass
        if hasattr(chunk, 'timestamp') and chunk.timestamp:
            dt = datetime.fromtimestamp(chunk.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return "2025-01-01 23:39:00"

    @staticmethod
    def _sanitize(value: str) -> str:
        """Sanitize a string for use as API IDs."""
        try:
            value.encode('ascii')
            return value
        except UnicodeEncodeError:
            return f"id_{hashlib.md5(value.encode()).hexdigest()[:12]}"

    # ── ingestion (memU v4: commit_results) ────────────────────────────────────

    # Fixed project-level user_id so memU web UI shows all files under one scope.
    # agent_id differentiates per-conversation.
    _PROJECT_USER_ID = "0a40ff00-b8a3-438e-aeb6-28b0e49c64ee"

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via memU cloud commit_results API.

        Each message is formatted as a self-contained bullet point with date
        and speaker, then all messages in a chunk are committed as a single
        recall file.  No LLM extraction needed — the raw content preserves
        all information, and memU's server splits it into searchable segments.

        Uses a fixed project-level user_id for web UI visibility, with
        agent_id = conversation_id for scope separation.

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

            chat_time = self._parse_chat_time(chunk)
            conv_id = chunk.conversation_id
            safe_conv = self._sanitize(conv_id)

            # Format each message as a self-contained line with date
            content_lines = []
            for msg in chunk.messages:
                content_lines.append(f"- [{chat_time}] {msg.speaker_name}: {msg.content}")
            content = "\n".join(content_lines)

            # 按日期区分 recall file name，避免各天互相覆盖
            date_str = chat_time[:10] if chat_time else "unknown"
            recall_file_name = f"memory_{safe_conv}_{date_str}"

            payload = {
                "user": {
                    "user_id": self._PROJECT_USER_ID,
                    "agent_id": safe_conv,
                },
                "recall_files": [
                    {
                        "name": recall_file_name,
                        "track": "memory",
                        "description": f"Conversation log for {conv_id} on {chat_time[:10]}",
                        "content": content,
                    }
                ],
                "resource": [],
            }

            logger.info(
                "ADD: conv=%s msgs=%d chat_time=%s",
                conv_id, len(chunk.messages), chat_time,
            )

            success = await self._commit_results(payload)
            if success:
                total_added += 1
            else:
                total_failed += 1

        return {
            "type": "memu_cloud",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
        }

    async def _commit_results(self, payload: dict) -> bool:
        """POST to /api/v4/memory/ (commit_results endpoint).

        Args:
            payload: commit_results payload with user, recall_files, resource

        Returns:
            True if successful, False otherwise
        """
        session = await self._get_session()
        url = f"{self.api_url}/api/v4/memory"  # no trailing slash (v4 quirk)

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(url, json=payload) as response:
                        if response.status == 429:
                            text = await response.text()
                            logger.warning("Rate limited (429) on attempt %d: %s", attempt + 1, text[:200])
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()
                        logger.info("commit_results OK: recall_files=%d, resources=%d",
                                    len(result.get("recall_files", [])),
                                    len(result.get("resources", [])))
                        return True

            except Exception as exc:
                logger.warning(
                    "COMMIT attempt %d/%d failed (user=%s): %s",
                    attempt + 1, self.max_retries,
                    payload.get("user", {}).get("user_id", "?"),
                    str(exc)[:200],
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "COMMIT failed after %d attempts for user=%s",
                        self.max_retries,
                        payload.get("user", {}).get("user_id", "?"),
                    )
                    return False

        return False

    # ── search (memU v4: progressive_retrieve) ─────────────────────────────────

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via memU cloud progressive_retrieve API.

        Single search call using conversation_id as the user scope
        (simplified, no per-speaker splitting).

        Args:
            query: Query text
            conversation_id: Conversation ID
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories
        """
        top_k = kwargs.get("top_k", self.config.get("search", {}).get("top_k", 40))
        safe_conv = self._sanitize(conversation_id)

        session = await self._get_session()
        url = f"{self.api_url}/api/v4/memory/search"

        payload = {
            "query": query,
            "user_id": self._PROJECT_USER_ID,
            "agent_id": safe_conv,
        }

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(url, json=payload) as response:
                        if response.status == 429:
                            text = await response.text()
                            logger.warning("Rate limited (429) on search attempt %d: %s", attempt + 1, text[:200])
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        data = await response.json()

                        # Parse segments (fine-grained searchable units)
                        memories: List[RetrievedMemory] = []
                        seen_contents: set = set()

                        for seg in data.get("segments", []):
                            content = seg.get("text", "")
                            if not content or content in seen_contents:
                                continue
                            seen_contents.add(content)
                            memories.append(RetrievedMemory(
                                content=content,
                                score=seg.get("score", 0.0),
                                metadata={
                                    "recall_file_id": seg.get("recall_file_id", ""),
                                    "memory_type": "segment",
                                },
                            ))

                        # Parse files for additional content
                        for f in data.get("files", []):
                            content = f.get("content", "")
                            if not content or content in seen_contents:
                                continue
                            seen_contents.add(content)
                            memories.append(RetrievedMemory(
                                content=content,
                                score=f.get("score", 0.0),
                                metadata={
                                    "file_name": f.get("name", ""),
                                    "description": f.get("description", ""),
                                    "memory_type": "file",
                                },
                            ))

                        memories.sort(key=lambda x: x.score, reverse=True)

                        return SearchResult(
                            question_id=kwargs.get("question_id", ""),
                            query=query,
                            conversation_id=conversation_id,
                            results=memories[:top_k],
                            retrieval_metadata={
                                "adapter": "memu_cloud",
                                "total_results": len(memories),
                            },
                        )

            except Exception as exc:
                logger.warning(
                    "SEARCH attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, str(exc)[:200],
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("SEARCH failed after %d attempts", self.max_retries)
                    return SearchResult(
                        question_id=kwargs.get("question_id", ""),
                        query=query,
                        conversation_id=conversation_id,
                        results=[],
                        retrieval_metadata={"adapter": "memu_cloud", "error": str(exc)[:100]},
                    )

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={"adapter": "memu_cloud"},
        )

    # ── answer ──────────────────────────────────────────────────────────────────

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
        model = llm_config.get("model", "deepseek-chat")
        api_key = llm_config.get("api_key", "")
        base_url = llm_config.get("base_url", "https://openrouter.ai/api/v1")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        prompt = ANSWER_PROMPT_MEMU.format(context=context, question=query)

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