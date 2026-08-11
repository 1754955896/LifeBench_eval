"""
Memos Cloud Adapter for LifeBench_eval.

API Reference:
- Base URL: https://memos.memtensor.cn/api/openmem/v1
- Auth: Token authorization header

Modified to follow OmniMemEval approach:
- add_chunks: only the human speaker's perspective is stored — speaker_b
  is always the assistant ("{speaker_a}的Assistant"), so a single user_id
  per conversation suffices; assistant messages get role=assistant
- search: query the human speaker's user_id only
- answer: updated prompt instruction 8 to match Omni's LOCOMO_ANSWER_PROMPT
"""

import asyncio
import logging
from datetime import datetime
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

        # OmniMemEval: track conversation -> speakers mapping for multi-speaker search
        # {conversation_id: [{"speaker_name": str, "user_id": str}, ...]}
        self._speaker_map: Dict[str, List[Dict[str, str]]] = {}

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

    # ── helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_chat_time(chunk) -> str:
        """Extract and format chat_time from a chunk (same logic as original)."""
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

    def _sanitize_user_id(self, user_id: str) -> str:
        """Sanitize user_id for API compatibility."""
        import hashlib
        try:
            user_id.encode('ascii')
            return user_id
        except UnicodeEncodeError:
            safe_id = f"user_{hashlib.md5(user_id.encode()).hexdigest()[:12]}"
            logger.info("Sanitized user_id: %s -> %s", user_id, safe_id)
            return safe_id

    @staticmethod
    def _is_assistant(name: str) -> bool:
        """True if the speaker name marks an assistant (e.g. "于晓薇的Assistant")."""
        lowered = name.lower()
        return "assistant" in lowered or lowered in ("ai", "bot")

    def _human_speakers(self, speaker_names: List[str]) -> List[str]:
        """Filter out assistant speakers, keeping only the human(s)."""
        humans = [n for n in speaker_names if not self._is_assistant(n)]
        # Fallback: if everything looks like an assistant, keep the first speaker
        return humans or speaker_names[:1]

    def _register_speakers(self, conv_id: str, speaker_names: List[str]) -> None:
        """Register human speaker names for a conversation.

        Creates per-human user_ids: {conv_id}_speaker_{name}
        """
        if conv_id not in self._speaker_map:
            self._speaker_map[conv_id] = []
        existing_names = {e["speaker_name"] for e in self._speaker_map[conv_id]}
        for name in self._human_speakers(speaker_names):
            if name not in existing_names:
                user_id = f"{conv_id}_speaker_{name}"
                self._speaker_map[conv_id].append({
                    "speaker_name": name,
                    "user_id": user_id,
                })
                existing_names.add(name)

    # ── ingestion ───────────────────────────────────────────────────────────────

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via Memos Cloud API.

        OmniMemEval approach:
        - Identify all unique speaker names in the chunk
        - For each speaker, create a per-speaker user_id
        - Send ALL messages under each user_id, with roles assigned from
          that speaker's perspective (own = user, other = assistant)

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

            # Identify all unique speaker names in this chunk
            speaker_names = list(dict.fromkeys(
                msg.speaker_name for msg in chunk.messages
            ))

            # Only the human's perspective is stored — speaker_b is always
            # the assistant ("{speaker_a}的Assistant"), so one user_id per
            # conversation suffices and the chunk is sent exactly once.
            self._register_speakers(conv_id, speaker_names)
            human_names = self._human_speakers(speaker_names)

            for speaker_name in human_names:
                speaker_user_id = f"{conv_id}_speaker_{speaker_name}"

                formatted_messages = []
                for msg in chunk.messages:
                    # Own words -> user, everyone else's (assistant) -> assistant
                    role = "user" if msg.speaker_name == speaker_name else "assistant"
                    formatted_messages.append({
                        "role": role,
                        "name": msg.speaker_name,
                        "content": f"{msg.speaker_name}: {msg.content}",
                        "chat_time": chat_time,
                    })

                logger.info(
                    "ADD: conv=%s speaker=%s msgs=%d chat_time=%s",
                    conv_id, speaker_name, len(formatted_messages), chat_time,
                )

                success = await self._add_messages(
                    messages=formatted_messages,
                    user_id=speaker_user_id,
                    conversation_id=conv_id,
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

    async def _add_messages(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        conversation_id: str,
    ) -> bool:
        """Add messages to Memos Cloud.

        Args:
            messages: List of message dicts
            user_id: Per-speaker User ID
            conversation_id: Shared conversation ID

        Returns:
            True if successful, False otherwise
        """
        session = await self._get_session()
        url = f"{self.api_url}/add/message"

        safe_user_id = self._sanitize_user_id(user_id)
        safe_conv_id = self._sanitize_user_id(conversation_id)

        payload = {
            "user_id": safe_user_id,
            "conversation_id": safe_conv_id,
            "messages": messages,
            "async_mode": False,
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

    # ── search ──────────────────────────────────────────────────────────────────

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via Memos Cloud API.

        Searches the registered human speaker's user_id (speaker_b is the
        assistant and is not stored separately).

        Falls back to single-user search when no speakers are registered for
        this conversation.

        Args:
            query: Query text
            conversation_id: Conversation ID
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories and preferences
        """
        top_k = kwargs.get("top_k", self.config.get("top_k", 40))
        include_preference = kwargs.get("include_preference", True)

        speaker_entries = self._speaker_map.get(conversation_id, [])
        if not speaker_entries:
            # Fall back: search with conversation_id as user_id (original behaviour)
            return await self._search_single_user(
                query, conversation_id, conversation_id, top_k,
                include_preference, kwargs,
            )

        # OmniMemEval: search each speaker separately, merge results
        all_memories: List[RetrievedMemory] = []
        all_preferences: List[RetrievedMemory] = []

        for entry in speaker_entries:
            result = await self._search_single_user(
                query, entry["user_id"], conversation_id, top_k,
                include_preference, kwargs,
            )
            speaker_name = entry["speaker_name"]
            for mem in result.results:
                is_pref = bool(mem.metadata.get("preference_type"))
                target = all_preferences if is_pref else all_memories
                mem.metadata["speaker"] = speaker_name
                target.append(mem)

        # Merge memories and preferences, sort by score, truncate
        combined = all_memories + all_preferences
        combined.sort(key=lambda x: x.score, reverse=True)
        combined = combined[:top_k]

        pref_string = "\n".join(p.content for p in all_preferences)

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=combined,
            retrieval_metadata={
                "adapter": "memos_cloud",
                "total_results": len(combined),
                "speakers": [e["speaker_name"] for e in speaker_entries],
                "preferences": {"pref_string": pref_string} if all_preferences else {},
            }
        )

    async def _search_single_user(
        self,
        query: str,
        user_id: str,
        conversation_id: str,
        top_k: int,
        include_preference: bool,
        kwargs: dict,
    ) -> SearchResult:
        """Search memories for a single user_id."""
        session = await self._get_session()
        url = f"{self.api_url}/search/memory"

        safe_user_id = self._sanitize_user_id(user_id)

        payload = {
            "query": query,
            "user_id": safe_user_id,
            "memory_limit_number": top_k,
            "include_preference": include_preference,
            "preference_limit_number": 9,
            "relativity": 0,
            "context_format": "mixed",
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

                        memories = []
                        for item in memory_list:
                            memories.append(RetrievedMemory(
                                content=item.get("memory_value", ""),
                                score=item.get("relativity", 0.0),
                                metadata={
                                    "memory_id": item.get("id", ""),
                                    "created_at": item.get("memory_time", ""),
                                    "memory_type": item.get("memory_type", ""),
                                },
                            ))

                        preferences = []
                        for item in preference_list:
                            preferences.append(RetrievedMemory(
                                content=item.get("preference", ""),
                                score=item.get("relativity", 0.0),
                                metadata={
                                    "preference_id": item.get("id", ""),
                                    "created_at": item.get("create_time", ""),
                                    "preference_type": item.get("preference_type", ""),
                                },
                            ))

                        all_results = memories + preferences
                        all_results.sort(key=lambda x: x.score, reverse=True)
                        all_results = all_results[:top_k]

                        pref_string = "\n".join(p.content for p in preferences)

                        return SearchResult(
                            question_id=kwargs.get("question_id", ""),
                            query=query,
                            conversation_id=conversation_id,
                            results=all_results,
                            retrieval_metadata={
                                "adapter": "memos_cloud",
                                "total_results": len(all_results),
                                "preferences": {"pref_string": pref_string} if preferences else {},
                            }
                        )

            except Exception as exc:
                logger.warning(
                    "SEARCH attempt %d/%d failed (user=%s): %s",
                    attempt + 1, self.max_retries, user_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "SEARCH failed after %d attempts for user=%s",
                        self.max_retries, user_id
                    )
                    return SearchResult(
                        question_id=kwargs.get("question_id", ""),
                        query=query,
                        conversation_id=conversation_id,
                        results=[],
                        retrieval_metadata={"adapter": "memos_cloud"}
                    )

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={"adapter": "memos_cloud"}
        )

    # ── answer ──────────────────────────────────────────────────────────────────

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
