"""
TencentDB-Agent-Memory Adapter for LifeBench_eval.

Connects to TencentDB-Agent-Memory Gateway (Docker) to provide
memory storage and retrieval capabilities.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


@register_adapter("tencentdb")
class TencentDBAdapter(BaseAdapter):
    """TencentDB-Agent-Memory adapter with Docker Gateway mode.

    Configuration:
        host: Gateway URL (default: http://localhost:8420)
        api_key: Optional API key for gateway auth
        max_retries: Maximum retry attempts (default 5)
        retry_delay: Base delay in seconds between retries (default 3.0)
        timeout: HTTP request timeout in seconds (default 300.0)
        session_key_prefix: Prefix for session keys (default: "lifebench")
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        # Gateway connection settings
        self.host = config.get("host", "http://localhost:8420")
        self.api_key = config.get("api_key", "")
        self.max_retries = config.get("max_retries", 5)
        self.retry_delay = config.get("retry_delay", 3.0)
        self.timeout = config.get("timeout", 300.0)
        self.session_key_prefix = config.get("session_key_prefix", "lifebench")

        self._session: Optional[aiohttp.ClientSession] = None

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=connector,
            )
        return self._session

    async def close(self) -> None:
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _make_session_key(self, conversation_id: str) -> str:
        """Create a session key from conversation ID.

        Uses hash of conversation_id to avoid encoding issues with non-ASCII characters.
        """
        import hashlib
        # Use a hash to avoid encoding issues with non-ASCII characters in session_key
        safe_id = hashlib.md5(conversation_id.encode('utf-8')).hexdigest()[:8]
        return f"{self.session_key_prefix}_{safe_id}"

    def _is_user_message(self, msg) -> bool:
        """Check if a message is from a user based on speaker_name.

        Convention: messages are from user unless speaker_name contains "assistant".
        """
        speaker = msg.speaker_name.lower() if msg.speaker_name else ""
        speaker_id = msg.speaker_id.lower() if msg.speaker_id else ""
        # Assistant messages have "assistant" in speaker_name
        if "assistant" in speaker or "assistant" in speaker_id:
            return False
        # Everything else is treated as user message
        return True

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via TencentDB Gateway API using /seed for batch import.

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

            session_key = self._make_session_key(chunk.conversation_id)
            logger.info(f"ADD: conversation_id={chunk.conversation_id}, session_key={session_key}, messages={len(chunk.messages)}")

            # Build seed data format for /seed endpoint
            seed_data = self._build_seed_data(chunk, session_key)
            if not seed_data:
                logger.warning(f"ADD: No seed data built for {chunk.conversation_id}")
                total_failed += 1
                continue

            # Use /seed for batch import
            success = await self._seed_session(seed_data, session_key)
            if success:
                total_added += 1
                # Wait for L0 processing to complete
                await asyncio.sleep(2.0)
            else:
                total_failed += 1

        # Trigger L1/L2 extraction after all chunks processed
        if chunks and chunks[0].conversation_id and total_added > 0:
            await self.session_end(chunks[0].conversation_id)

        return {
            "type": "tencentdb",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
        }

    def _build_seed_data(self, chunk: ChunkedMessage, session_key: str) -> Dict[str, Any]:
        """Build seed data format from chunk messages.

        Args:
            chunk: ChunkedMessage object
            session_key: Session key

        Returns:
            Seed data dict for /seed endpoint
        """
        messages = chunk.messages
        if not messages:
            return {}

        # conversations: 2D array of rounds, each round is array of messages
        conversations = []
        i = 0
        processed = set()

        while i < len(messages):
            if i in processed:
                i += 1
                continue

            msg = messages[i]
            if not self._is_user_message(msg):
                processed.add(i)
                i += 1
                continue

            user_msg = msg

            # Look ahead for assistant message
            assistant_content = "ok"
            if i + 1 < len(messages) and not self._is_user_message(messages[i + 1]):
                assistant_msg = messages[i + 1]
                assistant_content = assistant_msg.content
                processed.add(i + 1)

            processed.add(i)

            # Build round as array of messages (timestamps handled by Gateway with auto_fill_timestamps=True)
            round_messages = [
                {"role": "user", "content": user_msg.content},
                {"role": "assistant", "content": assistant_content},
            ]

            conversations.append(round_messages)
            i += 1

        if not conversations:
            return {}

        return {
            "sessions": [
                {
                    "sessionKey": session_key,
                    "conversations": conversations,
                }
            ]
        }

    async def _seed_session(self, seed_data: Dict[str, Any], session_key: str) -> bool:
        """Import session data via /seed endpoint.

        Args:
            seed_data: Seed data dict with sessions
            session_key: Session key (fallback if not in data)

        Returns:
            True if successful
        """
        session = await self._get_session()

        payload = {
            "data": seed_data,
            "session_key": session_key,  # fallback if session lacks sessionKey
            "auto_fill_timestamps": True,
        }

        logger.info(f"SEED: session_key={session_key}, sessions={len(seed_data.get('sessions', []))}")

        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.host}/seed",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"SEED success: {result.get('sessions_processed', 0)} sessions, "
                                    f"rounds={result.get('rounds_processed', 0)}, "
                                    f"messages={result.get('messages_processed', 0)}")
                        return True
                    if resp.status == 400:
                        result = await resp.json()
                        logger.warning(f"SEED validation error: {result.get('error', 'unknown')}")
                        return False

            except Exception as exc:
                exc_type = type(exc).__name__
                exc_msg = str(exc) if str(exc) else repr(exc)
                logger.warning(
                    "SEED attempt %d/%d failed (session=%s): %s: %s",
                    attempt + 1, self.max_retries, session_key, exc_type, exc_msg[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        logger.error("SEED failed after %d attempts for session=%s", self.max_retries, session_key)
        return False

    async def _capture_turn(
        self,
        user_content: str,
        assistant_content: str,
        session_key: str,
        timestamp: Optional[int] = None,
    ) -> bool:
        """Capture a single turn via Gateway API.

        Args:
            user_content: User message content
            assistant_content: Assistant message content
            session_key: Session key
            timestamp: Optional epoch timestamp for the user message

        Returns:
            True if successful, False otherwise
        """
        session = await self._get_session()

        payload = {
            "user_content": user_content,
            "assistant_content": assistant_content,
            "session_key": session_key,
        }

        # Pass timestamp via messages array (Gateway expects role + content + timestamp)
        if timestamp:
            payload["messages"] = [
                {"role": "user", "content": user_content, "timestamp": timestamp},
                {"role": "assistant", "content": assistant_content, "timestamp": timestamp},
            ]

        logger.info(f"Capture payload: user_content='{user_content[:50]}...', session_key='{session_key}', timestamp={timestamp}")

        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.host}/capture", json=payload
                ) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    if resp.status == 200:
                        return True
                    # 4xx errors - don't retry
                    if 400 <= resp.status < 500:
                        result = await resp.json()
                        logger.warning(f"Capture failed: {result}")
                        return False

            except Exception as exc:
                logger.warning(
                    "CAPTURE attempt %d/%d failed (session=%s): %s",
                    attempt + 1, self.max_retries, session_key, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        logger.error("CAPTURE failed after %d attempts for session=%s", self.max_retries, session_key)
        return False

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via TencentDB Gateway API.

        Uses /search/memories (L1) and /search/conversations (L0) directly,
        NOT /recall which returns scene-navigation index instead of actual content.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as session_key)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k, strategy)

        Returns:
            SearchResult with retrieved memories
        """
        top_k = kwargs.get("top_k", 5)
        session_key = self._make_session_key(conversation_id)

        session = await self._get_session()

        # Search L1 memories (structured memories extracted from conversations)
        search_payload = {
            "query": query,
            "limit": top_k,
            "session_key": session_key,  # 搜索时限定session
        }

        memory_results = ""
        memory_total = 0
        memory_strategy = "unknown"
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.host}/search/memories", json=search_payload
                ) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    if resp.status == 200:
                        result = await resp.json()
                        memory_results = result.get("results", "")
                        memory_total = result.get("total", 0)
                        memory_strategy = result.get("strategy", "unknown")
                        break

            except Exception as exc:
                logger.warning(
                    "SEARCH/MEMORIES attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        # Search L0 conversations (raw conversation messages)
        conv_results = ""
        conv_total = 0
        conv_payload = {
            "query": query,
            "limit": top_k,
            "session_key": session_key,
        }
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.host}/search/conversations", json=conv_payload
                ) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    if resp.status == 200:
                        result = await resp.json()
                        conv_results = result.get("results", "")
                        conv_total = result.get("total", 0)
                        logger.debug(f"L0 conv search raw results: total={conv_total}, results_len={len(conv_results)}, preview={conv_results[:200] if conv_results else 'empty'}")
                        break

            except Exception as exc:
                logger.warning(
                    "SEARCH/CONVERSATIONS attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        # Combine results: prefer L1 memories, supplement with L0 conversations
        combined_context = ""

        # Check if L1 memories have actual content
        has_l1_content = (
            memory_results
            and "No matching memories found" not in memory_results
            and memory_results.strip()
        )

        if has_l1_content:
            combined_context = "## L1 Memories (Structured)\n\n" + memory_results
        else:
            logger.info(f"L1 memories not found (total={memory_total}), will use L0 conversations")

        # Supplement or fallback to L0 conversations
        has_l0_content = (
            conv_total > 0
            and conv_results
            and conv_results.strip()
            and "No matching" not in conv_results
        )

        if has_l0_content:
            if combined_context:
                combined_context += "\n\n---\n\n## L0 Conversations (Raw Messages)\n\n" + conv_results
            else:
                combined_context = "## L0 Conversations (Raw Messages)\n\n" + conv_results
            logger.info(f"Using L0 conversations: {conv_total} messages found")
        elif conv_total > 0 and conv_results:
            # Gateway returned total > 0 but results is empty/error
            logger.warning(f"L0 conversation search returned total={conv_total} but no usable content")

        logger.info(f"Search results: L1={len(memory_results)}, L0_conv={len(conv_results)}, combined={len(combined_context)}")

        # Create retrieved memories from context
        retrieved_memories = []
        if combined_context:
            retrieved_memories.append(
                RetrievedMemory(
                    content=combined_context,
                    score=1.0,
                    metadata={
                        "memory_total": memory_total,
                        "memory_strategy": memory_strategy,
                        "conv_total": conv_total,
                    }
                )
            )

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=retrieved_memories,
            retrieval_metadata={
                "adapter": "tencentdb",
                "host": self.host,
                "memory_total": memory_total,
                "memory_strategy": memory_strategy,
                "conv_total": conv_total,
                "total_results": len(retrieved_memories),
            }
        )

    async def session_end(self, conversation_id: str) -> bool:
        """End a session to trigger L1/L2 extraction and wait for processing.

        Args:
            conversation_id: Conversation ID

        Returns:
            True if successful
        """
        session = await self._get_session()
        session_key = self._make_session_key(conversation_id)

        payload = {"session_key": session_key}

        try:
            async with session.post(
                f"{self.host}/session/end", json=payload
            ) as resp:
                if resp.status == 200:
                    logger.info(f"SESSION_END: {session_key} - waiting for L1/L2 extraction...")
                    # Wait for L1/L2 extraction to complete (后台处理是异步的)
                    await asyncio.sleep(5.0)
                    return True
                logger.warning(f"Session end failed: {resp.status}")
                return False
        except Exception as exc:
            logger.warning("Session end failed: %s", str(exc)[:200])
            return False

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
        base_url = llm_config.get("base_url", "https://api.deepseek.com")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        reference_date = kwargs.get("reference_date", "2023")

        prompt = f"""You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity.

## Step 3: COMBINE AND CROSS-REFERENCE
Combine facts from multiple memories about the same topic. For listing/counting questions, extract EVERY distinct item.

## Step 4: SELECT THE BEST ANSWER
Choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.

## Step 6: INCLUSION CHECK
If you found items during reasoning that you're tempted to exclude — STOP. Include them unless you have STRONG evidence they are wrong.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified" or "no record" — if ANY memory contains relevant information, give the best answer.

{context if context else "(No relevant memories found)"}

Question: {query}

Work through Steps 1-7, then give your final answer after "ANSWER:". """

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
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    resp.raise_for_status()
                    data = await resp.json()

            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"]
            return str(data)
        except Exception as exc:
            logger.error("Answer generation failed: %s", str(exc)[:200])
            return f"Error generating answer: {str(exc)[:100]}"
