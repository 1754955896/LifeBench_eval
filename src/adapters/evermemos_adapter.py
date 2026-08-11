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
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear
5. **TIME FILTERING (MANDATORY)**: You MUST only use memories that occurred ON OR BEFORE the question's timestamp. If context contains memories with dates AFTER the question's timestamp, IGNORE them completely. The question's timestamp is provided in the context header - respect this boundary.

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown → Memory 2 mentions A's hometown is LA → Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references]
- Original reference: [e.g., "last year" from May 2022]
- Calculated actual time: [e.g., "2021"]

## STEP 5: TIME FILTERING VERIFICATION (MANDATORY)
[Verify all memories used are on or before the question timestamp]
- Question timestamp: [extract from context header]
- Memories used and their timestamps: [list each memory with its timestamp]
- Filter out: [any memories that are AFTER the question timestamp and should NOT be used]
- If filtered memories exist: [explain why they were excluded]

## STEP 6: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 8: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 9: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the concise answer with ALL specific details preserved]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
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
        self.timeout = config.get("timeout", 600.0)
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

        # Track extraction state per conversation - used to trigger boundary detection
        self._extraction_triggered: Dict[str, bool] = {}

        # Shared httpx client for connection pooling
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(f"✅ EverMemOS Adapter initialized with output_dir={self.output_dir}")
        logger.info(f"   API URL: {self.api_url}")
        logger.info(f"   Search mode: {self.search_mode}")
        logger.info(f"   RPM limit: {self.rpm}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared httpx client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    async def close(self) -> None:
        """Cleanup resources - close shared httpx client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _format_timestamp(self, timestamp: Any) -> str:
        """Format timestamp to ISO format string."""
        if timestamp is None:
            return datetime.now().isoformat()
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp).isoformat()
        return str(timestamp)

    def _sanitize_content(self, text: str) -> str:
        """Remove characters that cause GBK encoding issues on the EverMemOS server.

        The EverMemOS API server runs on Windows with GBK console encoding.
        Characters outside GBK range (like emoji) cause HTTP 400 errors.
        """
        if not text:
            return text
        # Filter out characters that can't be encoded in GBK
        result = []
        for ch in text:
            try:
                ch.encode('gbk')
                result.append(ch)
            except UnicodeEncodeError:
                result.append('?')
        return ''.join(result)

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
                client = await self._get_client()
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

    async def _memorize_message(self, message: Dict[str, Any], conversation_id: str, client: httpx.AsyncClient = None) -> tuple:
        """Call memorize API for a single message.

        Args:
            message: Message dict
            conversation_id: Conversation ID
            client: Optional shared httpx client (will create one if not provided)

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
                _client = client or await self._get_client()
                resp = await _client.post(self.memorize_url, json=payload)
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
        """Ingest message chunks via EverMemOS HTTP API (concurrent with rate limiting).

        Messages are batched by dia_id: all messages with the same dia_id are
        concatenated into a single message with format:
        "speaker: xxx
        text: xxx
        ---
        speaker: yyy
        text: yyy"

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        # Collect all messages to process
        all_tasks = []
        conversation_ids = set()

        # First pass: collect all messages and ensure conversation meta
        for chunk in chunks:
            if not chunk.messages:
                continue

            conversation_id = chunk.conversation_id
            conversation_ids.add(conversation_id)

            # Ensure conversation metadata is saved first (sync, not parallelized)
            await self._ensure_conversation_meta(conversation_id)

            # Group messages by date (all messages from the same day = one API call)
            # This is critical: boundary detection needs the full day's messages together
            # to detect topic/time boundaries between sessions.
            date_groups: Dict[str, List] = {}
            for msg in chunk.messages:
                dia_id = getattr(msg, 'dia_id', None) or ''
                # Extract date from dia_id ("2025-01-01_note844" → "2025-01-01") or use timestamp
                group_key = dia_id[:10] if len(dia_id) >= 10 else ""
                if not group_key or len(group_key) != 10 or group_key[4] != '-':
                    group_key = chunk.session_id or f"_session_{conversation_id}"
                    if msg.timestamp:
                        group_key = f"{msg.timestamp.strftime('%Y-%m-%d')}_{group_key}"

                if group_key not in date_groups:
                    date_groups[group_key] = []
                date_groups[group_key].append(msg)

            # Create one combined message per date group
            for group_key, msgs in date_groups.items():
                if not msgs:
                    continue

                # Concatenate content: "speaker: xxx\ntext: xxx\n---\nspeaker: yyy\ntext: yyy"
                parts = []
                first_msg = msgs[0]
                for msg in msgs:
                    speaker = msg.speaker_name or getattr(msg, "speaker_id", None) or "Unknown"
                    content = msg.content or ""
                    parts.append(f"speaker: {speaker}\ntext: {content}")

                combined_content = self._sanitize_content("\n---\n".join(parts))

                sid = getattr(first_msg, "speaker_id", None) or first_msg.speaker_name or "unknown"
                message_dict = {
                    "message_id": f"{conversation_id}_{group_key}_{uuid.uuid4().hex[:8]}",
                    "create_time": self._format_timestamp(first_msg.timestamp),
                    "sender": sid,
                    "sender_name": first_msg.speaker_name or sid,
                    "type": "text",
                    "content": combined_content,
                    "group_id": conversation_id,
                    "group_name": conversation_id,
                    "scene": "assistant",
                    "refer_list": [],
                }
                all_tasks.append((message_dict, conversation_id))

        # Get shared client for all requests
        client = await self._get_client()

        # Semaphore to control concurrency based on rpm (requests per minute)
        # Use rpm // 2 as conservative estimate to avoid hitting rate limit
        # At least 1, at most rpm // 2 concurrent requests
        max_concurrent = max(1, self.rpm // 2) if self.rpm > 0 else 10
        sem = asyncio.Semaphore(max_concurrent)

        async def bounded_memorize(message_dict: Dict[str, Any], conv_id: str):
            async with sem:
                return await self._memorize_message(message_dict, conv_id, client)

        # Execute all memorize requests concurrently
        results = await asyncio.gather(
            *[bounded_memorize(msg_dict, conv_id) for msg_dict, conv_id in all_tasks],
            return_exceptions=True
        )

        # Count results
        total_messages = len(all_tasks)
        total_added = 0
        total_failed = 0
        total_extracted = 0

        for result in results:
            if isinstance(result, Exception):
                total_failed += 1
            else:
                success, status_info = result
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

        If no results found, triggers extraction first and retries.
        This handles the case where messages are still in the accumulation
        buffer and haven't been extracted into searchable memories yet.

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
                client = await self._get_client()
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("status") == "ok":
                        data = result.get("result", {})
                        memories = data.get("memories", [])
                        metadata = data.get("metadata", {})

                        # If 0 results and extraction not triggered yet, trigger it and retry
                        if not memories and not self._extraction_triggered.get(conversation_id):
                            self._extraction_triggered[conversation_id] = True
                            logger.info(f"0 results for conv {conversation_id}, triggering extraction...")
                            await self._trigger_extraction(conversation_id)
                            # Retry the search (continues the loop)
                            await asyncio.sleep(2)
                            continue

                        retrieved = []
                        for mem in memories:
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

    async def _trigger_extraction(self, conversation_id: str) -> None:
        """Trigger extraction of accumulated messages for a conversation.

        Sends a memorize call to push accumulated messages through boundary detection.
        The server will include all accumulated history from Redis + this message,
        which should trigger boundary detection if there's sufficient accumulated data.
        """
        payload = {
            "message_id": f"trigger_{conversation_id}_extraction",
            "create_time": datetime.now().isoformat(),
            "sender": "System",
            "sender_name": "System",
            "type": "text",
            "content": "Please process accumulated messages.",
            "group_id": conversation_id,
            "group_name": conversation_id,
            "scene": "assistant",
            "refer_list": [],
        }
        try:
            client = await self._get_client()
            resp = await client.post(self.memorize_url, json=payload, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                ri = result.get('result', {})
                logger.info(
                    f"Trigger extraction for {conversation_id}: "
                    f"status_info={ri.get('status_info', 'unknown')}, "
                    f"count={ri.get('count', 0)}"
                )
        except Exception as exc:
            logger.warning(f"Trigger extraction failed for {conversation_id}: {str(exc)[:200]}")

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
            client = await self._get_client()
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and "choices" in data:
                answer = data["choices"][0]["message"]["content"]
                # 学习 native 版(stage4_response):只保留 FINAL ANSWER 之后的最终答案,
                # 丢弃 CoT 推理过程,避免 answer 过长且含推理噪音
                if "FINAL ANSWER:" in answer:
                    parts = answer.split("FINAL ANSWER:", 1)
                    if len(parts) > 1 and parts[1].strip():
                        answer = parts[1].strip()
                return answer
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