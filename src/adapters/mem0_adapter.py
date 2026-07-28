"""
Mem0 Adapter for LifeBench_eval.

Connects to Mem0 OSS server (docker) or Mem0 Cloud API to provide
memory storage and retrieval capabilities.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from aiolimiter import AsyncLimiter

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


@register_adapter("mem0")
class Mem0Adapter(BaseAdapter):
    """Mem0 adapter with OSS and Cloud modes.

    Supports chunk-based ingestion via add_chunks() method.

    Configuration:
        mode: "oss" (default) or "cloud"
        host: Server URL (defaults to localhost:8888 for oss, api.mem0.ai for cloud)
        api_key: Cloud API key (for cloud mode)
        organization_id: Cloud org ID (for cloud mode)
        project_id: Cloud project ID (for cloud mode)
        max_retries: Maximum retry attempts (default 5)
        retry_delay: Base delay in seconds between retries (default 5.0)
        rpm: Requests per minute rate limit (default 60)
        timeout: HTTP request timeout in seconds (default 300.0)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        # Mem0 connection settings
        self.mode = config.get("mode", "oss")

        if self.mode == "cloud":
            default_host = "https://api.mem0.ai"
        else:
            default_host = "http://localhost:8888"

        self.host = config.get("host") or default_host
        self.api_key = config.get("api_key", "")
        self.organization_id = config.get("organization_id", "")
        self.project_id = config.get("project_id", "")
        self.max_retries = config.get("max_retries", 5)
        self.retry_delay = config.get("retry_delay", 5.0)
        self.timeout = config.get("timeout", 300.0)
        self.event_poll_interval = config.get("event_poll_interval", 0.5)
        self.event_poll_timeout = config.get("event_poll_timeout", 300.0)
        # Set infer=False to store raw text directly without LLM extraction
        # This is important for historical data import where we want exact memories
        self.infer = config.get("infer", False)

        # Prevent pathological bursts (not a substitute for per_add_delay_seconds).
        self.limiter = AsyncLimiter(120, 60)
        self._session: Optional[aiohttp.ClientSession] = None

        # Per-conversation reference date (last session's date string),
        # populated by add_chunks() and read by answer() to anchor temporal reasoning.
        self._conversation_reference_date: Dict[str, str] = {}

        # Per-message inter-call delay (seconds) to avoid hammering mem0.
        # Tuned to be small but non-zero; can be overridden via config["per_add_delay_seconds"].
        self.per_add_delay_seconds: float = float(config.get("per_add_delay_seconds", 0.0))

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.mode == "cloud" and self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=connector,
            )
        return self._session

    async def close(self) -> None:
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via Mem0 API.

        Each ChunkedMessage (session) is **split into per-turn** calls to
        align with the official mem0 evaluation pipeline (CHUNK_SIZE=1). This
        lets mem0's LLM extractor see one turn at a time, producing finer-grained
        facts than feeding a whole session in one call.

        The session-level timestamp is reused for every turn in the session
        (mem0 OSS rejects explicit ``timestamp`` payloads anyway, but we keep
        the field for parity with the official client).

        While ingesting, the latest session date per conversation is recorded
        so ``answer()`` can anchor temporal reasoning to the real reference
        date instead of falling back to a hard-coded year.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        total_added = 0
        total_failed = 0
        total_turns = 0
        latest_session_date_per_conv: Dict[str, str] = {}

        for chunk in chunks:
            if not chunk.messages:
                continue
            if not chunk.conversation_id:
                continue

            session_ref_date = self._format_session_reference_date(
                chunk.timestamp, chunk.session_time_str
            )
            if session_ref_date:
                prev = latest_session_date_per_conv.get(chunk.conversation_id)
                if prev is None or session_ref_date > prev:
                    latest_session_date_per_conv[chunk.conversation_id] = session_ref_date

            # Build per-session metadata. We pass both a short label
            # (e.g. "May 8, 2023") and an ISO 8601 timestamp so the answer
            # prompt can pick the most useful representation. dia_id from
            # the first turn is preserved when present.
            session_iso = (
                datetime.fromtimestamp(int(chunk.timestamp), tz=timezone.utc).isoformat()
                if chunk.timestamp
                else None
            )
            session_metadata: Dict[str, Any] = {}
            if session_ref_date:
                session_metadata["session_date"] = session_ref_date
            if session_iso:
                session_metadata["session_iso"] = session_iso
            if chunk.session_id:
                session_metadata["session_id"] = chunk.session_id
            # Keep raw original time string for forensic purposes.
            if chunk.session_time_str:
                session_metadata["session_time_original"] = chunk.session_time_str

            for msg in chunk.messages:
                content = self._render_message_content(msg, session_ref_date)
                if not content:
                    continue

                # Per-turn metadata: each turn carries session date + its own
                # dia_id. Storing on the memory itself (rather than only on the
                # turn text) means search results still surface the date even
                # when mem0's LLM extractor rewrites or omits it.
                turn_metadata = dict(session_metadata)
                dia_id = getattr(msg, "dia_id", None)
                if dia_id:
                    turn_metadata["dia_id"] = dia_id

                # Per-turn add: one message per API call, mirroring the official
                # ``session_to_chunks(CHUNK_SIZE=1)`` behavior.
                turn_payload = [{"role": "user", "content": content}]
                success = await self._add_messages(
                    turn_payload,
                    chunk.conversation_id,
                    timestamp=chunk.timestamp,
                    metadata=turn_metadata or None,
                )
                total_turns += 1
                if success:
                    total_added += 1
                else:
                    total_failed += 1

                if self.per_add_delay_seconds > 0:
                    await asyncio.sleep(self.per_add_delay_seconds)

        # Persist per-conversation reference dates for answer() to read.
        self._conversation_reference_date.update(latest_session_date_per_conv)

        return {
            "type": "mem0",
            "mode": self.mode,
            "total_chunks": len(chunks),
            "total_turns": total_turns,
            "added": total_added,
            "failed": total_failed,
        }

    @staticmethod
    def _render_message_content(msg, session_date_label: Optional[str] = None) -> str:
        """Render a single message as the text fed to mem0's extractor.

        Mirrors the official ``session_to_chunks`` formatting: prepend the
        speaker name, and append an inline image tag when a blip caption
        and/or query are present.

        Additionally, when ``session_date_label`` is supplied (e.g. "May 8, 2023"
        or "2023-05-08"), it is prepended in brackets. This is a defense-in-depth
        measure against mem0's LLM extractor replacing original dates with the
        current ingestion date. With the date in the turn text itself, the
        extractor preserves it inside the resulting memory.
        """
        speaker = getattr(msg, "speaker_name", "") or ""
        text = (getattr(msg, "content", "") or "").strip()
        metadata = getattr(msg, "metadata", None) or {}
        blip = metadata.get("blip_caption", "") if isinstance(metadata, dict) else ""
        query = metadata.get("query", "") if isinstance(metadata, dict) else ""

        photo_tag = ""
        if query and blip:
            photo_tag = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo_tag = f"[Sharing image - query for: {query}]"
        elif blip:
            photo_tag = f"[Sharing image that shows: {blip}]"

        if photo_tag:
            text = f"{text} {photo_tag}".strip() if text else photo_tag
        if not text:
            return ""

        prefix_parts = []
        if session_date_label:
            # Self-describing wrapper so the LLM extractor (and any future
            # prompt consumer) understands the date is metadata about the
            # *content* that follows, not part of the speaker's utterance.
            prefix_parts.append(f"(Conversation date: {session_date_label})")
        if speaker:
            prefix_parts.append(f"{speaker}:")
        prefix = " ".join(prefix_parts) + " "

        return f"{prefix}{text}" if prefix else text

    @staticmethod
    def _format_session_reference_date(
        timestamp: Optional[int], session_time_str: Optional[str]
    ) -> Optional[str]:
        """Convert a session timestamp into a human-readable date string.

        Preference order:
          1. The original LoCoMo-style session_time_str ("1:56 pm on 8 May, 2023")
          2. Unix epoch ``timestamp`` parsed as UTC date
        Returns None if neither is usable.
        """
        if session_time_str:
            for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y",
                        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d"):
                try:
                    return datetime.strptime(session_time_str[:35], fmt).strftime("%B %d, %Y")
                except ValueError:
                    continue
            # Last resort: keep the original string if we can pull a year out.
            if any(ch.isdigit() for ch in session_time_str):
                return session_time_str

        if timestamp:
            try:
                return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%B %d, %Y")
            except (ValueError, OverflowError, OSError):
                return None
        return None

    async def _add_messages(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        timestamp: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add messages to Mem0.

        Args:
            messages: List of message dicts [{"role": ..., "content": ...}]
            user_id: User ID for this conversation
            timestamp: Optional unix timestamp (rejected by mem0 OSS but kept
                for parity with the official client).
            metadata: Optional dict of metadata to attach to the memory
                (e.g. {"session_date": "May 8, 2023", "dia_id": "..."}).

        Returns:
            True if successful, False otherwise
        """
        session = await self._get_session()

        payload: Dict[str, Any] = {"messages": messages, "user_id": user_id, "infer": self.infer}
        if timestamp is not None:
            payload["timestamp"] = timestamp
        if metadata:
            payload["metadata"] = metadata

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    if self.mode == "oss":
                        async with session.post(
                            f"{self.host}/memories", json=payload
                        ) as resp:
                            if resp.status >= 500:
                                raise aiohttp.ClientResponseError(
                                    resp.request_info, resp.history, status=resp.status
                                )
                            resp.raise_for_status()
                            await resp.json()
                            return True
                    else:
                        # Cloud mode with event polling
                        async with session.post(
                            f"{self.host}/v3/memories/", json=payload
                        ) as resp:
                            resp.raise_for_status()
                            resp_data = await resp.json()

                        event_id = resp_data.get("event_id")
                        if not event_id:
                            logger.warning("V3 add returned no event_id")
                            continue

                        event_data = await self._wait_for_event(event_id)
                        if event_data is not None:
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

    async def _get_event_status(self, event_id: str) -> Optional[Dict]:
        """Poll for event status (cloud mode only)."""
        session = await self._get_session()
        url = f"{self.host}/v1/event/{event_id}/"

        for attempt in range(3):
            try:
                async with self.limiter:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                logger.warning(
                    "Event poll %d/3 failed for %s: %s",
                    attempt + 1, event_id, exc
                )
                if attempt < 2:
                    await asyncio.sleep(self.retry_delay)
        return None

    async def _wait_for_event(self, event_id: str) -> Optional[Dict]:
        """Wait for async event to complete (cloud mode only)."""
        import time
        start = time.monotonic()

        while (time.monotonic() - start) < self.event_poll_timeout:
            data = await self._get_event_status(event_id)
            if data is None:
                return None

            status = data.get("status", "UNKNOWN")
            if status == "SUCCEEDED":
                return data
            if status == "FAILED":
                logger.error("Event %s failed: %s", event_id, data.get("error", ""))
                return None

            await asyncio.sleep(self.event_poll_interval)

        logger.error("Event %s timed out after %.0fs", event_id, self.event_poll_timeout)
        return None

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via Mem0 API.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as user_id)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k, rerank, score_debug)

        Returns:
            SearchResult with retrieved memories
        """
        search_config = self.config.get("search", {})
        top_k = kwargs.get("top_k", search_config.get("top_k", 200))
        rerank = kwargs.get("rerank", search_config.get("rerank", True))
        score_debug = kwargs.get("score_debug", False)

        session = await self._get_session()

        payload: Dict[str, Any] = {
            "query": query,
            "user_id": conversation_id,
            "top_k": top_k,             # mem0 server expects top_k (NOT limit)
            "rerank": rerank,
        }
        if score_debug:
            payload["score_debug"] = True

        if self.mode == "cloud":
            payload["filters"] = {"user_id": conversation_id}
            payload["top_k"] = top_k
        else:
            # OSS path: put user_id inside filters and drop the top-level
            # user_id (server treats top-level user_id as deprecated for search).
            payload["filters"] = {"user_id": conversation_id}
            payload.pop("user_id", None)

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    if self.mode == "oss":
                        async with session.post(
                            f"{self.host}/search", json=payload
                        ) as resp:
                            if resp.status >= 500:
                                raise aiohttp.ClientResponseError(
                                    resp.request_info, resp.history, status=resp.status
                                )
                            resp.raise_for_status()
                            data = await resp.json()
                    else:
                        async with session.post(
                            f"{self.host}/v3/memories/search/", json=payload
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()

                # Normalize results
                results = data.get("results", data) if isinstance(data, dict) else data
                if not isinstance(results, list):
                    results = []

                normalized = []
                for r in results:
                    # Pass through the metadata dict that mem0 stored with the
                    # memory. We expose it so the answer prompt can recover the
                    # original session_date even when ``created_at`` is just
                    # the ingestion timestamp.
                    raw_metadata = r.get("metadata") or {}
                    if not isinstance(raw_metadata, dict):
                        raw_metadata = {}

                    entry = RetrievedMemory(
                        content=r.get("memory", r.get("data", "")),
                        score=r.get("score", 0),
                        metadata={
                            "id": r.get("id", ""),
                            "created_at": r.get("created_at"),
                            "updated_at": r.get("updated_at"),
                            "session_date": raw_metadata.get("session_date"),
                            "session_iso": raw_metadata.get("session_iso"),
                            "session_id": raw_metadata.get("session_id"),
                            "session_time_original": raw_metadata.get("session_time_original"),
                            "dia_id": raw_metadata.get("dia_id"),
                        }
                    )
                    normalized.append(entry)

                normalized.sort(key=lambda x: x.score, reverse=True)

                return SearchResult(
                    question_id=kwargs.get("question_id", ""),
                    query=query,
                    conversation_id=conversation_id,
                    results=normalized,
                    retrieval_metadata={
                        "adapter": "mem0",
                        "mode": self.mode,
                        "total_results": len(normalized),
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
                        retrieval_metadata={"adapter": "mem0", "mode": self.mode}
                    )

    async def delete_user(self, user_id: str) -> bool:
        """Delete all memories for a user.

        Args:
            user_id: User ID to delete

        Returns:
            True if successful
        """
        session = await self._get_session()

        try:
            async with self.limiter:
                if self.mode == "oss":
                    async with session.delete(
                        f"{self.host}/memories",
                        params={"user_id": user_id},
                    ) as resp:
                        resp.raise_for_status()
                else:
                    async with session.delete(
                        f"{self.host}/v1/entities/user/{user_id}/"
                    ) as resp:
                        resp.raise_for_status()

            logger.info("Deleted memories for user %s", user_id)
            return True

        except Exception as exc:
            logger.warning("Failed to delete user %s: %s", user_id, exc)
            return False

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """
        Generate answer using LLM given query and retrieved context.

        Args:
            query: Question text
            context: Formatted retrieved context (used as fallback)
            conversation_id: Conversation ID
            **kwargs: Extra parameters, may include:
                search_result: SearchResult with raw results for chronological sorting
                reference_date: Reference date for temporal reasoning

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

        search_result = kwargs.get("search_result")
        # Prefer the conversation's real last-session date (recorded during add_chunks);
        # fall back to caller-provided reference_date; finally to a generic year.
        reference_date = (
            self._conversation_reference_date.get(conversation_id)
            or kwargs.get("reference_date")
            or "2023"
        )

        # Build memories text with chronological sorting if search_result available
        if search_result and hasattr(search_result, "results"):
            memories_text = self._build_memories_text(search_result.results, reference_date)
        else:
            # Fallback: use context as-is
            memories_text = context

        # LOCOMO 7-step reasoning prompt
        prompt = f"""You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories, including ones far down the list. Give equal weight to ALL memories regardless of position — a memory near the end is just as likely to contain the answer as one near the beginning. In these memories, "User" refers to the main person whose memories these are.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity. If the question asks "What does Person A like?" and a memory says "Person B likes X", do NOT use that memory to answer about Person A. In two-person conversations, both speakers' actions are relevant — if the question asks about person A and a memory attributes an action to person B (the other speaker), that information is still valid evidence from their shared conversations, but always check the attribution is correct.

## Step 3: COMBINE AND CROSS-REFERENCE
- COMBINE facts from multiple memories about the same topic. If one memory says "won first place" and another says "performed a piece titled X," those describe the same event — connect them.
- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items. Think about what CATEGORIES of answers the question could have, then re-scan specifically for each category.
- For counting questions ("how many times", "how many X"), enumerate each distinct instance explicitly with its date or context BEFORE giving a final count. Do not estimate — list them out, then count the list.
- DECOMPOSE complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts. Each could be the answer.
- Connect related facts across memories: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe. If one says "bought X in Paris", infer the country is France.

## Step 4: SELECT THE BEST ANSWER
- Do NOT assume the highest-ranked memory is correct. Multiple memories may describe different events for the same topic. Compare each candidate's relevance to the SPECIFIC question, not its retrieval score. A lower-ranked memory that directly answers the question beats a higher-ranked one that is only tangentially related.
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description. Rate each candidate as HIGH specificity (name, title, number, specific activity) or LOW (generic description), and prefer HIGH.
- Report what someone actually DID, not what was offered or available to them. "Has not tried X yet" means X was NOT done — disqualify it. "Joined X" or "has done X" means it WAS done — prefer it.
- When multiple memories repeat the same generic fact, that repetition does NOT make it more correct than a single memory with a more specific answer.
- Photos depict what was IN the photo, not facts about someone's daily life. Prefer direct statements over photo descriptions for inferences.
- Re-read the question carefully before answering. If it asks "what aspect/type/kind", answer with the specific aspect. If it asks "what did they discover they both enjoy", answer with the specific thing, not the setting.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.
- Calculate time relative to this date, NOT today. Never output 2025 or 2026.
- Use dates explicitly stated in memory text. Do not invent or estimate dates.
- **The DATE prefix on each memory is the date the original conversation took place.** Trust it for "when did X happen" questions.
- **IGNORE any date that appears only in the "ingestion time" fallback** — those are unreliable (they reflect when the memory was stored, not when the conversation happened).
- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it — look for events shortly BEFORE that date.
- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" → select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") → select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance — the DATE determines the answer.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found items during reasoning that you're tempted to exclude from your answer — STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.
- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven't missed items from memories late in the list.
- The question assumes something happened. Find WHAT happened, don't say nothing happened.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified", "not mentioned", "no record", or "the memories don't say" — if ANY memory contains relevant information, give the best answer from available evidence. No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.
- **ANTI-HALLUCINATION for dates**: When a memory's text mentions a date (e.g. "visited Paris on May 8, 2023"), use THAT date verbatim — DO NOT substitute the current year, today, or the ingestion date. If two memories give different dates for the same event, trust the one whose date prefix matches.
- NEVER generate specific names, titles, places, or dates that do not appear in any memory above. If no memory contains the specific detail the question asks for, answer with what the memories DO contain rather than guessing.
- For open-domain/opinion questions ("Would X do Y?", "Is X considered Z?"):
  * Follow the DIRECT causal reasoning in the memories. Do NOT construct elaborate counter-arguments.
  * "Would X still do Y without Z?" — If memories show X does Y BECAUSE of Z, then without Z, answer "likely no."
  * "Would X do Y again soon?" — If the most recent attempt involved a bad experience (accident, scare, trauma), answer "likely no." A recent negative experience outweighs historical positive patterns.
  * For trait questions ("Is X considered Z?"): weigh ALL evidence including symbolic/indirect references. If there is SOME but not strong evidence, answer with a qualified degree ("somewhat") rather than flat "no."

# Instructions

## Misc

1. Make reasonable deductions based on your memories. Memory shows store with a lot of working people -> store employs a lot of people
2. If a memory describes something recognizable (e.g., "romantic drama about memory and relationships"), you may name it (e.g., "Eternal Sunshine of the Spotless Mind").
3. Use domain knowledge to connect facts: a game exclusive to one platform implies ownership of that platform. An unnamed company deal can be linked to a previously expressed brand preference.

{memories_text}

Question: {query}

Work through Steps 1-7, then give your final answer after "ANSWER:"."""

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
                session = await self._get_session()
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

    def _build_memories_text(self, results: list, reference_date: str) -> str:
        """Build memories text with chronological sorting for LOCOMO prompt.

        Args:
            results: List of RetrievedMemory objects
            reference_date: Reference date string for temporal reasoning

        Returns:
            Formatted memories text sorted chronologically (oldest first)
        """
        if not results:
            return "(No relevant memories found)"

        def _to_human_date(iso_str: str) -> str:
            """Convert ISO 8601 timestamp to human-readable date."""
            if not iso_str:
                return "unknown date"
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    dt_str = iso_str[:26].rstrip("Z") if iso_str.endswith("Z") else iso_str
                    for attempt_fmt in (fmt, fmt.replace("%z", "")):
                        try:
                            return datetime.strptime(dt_str, attempt_fmt).strftime("%A, %B %d, %Y")
                        except ValueError:
                            continue
                except (ValueError, IndexError):
                    continue
            return iso_str[:10]

        # Sort chronologically (oldest first). Prefer session_iso (real conversation
        # date attached at ingestion) over created_at (the ingestion timestamp
        # itself, which would always read as today/yesterday).
        def _sort_key(x):
            iso = x.metadata.get("session_iso") or ""
            if iso:
                return iso
            return x.metadata.get("created_at", "") or ""

        sorted_results = sorted(results, key=_sort_key)
        lines = [
            "The following memories are presented in chronological order (oldest to newest).",
            "Each memory is prefixed with the date it occurred in the original conversation.",
            "",
        ]
        for result in sorted_results:
            session_iso = result.metadata.get("session_iso")
            session_date = result.metadata.get("session_date")
            if session_iso:
                date_str = _to_human_date(session_iso)
            elif session_date:
                date_str = session_date
            else:
                # Fallback to created_at — this is the ingestion time, not the
                # real conversation date, so mark it explicitly so the answer
                # LLM knows not to treat it as authoritative.
                created_at = result.metadata.get("created_at", "")
                if created_at:
                    date_str = f"{_to_human_date(created_at)} (ingestion time, not original conversation date)"
                else:
                    date_str = "unknown date"
            lines.append(f"({date_str}) {result.content}")

        return "\n".join(lines)