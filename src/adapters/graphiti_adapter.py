"""
Graphiti Adapter for LifeBench_eval.

Connects to Graphiti API server (docker) to provide
temporal context graph memory storage and retrieval.
"""
import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp
from aiolimiter import AsyncLimiter

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


def _to_ascii_safe(text: str) -> str:
    """Convert string to ASCII-safe group_id for neo4j."""
    return f"g_{hashlib.md5(text.encode('utf-8')).hexdigest()}"
@register_adapter("graphiti")
class GraphitiAdapter(BaseAdapter):
    """Graphiti adapter for temporal context graph memory.

    Configuration:
        host: Graphiti API server URL (default http://localhost:8000)
        max_retries: Maximum retry attempts (default 5)
        retry_delay: Base delay in seconds between retries (default 5.0)
        rpm: Requests per minute rate limit (default 60)
        timeout: HTTP request timeout in seconds (default 300.0)
        search_poll_interval: Seconds between search polls (default 10.0)
        search_max_attempts: Max polls for search results (default 12)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        self.host = config.get("host", "http://localhost:8000")
        self.max_retries = config.get("max_retries", 5)
        self.retry_delay = config.get("retry_delay", 5.0)
        self.timeout = config.get("timeout", 300.0)
        self.search_poll_interval = config.get("search_poll_interval", 10.0)
        self.search_max_attempts = config.get("search_max_attempts", 12)
        # Wait time after add for worker to process (seconds)
        self.add_process_wait = config.get("add_process_wait", 30.0)
        # Verification: check if data was stored after add
        self.verify_add = config.get("verify_add", True)

        self.limiter = AsyncLimiter(100000, 60)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(
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
        """Ingest message chunks via Graphiti API.

        Each chunk's messages are sent as a single /messages call.
        The API processes asynchronously, so we poll for completion.

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

            # Build messages payload - one API call per conversation
            messages = []
            for msg in chunk.messages:
                msg_dict = {
                    "content": msg.content,
                    "role_type": "user",
                    "role": msg.speaker_name or msg.speaker_id or "user",
                    "source_description": f"conversation {chunk.conversation_id}",
                }
                # Pass timestamp if available (as ISO string for JSON serialization)
                if chunk.timestamp is not None:
                    from datetime import datetime, timezone
                    dt = datetime.fromtimestamp(chunk.timestamp, tz=timezone.utc)
                    msg_dict["timestamp"] = dt.isoformat()
                messages.append(msg_dict)

            logger.info(f"ADD: group={chunk.conversation_id} -> safe_id={_to_ascii_safe(chunk.conversation_id)}, "
                        f"messages={len(messages)}, timestamp={chunk.timestamp}")

            success = await self._add_messages(
                messages=messages,
                group_id=_to_ascii_safe(chunk.conversation_id),
                conversation_id=chunk.conversation_id,
            )

            if success:
                total_added += 1
                # Verify data was stored after add
                if self.verify_add:
                    await self._verify_and_wait_processing(
                        group_id=_to_ascii_safe(chunk.conversation_id),
                        conversation_id=chunk.conversation_id,
                        num_messages=len(messages),
                    )
            else:
                total_failed += 1

        return {
            "type": "graphiti",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
        }

    async def _add_messages(
        self,
        messages: List[Dict[str, str]],
        group_id: str,
        conversation_id: str = "",
    ) -> bool:
        """Add messages to Graphiti.

        Args:
            messages: List of message dicts
            group_id: Group/conversation ID (ASCII-safe)
            conversation_id: Original conversation ID for logging

        Returns:
            True if successful, False otherwise
        """
        session = await self._get_session()

        payload = {
            "group_id": group_id,
            "messages": messages,
        }

        logger.info(f"POST /messages: group_id={group_id} ({conversation_id}), {len(messages)} messages")

        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(
                        f"{self.host}/messages",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        resp.raise_for_status()
                        await resp.json()
                        return True

            except Exception as exc:
                logger.warning(
                    "ADD attempt %d/%d failed (group=%s): %s",
                    attempt + 1, self.max_retries, group_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "ADD failed after %d attempts for group=%s",
                        self.max_retries, group_id
                    )
                    return False

        return False

    async def _verify_and_wait_processing(
        self,
        group_id: str,
        conversation_id: str,
        num_messages: int,
    ) -> None:
        """Wait for graphiti worker to process messages and verify data was stored.

        Args:
            group_id: ASCII-safe group ID
            conversation_id: Original conversation ID for logging
            num_messages: Number of messages sent
        """
        logger.info(f"VERIFY: waiting %.1fs for worker to process %d messages for {conversation_id}",
                    self.add_process_wait, num_messages)
        await asyncio.sleep(self.add_process_wait)

        # Query neo4j to verify facts were created
        try:
            session = await self._get_session()
            payload = {
                "group_ids": [group_id],
                "query": "check",
                "max_facts": 1,
            }
            async with session.post(
                f"{self.host}/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    fact_count = len(data.get("facts", []))
                    logger.info(f"VERIFY: group={conversation_id}, stored facts check: {fact_count}")
                else:
                    logger.warning(f"VERIFY: search check failed with status {resp.status}")
        except Exception as e:
            logger.warning(f"VERIFY: failed to check stored facts: {e}")

    async def _wait_for_queue_drain(self, timeout: float = 300.0) -> bool:
        """Wait for graphiti worker queue to drain to 0.

        Uses docker stats to check container CPU - when queue is being processed,
        CPU will be active. When queue is empty, CPU will be idle.

        Returns:
            True if queue drained, False if timeout
        """
        import subprocess
        start = time.time()
        check_interval = 5.0

        # Track if we're seeing activity (CPU > threshold) which means queue is being processed
        idle_count = 0
        required_idle_checks = 3  # Need 3 consecutive idle checks to confirm queue is empty

        while time.time() - start < timeout:
            elapsed = int(time.time() - start)

            try:
                # Check docker stats for the graphiti container
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", "graphiti-graph-1"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    cpu_str = result.stdout.strip().rstrip("%")
                    try:
                        cpu_val = float(cpu_str)
                        logger.info(f"QUEUE DRAIN: CPU={cpu_val}%, elapsed={elapsed}s")

                        # If CPU is very low for multiple checks, queue is likely empty
                        if cpu_val < 5.0:
                            idle_count += 1
                            if idle_count >= required_idle_checks:
                                logger.info(f"QUEUE DRAIN: Queue appears empty after {elapsed}s")
                                return True
                        else:
                            idle_count = 0  # Reset counter if we see activity
                    except ValueError:
                        pass
            except Exception as e:
                logger.debug(f"QUEUE DRAIN: stats check failed: {e}")

            await asyncio.sleep(check_interval)

        logger.warning(f"QUEUE DRAIN: Timeout after {timeout}s")
        return False

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via Graphiti API.

        Waits for queue to drain before searching, then polls for results.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as group_id)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories
        """
        top_k = kwargs.get("top_k", 10)

        # Wait for queue to drain before searching
        logger.info(f"SEARCH: Waiting for queue to drain before search...")
        await self._wait_for_queue_drain(timeout=300.0)

        session = await self._get_session()

        payload = {
            "group_ids": [_to_ascii_safe(conversation_id)],
            "query": query,
            "max_facts": top_k,
        }

        logger.info(f"SEARCH: query={query[:50]}..., conversation_id={conversation_id} -> safe_id={_to_ascii_safe(conversation_id)}, max_facts={top_k}")

        for attempt in range(self.max_retries):
            try:
                # Poll for search results
                search_success = False
                facts = []

                for poll_attempt in range(self.search_max_attempts):
                    async with self.limiter:
                        async with session.post(
                            f"{self.host}/search",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60),
                        ) as resp:
                            body = await resp.text()
                            if resp.status >= 500:
                                logger.warning(
                                    "Search poll %d returned %d: %s",
                                    poll_attempt + 1, resp.status, body[:500]
                                )
                                await asyncio.sleep(self.search_poll_interval)
                                continue
                            resp.raise_for_status()
                            data = await resp.json()

                    if resp.status == 200:
                        facts = data.get("facts", [])
                        if len(facts) > 0:
                            logger.info(f"SEARCH: found {len(facts)} facts after poll {poll_attempt+1}")
                            for f in facts:
                                logger.info(f"  FACT: {f.get('fact', '')[:80]}")
                            search_success = True
                            break
                        else:
                            logger.warning(
                                "SEARCH poll %d/%d: 0 facts, waiting %.1fs...",
                                poll_attempt + 1, self.search_max_attempts,
                                self.search_poll_interval
                            )
                            await asyncio.sleep(self.search_poll_interval)
                    else:
                        logger.warning(
                            "Search poll %d returned status %d: %s",
                            poll_attempt + 1, resp.status, body[:200]
                        )
                        await asyncio.sleep(self.search_poll_interval)

                if not search_success and not facts:
                    logger.warning(f"SEARCH: no facts found after {self.search_max_attempts} polls for group {conversation_id}")

                # Normalize results
                normalized = []
                for fact in facts:
                    entry = RetrievedMemory(
                        content=fact.get("fact", ""),
                        score=1.0,  # Graphiti doesn't provide scores
                        metadata={
                            "uuid": fact.get("uuid", ""),
                            "name": fact.get("name", ""),
                            "valid_at": fact.get("valid_at"),
                            "invalid_at": fact.get("invalid_at"),
                        }
                    )
                    normalized.append(entry)

                return SearchResult(
                    question_id=kwargs.get("question_id", ""),
                    query=query,
                    conversation_id=conversation_id,
                    results=normalized,
                    retrieval_metadata={
                        "adapter": "graphiti",
                        "total_results": len(normalized),
                    }
                )

            except aiohttp.ClientResponseError as exc:
                # Try to read response body for more details
                detail = str(exc)
                if exc.status == 500:
                    try:
                        body = await exc.response.text()
                        detail = f"500 - {body[:500]}"
                    except Exception:
                        pass
                logger.warning(
                    "SEARCH attempt %d/%d failed (group=%s): %s",
                    attempt + 1, self.max_retries, conversation_id, detail[:300]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "SEARCH failed after %d attempts for group=%s",
                        self.max_retries, conversation_id
                    )
                    return SearchResult(
                        question_id=kwargs.get("question_id", ""),
                        query=query,
                        conversation_id=conversation_id,
                        results=[],
                        retrieval_metadata={"adapter": "graphiti"}
                    )
            except Exception as exc:
                logger.warning(
                    "SEARCH attempt %d/%d failed (group=%s): %s",
                    attempt + 1, self.max_retries, conversation_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "SEARCH failed after %d attempts for group=%s",
                        self.max_retries, conversation_id
                    )
                    return SearchResult(
                        question_id=kwargs.get("question_id", ""),
                        query=query,
                        conversation_id=conversation_id,
                        results=[],
                        retrieval_metadata={"adapter": "graphiti"}
                    )

    async def delete_group(self, group_id: str) -> bool:
        """Delete all data for a group.

        Args:
            group_id: Group ID to delete

        Returns:
            True if successful
        """
        session = await self._get_session()

        try:
            async with self.limiter:
                async with session.delete(
                    f"{self.host}/group/{group_id}",
                ) as resp:
                    resp.raise_for_status()

            logger.info("Deleted group %s", group_id)
            return True

        except Exception as exc:
            logger.warning("Failed to delete group %s: %s", group_id, exc)
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
        base_url = llm_config.get("base_url", "https://api.deepseek.com/v1")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        search_result = kwargs.get("search_result")
        reference_date = kwargs.get("reference_date", "2023")

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
- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it — look for events shortly BEFORE that date.
- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" → select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") → select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance — the DATE determines the answer.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found items during reasoning that you're tempted to exclude from your answer — STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.
- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven't missed items from memories late in the list.
- The question assumes something happened. Find WHAT happened, don't say nothing happened.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified", "not mentioned", "no record", or "the memories don't say" — if ANY memory contains relevant information, give the best answer from available evidence. No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.
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

    def _build_memories_text(self, results: list, reference_date: str) -> str:
        """Build memories text with chronological sorting for LOCOMO prompt."""
        from datetime import datetime

        if not results:
            return "(No relevant memories found)"

        def _to_human_date(iso_str: str) -> str:
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

        sorted_results = sorted(results, key=lambda x: x.metadata.get("created_at", "") or "")
        lines = [
            "The following memories are presented in chronological order (oldest to newest).",
            "",
        ]
        for result in sorted_results:
            created_at = result.metadata.get("created_at", "")
            if created_at:
                date_str = _to_human_date(created_at)
                lines.append(f"({date_str}) {result.content}")
            else:
                lines.append(f"(unknown date) {result.content}")

        return "\n".join(lines)
