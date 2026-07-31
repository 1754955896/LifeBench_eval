"""
memU-server Adapter for LifeBench_eval.

Connects to memU-server (Docker) for memory storage and retrieval.
memU-server provides synchronous /memorize and /retrieve endpoints.

API Endpoints (memU-server running locally):
    POST /memorize - Ingest conversation (synchronous)
    POST /retrieve - Search memories (synchronous)
    GET  /        - Health check
"""
import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import aiohttp

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

if TYPE_CHECKING:
    from src.models.message import Conversation, Message

logger = logging.getLogger(__name__)


@register_adapter("memu_server")
class MemUServerAdapter(BaseAdapter):
    """memU-server adapter.

    memU-server runs as a Docker container with a FastAPI server.
    It provides synchronous /memorize and /retrieve endpoints.

    Configuration:
        host: memU-server API base URL (default: http://localhost:8000)
        timeout: HTTP request timeout in seconds (default: 120)
        max_retries: Maximum retry attempts (default: 3)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        self.host = config.get("host", "http://localhost:8000").rstrip("/")
        self.timeout = config.get("timeout", 120)
        self.max_retries = config.get("max_retries", 3)
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"✅ MemUServerAdapter initialized: host={self.host}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=100)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
        return self._session

    async def add_chunks(self, chunks: List[ChunkedMessage], **kwargs) -> Dict[str, Any]:
        """
        Ingest message chunks via memU-server /memorize endpoint.

        memU-server's /memorize is synchronous - returns complete result immediately.
        Each chunk maps to one conversation for one user.

        Args:
            chunks: List of ChunkedMessage objects

        Returns:
            Dict with added count and results
        """
        session = await self._get_session()
        added = 0
        failed = 0
        results = []

        for chunk in chunks:
            conv_id = chunk.conversation_id
            messages = chunk.messages

            # Extract user_id from conversation_id
            user_id = conv_id

            # Convert messages to memU-server format
            memu_messages = []
            for msg in messages:
                role = "user" if msg.speaker_name.lower().startswith("user") else "assistant"
                content = msg.content
                timestamp = msg.timestamp.isoformat() + "Z" if msg.timestamp else None
                memu_messages.append({
                    "role": role,
                    "content": content,
                    "name": msg.speaker_name,
                    "time": timestamp,
                })

            payload = {
                "user_id": user_id,
                "conversation": {
                    "id": conv_id,
                    "messages": memu_messages,
                }
            }

            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{self.host}/memorize",
                        json=payload,
                    ) as resp:
                        result = await resp.json()
                        if resp.status in (200, 201):
                            added += 1
                            results.append({"conversation_id": conv_id, "status": "success"})
                            break
                        else:
                            logger.warning(f"memorize failed for {conv_id}: {result}")
                            if attempt == self.max_retries - 1:
                                failed += 1
                except Exception as e:
                    logger.warning(f"memorize error for {conv_id} (attempt {attempt+1}): {e}")
                    if attempt == self.max_retries - 1:
                        failed += 1
                    else:
                        await asyncio.sleep(2 ** attempt)

        return {
            "added": added,
            "failed": failed,
            "results": results,
        }

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """
        Search memories via memU-server /retrieve endpoint.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as user_id)
            index: Index (unused for memU-server)
            **kwargs: Additional parameters (top_k)

        Returns:
            SearchResult with retrieved memories
        """
        session = await self._get_session()

        # top_k from kwargs or config
        top_k = kwargs.get("top_k", self.config.get("search", {}).get("top_k", 10))

        # Use conversation_id as user_id
        user_id = conversation_id

        payload = {
            "query": query,
            "user_id": user_id,
        }

        try:
            async with session.post(
                f"{self.host}/retrieve",
                json=payload,
            ) as resp:
                result = await resp.json()
        except Exception as e:
            logger.error(f"retrieve error: {e}")
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
            )

        if resp.status != 200:
            logger.warning(f"retrieve failed: status={resp.status}, result={result}")
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
            )

        # Parse result
        items = result.get("result", {}).get("items", [])
        retrieved_memories = []
        for item in items[:top_k]:
            summary = item.get("summary", "")
            memory_type = item.get("memory_type", "")
            score = item.get("score", 0.0)
            resource_id = item.get("resource_id", "")

            retrieved_memories.append(RetrievedMemory(
                content=summary,
                score=score,
                metadata={
                    "memory_type": memory_type,
                    "resource_id": resource_id,
                }
            ))

        # Build formatted context
        context_parts = []
        for i, mem in enumerate(retrieved_memories, 1):
            context_parts.append(f"{i}. {mem.content}")
        formatted_context = "\n\n".join(context_parts) if context_parts else ""

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=retrieved_memories,
            retrieval_metadata={
                "system": "memu_server",
                "host": self.host,
                "total_found": len(items),
                "formatted_context": formatted_context,
            },
        )

    async def answer(self, query: str, context: str, conversation_id: str = "", **kwargs) -> str:
        """
        Generate answer using LLM given query and retrieved context.

        Args:
            query: Question text
            context: Formatted retrieved context
            conversation_id: Conversation ID
            **kwargs: Extra parameters, may include:
                search_result: SearchResult with raw results for chronological sorting
                reference_date: Reference date for temporal reasoning

        Returns:
            Generated answer string
        """
        llm_config = self.config.get("llm", {})
        model = llm_config.get("model", "deepseek-chat")
        api_key = llm_config.get("api_key", "")
        base_url = llm_config.get("base_url", "https://api.deepseek.com")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        search_result = kwargs.get("search_result")
        reference_date = kwargs.get("reference_date", "2023")

        if search_result and hasattr(search_result, "results"):
            memories_text = self._build_memories_text(search_result.results, reference_date)
        else:
            memories_text = context

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
            session = await self._get_session()
            async with session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                status = resp.status
                body = await resp.text()
                if status >= 500:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=status
                    )
                if status != 200:
                    logger.warning("Answer API non-200: status=%s body=%s", status, body[:300])
                resp.raise_for_status()
                data = await resp.json()

            if isinstance(data, dict) and "choices" in data:
                content = data["choices"][0]["message"]["content"]
                if content:
                    return content
            logger.warning("Answer API: no content in response: %s", str(data)[:200])
            return str(data)
        except asyncio.TimeoutError:
            logger.error("Answer generation timed out after 600s")
            return "Error generating answer: timeout"
        except aiohttp.ClientResponseError as exc:
            logger.error("Answer generation HTTP error [%s]: %s", exc.status, exc.message or exc)
            return f"Error generating answer: HTTP {exc.status}"
        except Exception as exc:
            logger.error("Answer generation failed [%s]: %s", type(exc).__name__, exc)
            return f"Error generating answer: {type(exc).__name__}"

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

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "memU-server",
            "type": "docker_api",
            "host": self.host,
            "description": "memU-server Docker adapter",
        }
