"""
MindMemOS Adapter for LifeBench_eval.

Uses mindmemos_sdk for API communication with MindMemOS server.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import RetrievedMemory, SearchResult

logger = logging.getLogger(__name__)

# Try to import litellm for LLM calls
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

# LoCoMo grounding rules for answer generation (same as mindmemos_eval)
LOCOMO_ANSWER_GROUNDING_RULES = """# LoCoMo memory grounding rules
- The retrieved memories are all from the same LoCoMo conversation as the question.
- Flat vanilla memories often store the named participant as "the user". If the question names a person, do not
  require that person's name to appear inside a relevant memory. Treat "the user" as a candidate alias for the named
  participant when the memory contains the requested fact.
- If a memory contains both "the user" and the named person, keep their roles separate; use the fact attributed to the
  named person, not facts attributed only to "the user".
- Do not answer that information is unavailable when a retrieved memory directly contains the requested fact, object,
  date, number, named entity, or image caption.
- For questions asking what two named people "both" share, appreciate, like, or have in common, choose a theme or fact
  supported by memories involving both people, reciprocal support, or shared activities. Do not answer with a detail
  that belongs to only one of them.
- For shared-answer questions, prefer the most concrete shared object or activity in the memories, such as outdoor
  experiences, nature, a title, or a place. Avoid abstract relationship labels like "mutual support" when a concrete
  shared activity or object is available.
- Memories with the same event_time/source_timestamp usually describe the same episode. Combine their details before
  deciding a requested fact is missing.
- If one memory in an episode matches the event and another same-event memory contains the specific missing detail,
  use the specific detail.
- For questions with relative dates such as "last week", "last weekend", "yesterday", or "the week before <date>",
  resolve the relative time using the memory event_time/source_timestamp shown in the context.
- Prefer specific facts from the retrieved memories over generic summaries. Preserve names, numbers, dates, teams,
  places, programming languages, image captions, and meal names exactly when present.
- Use lightweight common knowledge only to decode concrete retrieved entities when the question asks for a category
  or artist, such as mapping a well-known movie to its genre or a well-known song to its artists.
- If a named person recommended a concrete movie, book, song, game, or other title, treat that title as evidence about
  the named person's preference when the question asks for a genre, category, or artist.
"""

LOCOMO_ANSWER_PROMPT_EN = """
You answer LoCoMo benchmark questions using only the retrieved flat memories.

The memories are search results from one conversation. They are not structured entity slices, and they may omit the
speaker's proper name even when the question uses that name.

{grounding_rules}

# Answer procedure
1. Identify the person, event, date, object, number, or category requested by the question.
2. Scan every retrieved memory, including lower-ranked memories, before saying the information is unavailable.
3. Treat repeated memories and same-event memories as evidence from one episode. Merge complementary details from them.
4. For relative dates, calculate from the memory event_time/source_timestamp. For example, an event_time of
   2023-08-09 with "last week" means the week before 9 August 2023.
5. If the retrieved memories contain the requested fact through a candidate alias such as "the user", answer the fact
   directly instead of saying the named person is not mentioned.
6. If the question asks about two people, first look for common/shared themes instead of single-person details.
7. For shared themes, prefer concrete activities, objects, places, or genres over abstract relationship summaries.

# Retrieved memories
{context}

# Question
{question}

Return only the final answer inside <answer> and </answer>. Keep it brief, but include all exact requested names,
numbers, dates, places, teams, programming languages, image captions, and meal names.
"""


def _extract_answer(full_response: str) -> str:
    """Extract the answer from model output (same as LoCoMo)."""
    answer = full_response
    if "<answer>" in answer:
        answer = answer.split("<answer>")[1]
    if "</answer>" in answer:
        answer = answer.split("</answer>")[0]
    return answer.strip()


def _parse_locomo_datetime(date_str: str) -> Optional[datetime]:
    """Parse LoCoMo datetime string to datetime object.

    Handles formats like:
    - "1:56 pm on 8 May, 2023"
    - "1:56 pm on 8 May 2023"
    - "2023-05-08" (ISO format - defaults to 22:00:00 if no time provided)
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    # Try LoCoMo natural language formats (these have explicit time)
    formats_with_time = [
        "%I:%M %p on %d %B, %Y",    # "1:56 pm on 8 May, 2023"
        "%I:%M %p on %d %B %Y",    # "1:56 pm on 8 May 2023"
    ]

    for fmt in formats_with_time:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Try ISO format - if no time provided, default to 22:00:00
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(hour=22, minute=0, second=0)
    except ValueError:
        pass

    return None


def _session_timestamp_millis(date_str: str) -> Optional[int]:
    """Parse a LoCoMo session date string and return milliseconds timestamp.

    Same logic as LoCoMo env.py session_timestamp_millis.
    """
    dt = _parse_locomo_datetime(date_str)
    if dt is None:
        return None
    # Assume UTC if no timezone info
    from datetime import timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# LoCoMo context building helpers (same as mindmemos_eval)
_QUESTION_NAME_STOPWORDS = {
    "According", "Apr", "April", "Aug", "August", "Dec", "December",
    "Did", "Does", "Feb", "February", "Friday", "How", "Jan", "January",
    "Jul", "July", "Jun", "June", "Mar", "March", "May", "Monday",
    "Nov", "November", "Oct", "October", "Saturday", "Sep", "Sept",
    "September", "Sunday", "The", "Thursday", "Tuesday", "Wednesday",
    "What", "When", "Where", "Which", "Who", "Whose",
}

_EVENT_TIME_RE = re.compile(r"\[(?:event_time|source_timestamp):\s*([^;\]]+)")
_USER_ALIAS_RE = re.compile(r"\bthe user\b", re.IGNORECASE)


def _question_focus_names(question: str) -> list[str]:
    """Extract person names from question that are focus of the question."""
    names: list[str] = []
    for name in re.findall(r"\b[A-Z][a-z]+\b", question):
        if name in _QUESTION_NAME_STOPWORDS:
            continue
        if name not in names:
            names.append(name)
    return names


def _repeated_event_times(memories: list[str]) -> list[str]:
    """Find event times that appear in multiple memories (same-event clusters)."""
    counts: dict[str, int] = {}
    for memory in memories:
        match = _EVENT_TIME_RE.search(memory)
        if not match:
            continue
        event_time = match.group(1).strip()
        if not event_time or event_time == "unknown time":
            continue
        counts[event_time] = counts.get(event_time, 0) + 1
    return [event_time for event_time, count in counts.items() if count > 1]


def _mentions_focus_name(memory: str, focus_names: list[str]) -> bool:
    """Check if memory mentions any of the focus names."""
    return any(re.search(rf"\b{re.escape(name)}\b", memory) for name in focus_names)


def build_answer_context(memories: list[str], question: str = "") -> str:
    """Format retrieved memories for the LoCoMo answer prompt (verbatim port from LoCoMo).

    This adds:
    - Question focus names
    - Alias rules for "the user"
    - Same-event clusters
    - Participant-role notes
    """
    lines: list[str] = []
    focus_names = _question_focus_names(question)
    if focus_names:
        joined_names = ", ".join(focus_names)
        lines.append(f"Question focus names: {joined_names}")
        lines.append(
            f"Alias rule for this question: when a relevant memory says \"the user\", treat it as a candidate memory "
            f"about {joined_names} only if the memory does not already distinguish the named person from the user."
        )
        if re.search(r"\bboth\b", question, re.IGNORECASE) and len(focus_names) >= 2:
            lines.append(
                "Shared-answer rule: this question asks about multiple people, so prefer a common theme or shared "
                "activity over details about only one person."
            )
            lines.append(
                "Shared-answer specificity: choose concrete shared activities, objects, places, or genres before "
                'abstract relationship labels like "mutual support".'
            )
        lines.append("")
    repeated_event_times = _repeated_event_times(memories)
    if repeated_event_times:
        event_times = ", ".join(repeated_event_times[:5])
        lines.append(f"Same-event clusters: {event_times}")
        lines.append("Combine details from memories with these event_time/source_timestamp values before answering.")
        lines.append("")
    lines.append("Reference memories:")
    if not memories:
        lines.append("No relevant memories.")
        return "\n".join(lines)
    joined_names = ", ".join(focus_names)
    for index, memory in enumerate(memories, start=1):
        lines.append(f"{index}. {memory}")
        if focus_names and _USER_ALIAS_RE.search(memory):
            if _mentions_focus_name(memory, focus_names):
                lines.append(
                    f"   Participant-role note: memory {index} mentions both \"the user\" and {joined_names}; keep "
                    "their roles separate and use only facts attributed to the named participant as direct evidence "
                    "about that participant."
                )
            else:
                lines.append(
                    f"   Subject alias note: \"the user\" in memory {index} can refer to {joined_names} "
                    "when this memory contains the requested fact."
                )
    return "\n".join(lines)


# Try to import mindmemos_sdk, fall back to httpx if not available
try:
    from mindmemos_sdk.memory import AsyncMemoryClient, MemorySearchHit
    from mindmemos_sdk.transport import AsyncHttpTransport

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


@register_adapter("mindmemos")
class MindMemOSAdapter(BaseAdapter):
    """
    MindMemOS adapter using mindmemos_sdk.

    API endpoints (via SDK):
    - POST /v1/memory/add - Add memories
    - POST /v1/memory/search - Search memories
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        # MindMemOS API configuration
        self.api_base_url = config.get("api_base_url", "http://127.0.0.1:8000")
        self.api_key = config.get("api_key", "dev-api-key-001")
        self.timeout = config.get("timeout", 120.0)
        self.search_top_k = int(config.get("search_top_k", 50))
        self.rerank = bool(config.get("rerank", False))
        self.search_strategy = config.get("search_strategy", "agentic")

        # LLM configuration for answer generation
        llm_config = config.get("llm", {})
        self.llm_provider = llm_config.get("provider", "openai")
        self.llm_model = llm_config.get("model", "deepseek-chat")
        self.llm_api_key = llm_config.get("api_key", "")
        self.llm_base_url = llm_config.get("base_url", "https://openrouter.ai/api/v1")
        self.llm_temperature = llm_config.get("temperature", 0)
        self.llm_max_tokens = llm_config.get("max_tokens", 32768)

        # HTTP client (for non-SDK fallback)
        self._client: Optional[Any] = None

        # Track added memories count
        self._total_memories_added = 0

        logger.info(f"MindMemOS Adapter initialized")
        logger.info(f"  API URL: {self.api_base_url}")
        logger.info(f"  Search TopK: {self.search_top_k}")
        logger.info(f"  Search Strategy: {self.search_strategy}")
        logger.info(f"  Rerank: {self.rerank}")
        logger.info(f"  Using SDK: {HAS_SDK}")
        logger.info(f"  Using litellm: {HAS_LITELLM}")
        logger.info(f"  LLM: {self.llm_provider}/{self.llm_model}")

    async def _get_client(self) -> "AsyncMemoryClient":
        """Get or create SDK client."""
        if not HAS_SDK:
            raise RuntimeError(
                "mindmemos_sdk not installed. "
                "Install with: pip install mindmemos-sdk"
            )
        if self._client is None:
            transport = AsyncHttpTransport(
                base_url=self.api_base_url,
                api_key=self.api_key,
                timeout_seconds=self.timeout,
            )
            self._client = AsyncMemoryClient(transport)
        return self._client

    async def close(self) -> None:
        """Close SDK client."""
        if self._client is not None:
            await self._client._transport.aclose()
            self._client = None

    async def add_chunks(self, chunks: List[ChunkedMessage], **kwargs) -> Dict[str, Any]:
        """
        Add chunks to MindMemOS memory.

        Each ChunkedMessage becomes one add call.
        """
        if not chunks:
            return {"added": 0, "memories": 0}

        start_time = time.time()
        client = await self._get_client()

        total_added = 0
        total_memories = 0

        for chunk in chunks:
            messages = []
            # Use session-level timestamp for all messages (same as LoCoMo)
            # Pipeline passes timestamp in seconds, MindMemOS expects milliseconds
            # Use session_time_str for accurate timestamp (same as LoCoMo)
            session_timestamp_ms = None
            if chunk.session_time_str:
                # Parse LoCoMo datetime string like "1:56 pm on 8 May, 2023"
                session_timestamp_ms = _session_timestamp_millis(chunk.session_time_str)
            elif chunk.timestamp:
                # Fallback to chunk.timestamp (seconds, convert to ms)
                if isinstance(chunk.timestamp, datetime):
                    session_timestamp_ms = int(chunk.timestamp.timestamp() * 1000)
                else:
                    session_timestamp_ms = int(chunk.timestamp * 1000)

            for msg in chunk.messages:
                # Use speaker_name as role, default to "user"
                role = msg.speaker_name or "user"
                # Combine content with blip_caption and query (same as LoCoMo _message_text)
                content = msg.content or ""
                # blip_caption and query can be in metadata (from loader) or direct attributes (from locomo loader)
                blip_caption = getattr(msg, 'blip_caption', None) or (msg.metadata.get("blip_caption") if msg.metadata else None)
                query = getattr(msg, 'query', None) or (msg.metadata.get("query") if msg.metadata else None)
                if blip_caption:
                    content += f" [Shared image: {blip_caption}]"
                if query:
                    content += f" [Image context: {query}]"
                # Strip speaker prefix (same as LoCoMo strip_speaker_prefix)
                text = content
                for prefix in ("User: ", "Assistant: ", "user: ", "assistant: "):
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                        break
                messages.append({
                    "role": role,
                    "content": text,
                    "timestamp": session_timestamp_ms,
                })

            if not messages:
                continue

            try:
                metadata = {"locomo_session_key": chunk.session_id} if chunk.session_id else None
                result = await client.add(
                    messages=messages,
                    user_id=chunk.conversation_id,
                    mode="sync",
                    session_id=chunk.conversation_id,
                    metadata=metadata,
                )
                total_added += 1
                if result.memories:
                    total_memories += len(result.memories)

            except Exception as e:
                logger.error(f"Error adding chunk {chunk.conversation_id}: {e}")

        elapsed = time.time() - start_time
        self._total_memories_added += total_memories

        logger.info(
            f"ADD completed: {total_added} chunks, {total_memories} memories, {elapsed:.2f}s"
        )

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

        try:
            client = await self._get_client()
            search_result = await client.search(
                query=query,
                user_id=conversation_id,
                top_k=self.search_top_k,
                search_strategy=self.search_strategy,
                rerank=self.rerank,
                filters={"user_id": conversation_id},
                session_id=conversation_id,
            )

            elapsed = time.time() - start_time

            # Convert SDK hits to RetrievedMemory
            results = []
            for hit in search_result.memories:
                results.append(
                    RetrievedMemory(
                        content=hit.memory,
                        score=getattr(hit, "score", 0.0) or 0.0,
                        metadata={
                            "memory_id": getattr(hit, "id", ""),
                            "memory_type": getattr(hit, "memory_type", "fact"),
                            "event_time": getattr(hit, "event_time", None),
                            "source_timestamp": getattr(hit, "source_timestamp", None),
                        },
                    )
                )

            # Build formatted context with event_time/source_timestamp (same as LoCoMo)
            def _format_memory_for_answering(r: RetrievedMemory) -> str:
                event_time = r.metadata.get("event_time") if r.metadata else None
                source_timestamp = r.metadata.get("source_timestamp") if r.metadata else None
                if not event_time and not source_timestamp:
                    return r.content
                return (
                    f"[event_time: {event_time or 'unknown time'}; "
                    f"source_timestamp: {source_timestamp or 'unknown time'}] {r.content}"
                )

            formatted_context = "\n\n".join([
                f"[Memory {i+1}] {_format_memory_for_answering(r)}"
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
        Uses LoCoMo-style grounding rules and prompt format.
        """
        if not HAS_LITELLM:
            logger.warning("litellm not available, returning context as answer")
            return context

        if not self.llm_api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        # Get search_result for raw memories with metadata
        search_result = kwargs.get("search_result")

        # Build memories list with proper formatting (same as LoCoMo)
        if search_result and hasattr(search_result, 'results'):
            def _format_mem(r):
                event_time = r.metadata.get("event_time") if r.metadata else None
                source_timestamp = r.metadata.get("source_timestamp") if r.metadata else None
                if not event_time and not source_timestamp:
                    return r.content
                return (
                    f"[event_time: {event_time or 'unknown time'}; "
                    f"source_timestamp: {source_timestamp or 'unknown time'}] {r.content}"
                )
            memories_text = build_answer_context(
                [_format_mem(r) for r in search_result.results],
                question=query
            )
        else:
            # Fallback to pre-formatted context
            memories_text = context

        # Build prompt using LoCoMo format
        prompt = LOCOMO_ANSWER_PROMPT_EN.format(
            grounding_rules=LOCOMO_ANSWER_GROUNDING_RULES,
            context=memories_text,
            question=query,
        )

        try:
            response = await litellm.acompletion(
                model=f"{self.llm_provider}/{self.llm_model}",
                messages=[{"role": "user", "content": prompt}],
                api_key=self.llm_api_key,
                base_url=self.llm_base_url if self.llm_base_url else None,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
            full_response = response["choices"][0]["message"]["content"]
            answer_text = _extract_answer(full_response)
            return answer_text if answer_text else full_response
        except Exception as e:
            logger.error(f"LLM answer error: {e}")
            return f"Error: {str(e)}"

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
            "search_strategy": self.search_strategy,
            "using_sdk": HAS_SDK,
        }

    def build_lazy_index(self, conversations: List, output_dir: Any) -> Dict[str, Any]:
        """
        MindMemOS doesn't need lazy loading - it stores memories in Qdrant/Neo4j.
        Return empty index metadata.
        """
        return {}
