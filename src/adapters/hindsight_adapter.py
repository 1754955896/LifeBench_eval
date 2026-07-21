"""
Hindsight Adapter for LifeBench_eval.

Uses MemoryEngine directly for local operation without Docker dependency.
"""
import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


def _resolve_env_var(value: str) -> str:
    """Resolve ${VAR:default} style environment variable references at runtime.

    Args:
        value: String that may contain ${VAR} or ${VAR:default}

    Returns:
        Resolved string with env vars expanded
    """
    if not isinstance(value, str):
        return value

    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    return re.sub(pattern, replacer, value)


@register_adapter("hindsight")
class HindsightAdapter(BaseAdapter):
    """Hindsight adapter using MemoryEngine directly.

    Configuration:
        bank_id: Bank ID prefix (default "default")
        budget: Thinking budget - low/mid/high (default "mid")
        db_url: Database URL (default "pg0" for embedded)
        memory_llm_provider: LLM provider for memory (default from env)
        memory_llm_api_key: LLM API key for memory (default from env)
        memory_llm_model: LLM model for memory (default from env)
        memory_llm_base_url: LLM base URL for memory (default from env)
        answer_llm_provider: LLM provider for answer generation
        answer_llm_api_key: LLM API key for answer generation
        answer_llm_model: LLM model for answer generation
        answer_llm_base_url: LLM base URL for answer generation
        max_retries: Maximum retry attempts (default 5)
        retry_delay: Base delay in seconds between retries (default 2.0)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        # Bank configuration
        self.bank_id = config.get("bank_id", "default")
        self.budget_str = config.get("budget", "mid")

        # Database configuration
        self.db_url = config.get("db_url", os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"))

        # Memory LLM configuration (for fact extraction/consolidation)
        self.memory_llm_provider = config.get(
            "memory_llm_provider", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq")
        )
        self.memory_llm_api_key = config.get(
            "memory_llm_api_key", os.getenv("HINDSIGHT_API_LLM_API_KEY", "")
        )
        self.memory_llm_model = config.get(
            "memory_llm_model", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b")
        )
        self.memory_llm_base_url = config.get(
            "memory_llm_base_url", os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None
        )

        # Answer LLM configuration (falls back to memory LLM config)
        self.answer_llm_provider = config.get(
            "answer_llm_provider",
            config.get("llm", {}).get("provider", os.getenv("HINDSIGHT_API_ANSWER_LLM_PROVIDER", self.memory_llm_provider))
        )
        self.answer_llm_api_key = config.get(
            "answer_llm_api_key",
            config.get("llm", {}).get("api_key", os.getenv("HINDSIGHT_API_ANSWER_LLM_API_KEY", self.memory_llm_api_key))
        )
        self.answer_llm_model = config.get(
            "answer_llm_model",
            config.get("llm", {}).get("model", os.getenv("HINDSIGHT_API_ANSWER_LLM_MODEL", "gpt-4o-mini"))
        )
        self.answer_llm_base_url = config.get(
            "answer_llm_base_url",
            config.get("llm", {}).get("base_url", os.getenv("HINDSIGHT_API_ANSWER_LLM_BASE_URL", self.memory_llm_base_url) or "")
        )
        self.answer_llm_temperature = config.get("llm", {}).get("temperature", 0)
        self.answer_llm_max_tokens = config.get("llm", {}).get("max_tokens", 32768)

        # Retry configuration
        self.max_retries = config.get("max_retries", 5)
        self.retry_delay = config.get("retry_delay", 2.0)

        # Initialize MemoryEngine lazily
        self._memory: Optional[Any] = None
        self._llm_config: Optional[Any] = None

    def _get_bank_id(self, conversation_id: str) -> str:
        """Get bank_id for a conversation."""
        return f"{self.bank_id}_{conversation_id}"

    async def _get_memory(self):
        """Get MemoryEngine from builder's config (shared instance).

        If not found in config (standalone usage), creates a new instance.
        """
        if self._memory is None:
            # Try to get from config (set by builder)
            if "_hindsight_memory" in self.config:
                self._memory = self.config["_hindsight_memory"]
                logger.info("Using shared MemoryEngine from builder")
                return self._memory

            # Fallback: create own instance (standalone mode)
            from hindsight_api import MemoryEngine
            from hindsight_api.config import get_config

            # Configure logging
            get_config().configure_logging()

            # Resolve environment variables at runtime (in case builder set them after __init__)
            memory_llm_api_key = _resolve_env_var(self.memory_llm_api_key) or os.environ.get("LLM_API_KEY", "")
            memory_llm_base_url = _resolve_env_var(self.memory_llm_base_url) if self.memory_llm_base_url else None
            memory_llm_model = _resolve_env_var(self.memory_llm_model) or os.environ.get("LLM_MODEL", "deepseek-v4-flash")
            memory_llm_provider = _resolve_env_var(self.memory_llm_provider) or os.environ.get("LLM_PROVIDER", "openai")

            self._memory = MemoryEngine(
                db_url=self.db_url,
                memory_llm_provider=memory_llm_provider,
                memory_llm_api_key=memory_llm_api_key,
                memory_llm_model=memory_llm_model,
                memory_llm_base_url=memory_llm_base_url,
            )
            await self._memory.initialize()
            logger.info("MemoryEngine initialized (standalone mode)")
        return self._memory

    async def _get_llm_config(self):
        """Get or create LLMConfig for answer generation."""
        if self._llm_config is None:
            from hindsight_api.engine.llm_wrapper import LLMConfig

            # Resolve environment variables at runtime with fallbacks
            answer_llm_api_key = _resolve_env_var(self.answer_llm_api_key) or os.environ.get("LLM_API_KEY", "")
            answer_llm_base_url = _resolve_env_var(self.answer_llm_base_url) if self.answer_llm_base_url else os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
            answer_llm_model = _resolve_env_var(self.answer_llm_model) or os.environ.get("LLM_MODEL", "deepseek-v4-flash")
            answer_llm_provider = _resolve_env_var(self.answer_llm_provider) or os.environ.get("LLM_PROVIDER", "openai")

            self._llm_config = LLMConfig(
                provider=answer_llm_provider,
                api_key=answer_llm_api_key,
                base_url=answer_llm_base_url,
                model=answer_llm_model,
                reasoning_effort="high",
            )
        return self._llm_config

    def _map_budget(self, budget_str: str):
        """Map budget string to Budget enum."""
        from hindsight_api.engine.memory_engine import Budget

        budget_map = {
            "low": Budget.LOW,    # ~100 candidates
            "mid": Budget.MID,    # ~300 candidates
            "high": Budget.HIGH,  # ~1000 candidates
        }
        return budget_map.get(budget_str.lower(), Budget.MID)

    def _format_session_content(self, chunk: ChunkedMessage) -> str:
        """Format ChunkedMessage messages into session content string.

        Includes blip_caption and query fields for image context, matching the
        LoCoMo benchmark's data preservation approach.
        """
        lines = []
        for msg in chunk.messages:
            # Base content
            content = msg.content or ""

            # Include blip_caption if available (image description from BLIP model)
            blip_caption = getattr(msg, 'blip_caption', None)
            if not blip_caption and msg.metadata:
                blip_caption = msg.metadata.get("blip_caption", "")
            if blip_caption:
                content += f" [Shared image: {blip_caption}]"

            # Include query if available (image query/context)
            query = getattr(msg, 'query', None)
            if not query and msg.metadata:
                query = msg.metadata.get("query", "")
            if query:
                content += f" [Image context: {query}]"

            lines.append(f"{msg.speaker_name}: {content}")
        return "\n".join(lines)

    def _parse_session_time(self, chunk: ChunkedMessage) -> Optional[datetime]:
        """Parse session time from chunk timestamp or session_time_str."""
        if chunk.timestamp is not None:
            return datetime.fromtimestamp(chunk.timestamp, tz=timezone.utc)

        if chunk.session_time_str:
            # Try parsing common formats
            formats = [
                "%I:%M %p on %d %B, %Y",  # "1:56 pm on 8 May, 2023"
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(chunk.session_time_str, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

        return None

    async def close(self) -> None:
        """Cleanup resources - no-op since builder manages MemoryEngine lifecycle.

        The MemoryEngine is owned by the builder and will be closed via
        builder.cleanup() which is called after adapter.close() in cli.py.
        """
        # Note: MemoryEngine lifecycle is managed by HindsightBuilder
        # Do NOT close memory here - let builder handle it
        self._llm_config = None
        logger.debug("HindsightAdapter.close() - MemoryEngine managed by builder")

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via MemoryEngine batch ingestion.

        Groups all chunks per conversation_id into a single session for
        efficient batch processing.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        from hindsight_api.models import RequestContext

        total_added = 0
        total_failed = 0

        memory = await self._get_memory()

        # Group chunks by conversation_id
        conv_sessions: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            if not chunk.messages:
                continue

            content = self._format_session_content(chunk)
            if not content.strip():
                continue

            session = {
                "content": content,
                "context": f"Conversation session {chunk.session_id or chunk.conversation_id}",
                "event_date": self._parse_session_time(chunk),
            }

            conv_sessions.setdefault(chunk.conversation_id, []).append(session)

        # Ingest each conversation's sessions via retain_batch_async
        for conversation_id, sessions in conv_sessions.items():
            bank_id = self._get_bank_id(conversation_id)

            for attempt in range(self.max_retries):
                try:
                    await memory.retain_batch_async(
                        bank_id=bank_id,
                        contents=sessions,
                        request_context=RequestContext(),
                    )
                    total_added += 1
                    break
                except Exception as e:
                    logger.warning(
                        "RETAIN attempt %d/%d failed (bank_id=%s): %s",
                        attempt + 1, self.max_retries, bank_id, str(e)[:200]
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        logger.error(
                            "RETAIN failed after %d attempts for bank_id=%s",
                            self.max_retries, bank_id
                        )
                        total_failed += 1

        return {
            "type": "hindsight",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
        }

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via MemoryEngine recall.

        Args:
            query: Query text
            conversation_id: Conversation ID (used to derive bank_id)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k, question_date)

        Returns:
            SearchResult with retrieved memories
        """
        from hindsight_api.models import RequestContext

        top_k = kwargs.get("top_k", 200)
        question_id = kwargs.get("question_id", "")
        question_date = kwargs.get("question_date")
        max_tokens = kwargs.get("max_tokens", 4096)

        bank_id = self._get_bank_id(conversation_id)
        memory = await self._get_memory()
        budget = self._map_budget(self.budget_str)

        for attempt in range(self.max_retries):
            try:
                recall_result = await memory.recall_async(
                    bank_id=bank_id,
                    query=query,
                    budget=budget,
                    max_tokens=max_tokens,
                    question_date=question_date,
                    include_entities=True,
                    max_entity_tokens=2048,
                    include_chunks=True,
                    request_context=RequestContext(),
                )

                # Convert RecallResult to SearchResult (matching benchmark_runner.py pattern)
                results = recall_result.results or []

                # Extract entities and chunks for richer context
                entities_dict = {}
                if recall_result.entities:
                    for entity_name, entity_state in recall_result.entities.items():
                        entities_dict[entity_name] = entity_state.model_dump() if hasattr(entity_state, 'model_dump') else str(entity_state)

                chunks_dict = {}
                if recall_result.chunks:
                    for chunk_key, chunk_info in recall_result.chunks.items():
                        chunks_dict[chunk_key] = chunk_info.model_dump() if hasattr(chunk_info, 'model_dump') else str(chunk_info)

                normalized = []
                for r in results:
                    # MemoryFact uses 'text' for content and 'scores.final' for score
                    text = getattr(r, "text", "") or str(r)
                    scores = getattr(r, "scores", None)
                    score = scores.final if scores else 0.0

                    # Use model_dump() to preserve ALL fields from MemoryFact, matching benchmark behavior
                    if hasattr(r, 'model_dump'):
                        fact_info = r.model_dump()
                    else:
                        # Fallback: manually extract known fields
                        fact_info = {
                            "id": getattr(r, "id", "") or "",
                            "text": text,
                            "fact_type": getattr(r, "fact_type", "") or "",
                            "entities": getattr(r, "entities", None) or [],
                            "context": getattr(r, "context", "") or "",
                            "occurred_start": getattr(r, "occurred_start", "") or "",
                            "occurred_end": getattr(r, "occurred_end", "") or "",
                            "chunk_id": getattr(r, "chunk_id", "") or "",
                        }

                    entry = RetrievedMemory(
                        content=text,
                        score=score,
                        metadata=fact_info
                    )
                    normalized.append(entry)

                # Sort by score descending and limit to top_k
                normalized.sort(key=lambda x: x.score, reverse=True)
                normalized = normalized[:top_k]

                return SearchResult(
                    question_id=question_id,
                    query=query,
                    conversation_id=conversation_id,
                    results=normalized,
                    retrieval_metadata={
                        "adapter": "hindsight",
                        "bank_id": bank_id,
                        "total_results": len(normalized),
                        "budget": self.budget_str,
                        # Store full context for answer generation (matching benchmark_runner.py)
                        "entities": entities_dict,
                        "chunks": chunks_dict,
                        "trace": recall_result.trace.model_dump() if recall_result.trace and hasattr(recall_result.trace, 'model_dump') else (recall_result.trace or {}),
                    }
                )

            except Exception as e:
                logger.warning(
                    "SEARCH attempt %d/%d failed (bank_id=%s): %s",
                    attempt + 1, self.max_retries, bank_id, str(e)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "SEARCH failed after %d attempts for bank_id=%s",
                        self.max_retries, bank_id
                    )
                    return SearchResult(
                        question_id=question_id,
                        query=query,
                        conversation_id=conversation_id,
                        results=[],
                        retrieval_metadata={"adapter": "hindsight", "error": str(e)},
                    )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """Generate answer using LLM given query and retrieved context.

        Args:
            query: Question text
            context: Formatted retrieved context (fallback)
            conversation_id: Conversation ID
            **kwargs: Extra parameters including search_result (SearchResult object)

        Returns:
            Generated answer string
        """
        import json
        import pydantic

        class QuestionAnswer(pydantic.BaseModel):
            """Answer format for questions."""
            answer: str
            reasoning: str

        llm_config = await self._get_llm_config()

        if not llm_config.api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        # Build rich context matching benchmark_runner.py pattern
        # Convert SearchResult to RecallResult-like structure with full metadata
        search_result = kwargs.get("search_result")
        if search_result:
            # Build results list with full metadata preserved
            results_list = []
            for r in search_result.results:
                # Start with metadata dict for full fact info
                result_dict = dict(r.metadata) if r.metadata else {}
                # Ensure content and score are at result level (matching RecallResult)
                result_dict["text"] = r.content
                result_dict["score"] = r.score
                # Include entities if present in metadata
                if "entities" not in result_dict:
                    result_dict["entities"] = []
                results_list.append(result_dict)

            recall_dict = {
                # Top-level fields matching RecallResult structure
                "query": search_result.query,
                "total_results": search_result.retrieval_metadata.get("total_results", len(results_list)),
                # Results and context matching RecallResult.model_dump()
                "results": results_list,
                "entities": search_result.retrieval_metadata.get("entities", {}),
                "chunks": search_result.retrieval_metadata.get("chunks", {}),
                "trace": search_result.retrieval_metadata.get("trace", {}),
            }
            context = json.dumps(recall_dict, indent=2, ensure_ascii=False)
        else:
            # Fallback to pre-formatted context string
            pass

        # Prompt matching LoComo benchmark style (from benchmark_runner.py / locomo_benchmark.py)
        prompt = f"""You are a helpful expert assistant answering questions from lme_experiment users based on the provided context.

# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information or multiple instances of an event, say them all
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. If the answer is not explicitly stated in the memories, use logical reasoning based on the information available to answer (e.g. calculate duration of an event from different memories).

Context:

{context}

Question: {query}
Answer:"""

        for attempt in range(self.max_retries):
            try:
                answer_obj = await llm_config.call(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful expert assistant answering questions based on the provided context.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    response_format=QuestionAnswer,
                    scope="memory",
                )
                return answer_obj.answer
            except Exception as exc:
                logger.warning(
                    "Answer attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "Answer generation failed after %d attempts",
                        self.max_retries
                    )
                    return f"Error generating answer: {str(exc)[:100]}"
        return "Error generating answer"

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Hindsight",
            "type": "memory_engine",
            "description": "Hindsight Agent Memory System (direct MemoryEngine)",
            "adapter": "HindsightAdapter",
            "db_url": self.db_url,
            "budget": self.budget_str,
        }
