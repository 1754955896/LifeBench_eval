"""
Hindsight Adapter for LifeBench_eval.

Uses hindsight_client library to connect to Hindsight server (docker).
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


@register_adapter("hindsight")
class HindsightAdapter(BaseAdapter):
    """Hindsight adapter using hindsight_client library.

    Configuration:
        base_url: Server URL (default http://localhost:8888)
        bank_id: Bank ID prefix (default "default")
        max_retries: Maximum retry attempts (default 5)
        retry_delay: Base delay in seconds between retries (default 2.0)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        self.base_url = config.get("base_url", "http://localhost:8888")
        self.bank_id = config.get("bank_id", "default")
        self.max_retries = config.get("max_retries", 5)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.budget = config.get("budget", "mid")  # mid=~100 candidates vs high=300

        self._client = None

    def _get_bank_id(self, conversation_id: str) -> str:
        """Get bank_id for a conversation."""
        return f"{self.bank_id}_{conversation_id}"

    async def _get_client(self):
        """Get or create Hindsight client."""
        if self._client is None:
            from hindsight_client import Hindsight
            self._client = Hindsight(base_url=self.base_url)
        return self._client

    async def close(self) -> None:
        """Cleanup resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via Hindsight API.

        Merges all chunks per conversation_id into a single content string
        so the server can batch-process them efficiently.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        total_added = 0
        total_failed = 0

        client = await self._get_client()

        # Group all chunks by conversation_id
        conv_contents: Dict[str, List[str]] = {}
        for chunk in chunks:
            if not chunk.messages:
                continue
            content = "\n".join(f"{msg.speaker_name}: {msg.content}" for msg in chunk.messages)
            if content.strip():
                conv_contents.setdefault(chunk.conversation_id, []).append(content)

        # One aretain call per conversation (all messages merged)
        for conversation_id, content_list in conv_contents.items():
            merged_content = "\n---\n".join(content_list)
            bank_id = self._get_bank_id(conversation_id)

            try:
                await client.aretain(bank_id=bank_id, content=merged_content)
                total_added += 1
            except Exception as e:
                logger.warning(f"ADD attempt failed (bank_id={bank_id}): {e}")
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
        """Search memories via Hindsight API.

        Args:
            query: Query text
            conversation_id: Conversation ID (used to derive bank_id)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories
        """
        top_k = kwargs.get("top_k", 200)
        question_id = kwargs.get("question_id", "")

        bank_id = self._get_bank_id(conversation_id)
        client = await self._get_client()

        for attempt in range(self.max_retries):
            try:
                recall_result = await client.arecall(
                    bank_id=bank_id,
                    query=query,
                    budget=self.budget,
                )
                results = recall_result.results or []

                # Convert to RetrievedMemory
                normalized = []
                for r in results:
                    entry = RetrievedMemory(
                        content=getattr(r, "content", "") or str(r),
                        score=getattr(r, "score", 0) or 0,
                        metadata={
                            "id": getattr(r, "id", "") or "",
                            "bank_id": bank_id,
                        }
                    )
                    normalized.append(entry)

                # Sort by score descending
                normalized.sort(key=lambda x: x.score, reverse=True)

                # Limit to top_k
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

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Hindsight",
            "type": "online_api",
            "description": "Hindsight Agent Memory System",
            "adapter": "HindsightAdapter",
        }

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
        import aiohttp

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

        # Prompt from Hindsight LoComo benchmark
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

        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful expert assistant answering questions based on the provided context."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries):
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
                logger.warning("Answer attempt %d/%d failed: %s", attempt + 1, self.max_retries, str(exc)[:200])
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Answer generation failed after %d attempts", self.max_retries)
                    return f"Error generating answer: {str(exc)[:100]}"
        return "Error generating answer"
