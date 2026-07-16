"""
Cognee Adapter for LifeBench_eval.

Connects to Cognee Docker container via REST API to provide
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


@register_adapter("cognee")
class CogneeAdapter(BaseAdapter):
    """Cognee adapter using REST API.

    Configuration:
        host: Cognee API URL (default: http://localhost:8000)
        api_key: Optional API key for authentication
        timeout: HTTP request timeout in seconds (default 300.0)
        max_retries: Maximum retry attempts (default 5)
        retry_delay: Base delay in seconds between retries (default 5.0)
        top_k: Default number of results to return (default 40)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        self.host = config.get("host", "http://localhost:8000")
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 300.0)
        self.max_retries = config.get("max_retries", 5)
        self.retry_delay = config.get("retry_delay", 5.0)

        # Handle nested search config (search.top_k, search.search_type)
        search_config = config.get("search", {})
        self.top_k = search_config.get("top_k", config.get("top_k", 40))
        self.search_type = search_config.get("search_type", config.get("search_type", "CHUNKS"))

        # Concurrency control: limit concurrent requests to avoid overload
        self.max_concurrent = config.get("max_concurrent", 3)
        self._semaphore: Optional[asyncio.Semaphore] = None

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

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
        """Ingest message chunks via Cognee API.

        Each ChunkedMessage becomes a separate memory entry in Cognee.
        After all chunks are added, calls cognify to build the knowledge graph.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        total_added = 0
        total_failed = 0
        session = await self._get_session()

        # Track datasets that need cognify
        datasets_to_cognify = set()

        for chunk in chunks:
            if not chunk.messages:
                continue

            # Combine all messages into a single text content
            content_parts = []
            for msg in chunk.messages:
                speaker = msg.speaker_name or msg.speaker_id or "User"
                content_parts.append(f"{speaker}: {msg.content}")

            content = "\n".join(content_parts)
            dataset_name = chunk.conversation_id or "default"

            success = await self._add_memory(session, content, dataset_name)
            if success:
                total_added += 1
                datasets_to_cognify.add(dataset_name)
            else:
                total_failed += 1

        # Now cognify each dataset to build the knowledge graph
        for dataset_name in datasets_to_cognify:
            logger.info(f"Cognifying dataset: {dataset_name}")
            await self._cognify(session, dataset_name)
            # Wait for async processing to complete
            await asyncio.sleep(2)

        return {
            "type": "cognee",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
        }

    async def _add_memory(
        self, session: aiohttp.ClientSession, content: str, dataset_name: str
    ) -> bool:
        """Add a single memory via Cognee API.

        Args:
            session: aiohttp session
            content: Text content to remember
            dataset_name: Dataset name for this memory

        Returns:
            True if successful, False otherwise
        """
        semaphore = await self._get_semaphore()
        async with semaphore:
            api_base = f"{self.host}/api/v1"

            # Use remember API - single text input
            form = aiohttp.FormData()
            form.add_field("datasetName", dataset_name)
            form.add_field(
                "data",
                content.encode("utf-8"),
                filename="memory.txt",
                content_type="text/plain",
            )

            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{api_base}/add",
                        data=form,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status in (200, 201, 204):
                            return True
                        # On failure, try cognify anyway (dataset may already exist)
                        if attempt == 0:
                            logger.debug(f"Add response: {resp.status}")

                except Exception as exc:
                    logger.warning(
                        "ADD attempt %d/%d failed (dataset=%s): %s",
                        attempt + 1, self.max_retries, dataset_name, str(exc)[:200]
                    )
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 2, 4, 8, 16 seconds
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))

            return False

    async def _cognify(self, session: aiohttp.ClientSession, dataset_name: str) -> bool:
        """Build knowledge graph for dataset via Cognee API.

        Args:
            session: aiohttp session
            dataset_name: Dataset name to cognify

        Returns:
            True if successful, False otherwise
        """
        semaphore = await self._get_semaphore()
        async with semaphore:
            api_base = f"{self.host}/api/v1"

            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{api_base}/cognify",
                        json={"datasets": [dataset_name]},
                        timeout=aiohttp.ClientTimeout(total=self.timeout * 2),
                    ) as resp:
                        if resp.status in (200, 201):
                            # Wait for async processing
                            await asyncio.sleep(5)
                            return True
                        logger.warning(f"Cognify response: {resp.status}")

                except Exception as exc:
                    logger.warning(
                        "COGNIFY attempt %d/%d failed (dataset=%s): %s",
                        attempt + 1, self.max_retries, dataset_name, str(exc)[:200]
                    )
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 2, 4, 8, 16 seconds
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))

            return False

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via Cognee API.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as dataset_name)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories
        """
        semaphore = await self._get_semaphore()
        async with semaphore:
            top_k = kwargs.get("top_k", self.top_k)
            session = await self._get_session()
            api_base = f"{self.host}/api/v1"

            # Use conversation_id as dataset_name
            dataset_name = conversation_id or "default"

            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{api_base}/search",
                        json={
                            "query": query,
                            "datasets": [dataset_name],
                            "top_k": top_k,
                            "search_type": self.search_type,
                        },
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            results = data if isinstance(data, list) else data.get("results", [])

                            normalized = []
                            for r in results:
                                if isinstance(r, dict):
                                    content = r.get("text", r.get("content", str(r)))
                                    score = r.get("score", 0.0)
                                    metadata = r.get("metadata", {})
                                else:
                                    content = str(r)
                                    score = 0.0
                                    metadata = {}

                                normalized.append(
                                    RetrievedMemory(
                                        content=content,
                                        score=score,
                                        metadata=metadata,
                                    )
                                )

                            return SearchResult(
                                question_id=kwargs.get("question_id", ""),
                                query=query,
                                conversation_id=conversation_id,
                                results=normalized,
                                retrieval_metadata={
                                    "adapter": "cognee",
                                    "dataset": dataset_name,
                                    "total_results": len(normalized),
                                }
                            )

                        logger.warning(f"Search response: {resp.status}")

                except Exception as exc:
                    logger.warning(
                        "SEARCH attempt %d/%d failed (dataset=%s): %s",
                        attempt + 1, self.max_retries, dataset_name, str(exc)[:200]
                    )
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 2, 4, 8, 16 seconds
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))

            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"adapter": "cognee", "dataset": dataset_name}
            )

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
        model = llm_config.get("model", "deepseek-v4-flash")
        api_key = llm_config.get("api_key", "")
        base_url = llm_config.get("base_url", "https://api.deepseek.com")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.warning("No LLM API key configured for answer generation")
            return context

        prompt = f"""Based on the following retrieved memories, answer the question.

Memories:
{context}

Question: {query}

Answer:"""

        if provider == "openai" or "deepseek" in base_url.lower():
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
        else:
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
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"]
            return str(data)
        except Exception as exc:
            logger.error("Answer generation failed: %s", str(exc)[:200])
            return context
