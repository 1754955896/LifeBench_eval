"""
Graphiti Local Adapter for LifeBench_eval.

Directly connects to Neo4j and uses Graphiti core library directly,
bypassing the HTTP API queue worker for synchronous processing.
This provides faster response times and avoids queue processing delays.
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


def _to_ascii_safe(text: str) -> str:
    """Convert string to ASCII-safe group_id for neo4j."""
    return f"g_{hashlib.md5(text.encode('utf-8')).hexdigest()}"


@register_adapter("graphiti_local")
class GraphitiLocalAdapter(BaseAdapter):
    """Graphiti local adapter using direct Neo4j connection.

    This adapter bypasses the HTTP API and uses Graphiti core directly,
    providing synchronous processing without queue delays.

    Configuration:
        neo4j_uri: Neo4j bolt URL (default: bolt://localhost:7687)
        neo4j_user: Neo4j username (default: neo4j)
        neo4j_password: Neo4j password (default: password)
        llm_config: LLM configuration dict
        embedder_config: Embedder configuration dict
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        # Neo4j connection settings
        self.neo4j_uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = config.get("neo4j_user", "neo4j")
        self.neo4j_password = config.get("neo4j_password", "password")

        # LLM config
        self.llm_config = config.get("llm", {})

        # Embedder config
        self.embedder_config = config.get("embedder", {})

        # Graphiti client (initialized on first use)
        self._graphiti = None
        self._driver = None

    def _get_llm_client(self):
        """Create LLM client for DeepSeek or other OpenAI-compatible APIs."""
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.llm_client.config import LLMConfig
        from openai import AsyncOpenAI
        import httpx

        base_url = self.llm_config.get("base_url", "https://api.deepseek.com/v1")
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        api_key = self.llm_config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))

        # Create httpx client with timeout
        http_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(60.0, connect=30.0),
        )

        llm_config = LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=self.llm_config.get("model", "deepseek-chat"),
            temperature=0,
        )

        # Use OpenAIGenericClient for DeepSeek since it doesn't support responses.parse API
        # json_object mode is required for DeepSeek
        return OpenAIGenericClient(
            config=llm_config,
            structured_output_mode="json_object",
            client=http_client,
        )

    def _get_embedder_client(self):
        """Create embedder client using VECTORIZE config from .env."""
        from graphiti_core.embedder import OpenAIEmbedder
        from graphiti_core.embedder.openai import OpenAIEmbedderConfig

        # Use VECTORIZE_* env vars if available, otherwise fallback to embedder config
        embedder_key = os.environ.get("VECTORIZE_API_KEY", self.embedder_config.get("api_key", ""))
        embedder_base = os.environ.get("VECTORIZE_BASE_URL", self.embedder_config.get("base_url", ""))
        embedder_model = os.environ.get("VECTORIZE_MODEL", self.embedder_config.get("model", ""))
        embedder_dim = int(os.environ.get("VECTORIZE_DIMENSIONS", self.embedder_config.get("dimension", "1024")))

        if not embedder_key or not embedder_base:
            return None

        if not embedder_base.endswith("/v1"):
            embedder_base = embedder_base.rstrip("/") + "/v1"

        embedder_config = OpenAIEmbedderConfig(
            api_key=embedder_key,
            base_url=embedder_base,
            embedding_model=embedder_model or "text-embedding-3-small",
            embedding_dim=embedder_dim,
        )

        return OpenAIEmbedder(config=embedder_config)

    async def _get_graphiti(self):
        """Get or create Graphiti client."""
        if self._graphiti is None:
            from graphiti_core.graphiti import Graphiti

            llm_client = self._get_llm_client()
            embedder_client = self._get_embedder_client()

            self._graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder_client,
            )

        return self._graphiti

    async def close(self) -> None:
        """Cleanup resources."""
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via Graphiti core directly.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        total_added = 0
        total_failed = 0

        graphiti = await self._get_graphiti()

        for chunk in chunks:
            if not chunk.messages:
                continue

            group_id = _to_ascii_safe(chunk.conversation_id)
            logger.info(f"ADD: group={chunk.conversation_id} -> {group_id}, messages={len(chunk.messages)}")

            # Process each message as an episode
            for msg in chunk.messages:
                # Build episode name from content preview
                content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                episode_name = f"{msg.speaker_name}: {content_preview}"

                # Convert timestamp to datetime
                reference_time = None
                if msg.timestamp:
                    if isinstance(msg.timestamp, datetime):
                        reference_time = msg.timestamp
                    else:
                        reference_time = datetime.fromtimestamp(msg.timestamp, tz=timezone.utc)
                else:
                    # Use current time as fallback
                    reference_time = datetime.now(timezone.utc)

                try:
                    result = await graphiti.add_episode(
                        group_id=group_id,
                        name=episode_name,
                        episode_body=msg.content,
                        reference_time=reference_time,
                        source_description=f"conversation {chunk.conversation_id}",
                    )
                    logger.debug(f"ADD episode result: type={type(result)}, result={result}")
                    total_added += 1
                except Exception as e:
                    logger.warning(f"ADD episode failed: {type(e).__name__}: {e}")
                    import traceback
                    logger.warning(f"ADD episode traceback: {traceback.format_exc()[:500]}")
                    total_failed += 1

        return {
            "type": "graphiti_local",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
        }

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via Graphiti core.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as group_id)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories
        """
        top_k = kwargs.get("top_k", 10)

        graphiti = await self._get_graphiti()
        group_id = _to_ascii_safe(conversation_id)

        logger.info(f"SEARCH: query={query[:50]}..., group={group_id}, top_k={top_k}")

        try:
            result = await graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=top_k,
            )

            # Convert Graphiti results to SearchResult
            facts = []
            if result and isinstance(result, list):
                for r in result:
                    if hasattr(r, "fact") and r.fact:
                        facts.append(
                            RetrievedMemory(
                                content=r.fact,
                                score=getattr(r, "score", 1.0),
                                metadata={
                                    "uuid": getattr(r, "uuid", ""),
                                    "name": getattr(r, "name", ""),
                                    "valid_at": getattr(r, "valid_at", None),
                                    "invalid_at": getattr(r, "invalid_at", None),
                                },
                            )
                        )
                    elif hasattr(r, "name") and r.name:
                        facts.append(
                            RetrievedMemory(
                                content=r.name,
                                score=getattr(r, "score", 1.0),
                                metadata={
                                    "uuid": getattr(r, "uuid", ""),
                                    "name": getattr(r, "name", ""),
                                },
                            )
                        )

            logger.info(f"SEARCH: found {len(facts)} facts")

            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=facts,
                retrieval_metadata={
                    "adapter": "graphiti_local",
                    "total_results": len(facts),
                },
            )

        except Exception as e:
            logger.error(f"SEARCH failed: {e}")
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"adapter": "graphiti_local", "error": str(e)},
            )

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
        model = llm_config.get("model", "deepseek-chat")
        api_key = llm_config.get("api_key", "")
        base_url = llm_config.get("base_url", "https://api.deepseek.com/v1")
        temperature = llm_config.get("temperature", 0)
        max_tokens = llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        reference_date = kwargs.get("reference_date", "2023")

        # LOCOMO 7-step reasoning prompt (same as graphiti adapter)
        prompt = f"""You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity.

## Step 3: COMBINE AND CROSS-REFERENCE
Combine facts from multiple memories about the same topic.

## Step 4: SELECT THE BEST ANSWER
Choose the MOST SPECIFIC detail available.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.

## Step 6: INCLUSION CHECK
If you found items during reasoning that you're tempted to exclude — include them unless you have STRONG evidence they are wrong.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified" or "no record" — if ANY memory contains relevant information, give the best answer.

{context if context else "(No relevant memories found)"}

Question: {query}

Work through Steps 1-7, then give your final answer after "ANSWER:". """

        import aiohttp

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
