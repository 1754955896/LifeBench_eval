"""
Graphiti Local Adapter for LifeBench_eval.

Directly connects to Neo4j and uses Graphiti core library,
bypassing the HTTP API for synchronous processing.
Follows the same add_episode / search patterns as Graphiti's own
LongMemEval evaluation in tests/evals/eval_e2e_graph_building.py.
"""

import asyncio
import hashlib
import logging
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import httpx

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import RetrievedMemory, SearchResult

logger = logging.getLogger(__name__)


def _safe_group_id(conversation_id: str) -> str:
    """Derive an ASCII-safe Neo4j group_id from a conversation_id."""
    return f"g_{hashlib.md5(conversation_id.encode('utf-8')).hexdigest()}"


@register_adapter("graphiti_local")
class GraphitiLocalAdapter(BaseAdapter):
    """Graphiti local adapter using direct Neo4j connection.

    Mirrors the add_episode pattern from Graphiti's LongMemEval eval script:
      episode_body = '{speaker}: {content}'
      graphiti.add_episode(name='', episode_body=..., reference_time=...,
                           source=EpisodeType.message, group_id=...)

    Configuration:
        neo4j_uri:      Neo4j bolt URL  (default: bolt://localhost:7687)
        neo4j_user:     Neo4j username  (default: neo4j)
        neo4j_password: Neo4j password  (default: password)
        llm:            LLM config dict with api_key, base_url, model
        embedder:       Embedder config dict with api_key, base_url, model, dimension
        top_k:          Default search result count (default: 10)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        # Neo4j
        self.neo4j_uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = config.get("neo4j_user", "neo4j")
        self.neo4j_password = config.get("neo4j_password", "password")

        # LLM
        self.llm_config = config.get("llm", {})

        # Embedder (optional — graphiti can run without one, falling back to BM25-only search)
        self.embedder_config = config.get("embedder", {})

        # Reranker / cross-encoder
        self.rerank_config = config.get("rerank", {})

        # Search defaults
        self.top_k = config.get("search", {}).get("top_k", 10)

        # Batch size for add_episode (default 10)
        self.batch_size = max(1, int(config.get("batch_size", 10)))

        # Combined extraction: single LLM call per episode for nodes + edges
        self.use_combined_extraction = bool(config.get("use_combined_extraction", False))

        # LLM proxy URL — when set, LLM chat traffic is routed via proxy
        self.llm_proxy_url = config.get("llm_proxy_url", "").strip()

        # Max concurrent coroutines for graphiti operations (overrides SEMAPHORE_LIMIT)
        self.max_coroutines = config.get("max_coroutines", None)
        if self.max_coroutines is not None:
            self.max_coroutines = max(1, int(self.max_coroutines))

        # Per-conversation reference date for answer() temporal grounding
        self._conversation_reference_date: Dict[str, str] = {}

        # Lazy init — one Graphiti instance per group_id to avoid race conditions
        # when multiple samples run concurrently and share the same adapter.
        self._graphiti_by_group: Dict[str, Any] = {}
        self._graphiti_init_lock = asyncio.Lock()

        # Shared Neo4j driver — each Graphiti instance would otherwise create its
        # own driver with its own connection pool, overwhelming Neo4j when many
        # samples run concurrently. A single shared driver with one pool is safe
        # because Graphiti.clone() is a no-op (returns self).
        self._shared_driver: Any = None
        self._shared_driver_lock = asyncio.Lock()

        # Shared httpx.AsyncClient — each Graphiti instance would otherwise create
        # its own httpx client for the LLM provider, each with its own connection
        # pool. With 10 concurrent samples, this creates 10 independent connection
        # pools that together overwhelm the local TCP stack. A single shared client
        # with one pooled connection pool avoids port exhaustion.
        self._shared_http_client: Any = None
        self._shared_http_client_lock = asyncio.Lock()

        # Multi-key LLM client cache: key_index → OpenAIGenericClient
        self._llm_client_cache: dict[int, Any] = {}
        self._llm_keys: list[str] = []  # populated lazily
        self._key_rr: int = 0  # round-robin counter for key distribution

    # ------------------------------------------------------------------
    # LLM / Embedder / Reranker helpers (independently configurable)
    # ------------------------------------------------------------------

    async def _get_shared_http_client(self):
        """Return the single shared httpx.AsyncClient for all Graphiti instances.

        One connection pool shared across all concurrent samples avoids local
        TCP port exhaustion and connection-pool thrashing under high concurrency.
        Pool limits are raised to handle 200+ concurrent LLM requests (10 samples
        × 20 max_coroutines); 500 connections, 100 keepalive.
        """
        if self._shared_http_client is None:
            async with self._shared_http_client_lock:
                if self._shared_http_client is None:
                    import httpx
                    _limits = httpx.Limits(
                        max_connections=500,
                        max_keepalive_connections=100,
                    )
                    _timeout = httpx.Timeout(
                        connect=60.0, read=1200.0, write=1200.0, pool=1200.0,
                    )
                    self._shared_http_client = httpx.AsyncClient(
                        timeout=_timeout, limits=_limits,
                    )
        return self._shared_http_client

    def _get_llm_client(self, key_index: int = 0):
        """Create or return a cached LLM client for *key_index*.

        When multiple API keys are configured (via comma-separated string
        or YAML list), each key gets its own client. Conversations are
        pinned to a key by hashing group_id, distributing concurrent
        requests across keys and bypassing per-key rate limits.
        """
        if key_index in self._llm_client_cache:
            return self._llm_client_cache[key_index]

        from openai import AsyncOpenAI
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        # Lazy-init key list on first call
        if not self._llm_keys:
            self._llm_keys = self._resolve_api_keys("LLM")
            if not self._llm_keys:
                raise RuntimeError('No LLM API key configured')
            logger.info("LLM multi-key: %d key(s) available", len(self._llm_keys))
            print(f"[graphiti_local] Multi-key enabled: {len(self._llm_keys)} API keys")

        key_index = key_index % len(self._llm_keys)
        if key_index in self._llm_client_cache:
            return self._llm_client_cache[key_index]

        api_key = self._llm_keys[key_index]
        base_url = self._resolve_base_url("LLM")
        model = self.llm_config.get("model", os.environ.get("LLM_MODEL", "deepseek-chat"))

        if self.llm_proxy_url:
            base_url = self.llm_proxy_url

        llm_cfg = LLMConfig(api_key=api_key, base_url=base_url, model=model, temperature=0)

        _http_client = self._shared_http_client
        if _http_client is None:
            raise RuntimeError('Shared HTTP client not initialized')

        _timeout = httpx.Timeout(connect=60.0, read=1200.0, write=1200.0, pool=1200.0)
        if "openai.com" in base_url:
            async_client = AsyncOpenAI(
                api_key=api_key, base_url=base_url, http_client=_http_client,
                timeout=_timeout,
            )
            client = OpenAIClient(config=llm_cfg, client=async_client)
        else:
            async_client = AsyncOpenAI(
                api_key=api_key, base_url=base_url, http_client=_http_client,
                timeout=_timeout,
            )
            client = OpenAIGenericClient(
                config=llm_cfg, client=async_client, structured_output_mode="json_object",
                max_tokens=131072,
            )

        self._llm_client_cache[key_index] = client
        logger.info("LLM client key[%d/%d]: model=%s base=%s",
                     key_index, len(self._llm_keys), model, base_url)
        return client

    def _get_embedder_client(self):
        """Create an embedder client from YAML config with env fallbacks."""
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        key = self._resolve_api_key("VECTORIZE")
        base = self._resolve_base_url("VECTORIZE")
        model = self.embedder_config.get("model") or os.environ.get("VECTORIZE_MODEL", "")
        dim = int(
            self.embedder_config.get("dimension")
            or os.environ.get("VECTORIZE_DIMENSIONS", "1024")
        )
        provider = (
            self.embedder_config.get("provider")
            or os.environ.get("VECTORIZE_PROVIDER", "unknown")
        )

        if not key or not base:
            logger.warning("No embedder configured (VECTORIZE_API_KEY / VECTORIZE_BASE_URL) — "
                           "search will use BM25 only (no vector similarity)")
            return None

        if not base.endswith("/v1"):
            base = base.rstrip("/") + "/v1"

        cfg = OpenAIEmbedderConfig(
            api_key=key,
            base_url=base,
            embedding_model=model or "text-embedding-3-small",
            embedding_dim=dim,
        )
        logger.info("Embedder: provider=%s model=%s dim=%d",
                     provider, cfg.embedding_model, cfg.embedding_dim)
        return OpenAIEmbedder(config=cfg)

    def _get_cross_encoder(self):
        """Create a cross-encoder client from config or RERANK_* env vars.

        Reads from ``self.rerank_config`` (populated from the system YAML) first,
        falling back to ``os.environ`` for backwards compatibility.  Returns None
        to disable cross-encoder reranking when neither source is configured.
        """
        from graphiti_core.cross_encoder.openai_reranker_client import (
            OpenAIRerankerClient,
        )
        from graphiti_core.llm_client.config import LLMConfig

        # Resolve values: config ＞ env var ＞ default
        provider = (
            self.rerank_config.get("provider")
            or os.environ.get("RERANK_PROVIDER", "")
        ).strip().lower()
        api_key = (
            self.rerank_config.get("api_key")
            or self._resolve_api_key("RERANK")
        )
        base_url = (
            self.rerank_config.get("base_url")
            or os.environ.get("RERANK_BASE_URL", "")
        ).strip().rstrip("/")
        model = (
            self.rerank_config.get("model")
            or os.environ.get("RERANK_MODEL", "")
        ).strip()

        if not provider or not api_key:
            logger.info("No RERANK_PROVIDER configured — cross-encoder reranking disabled")
            return None

        # Dedicated rerank API (SiliconFlow, Jina AI, Cohere, etc.)
        if base_url.endswith("/rerank") or base_url.endswith("/rerank/v1"):
            logger.info("Reranker: dedicated API provider=%s model=%s base=%s",
                         provider, model, base_url)
            from graphiti_core.cross_encoder.client import CrossEncoderClient

            class APIRerankerClient(CrossEncoderClient):
                def __init__(self, api_key, base_url, model=""):
                    self.api_key = api_key
                    self.base_url = base_url
                    self.model = model

                async def rank(self, query, passages):
                    if not passages:
                        return []
                    payload = {
                        "query": query,
                        "documents": passages,
                        "top_n": len(passages),
                    }
                    if self.model:
                        payload["model"] = self.model
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    }
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                self.base_url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=30),
                            ) as resp:
                                if resp.status >= 500:
                                    raise aiohttp.ClientResponseError(
                                        resp.request_info, resp.history, status=resp.status)
                                resp.raise_for_status()
                                data = await resp.json()
                        results = data.get("results", [])
                        scored = []
                        for r in results:
                            idx = r.get("index", 0)
                            score = r.get("relevance_score", 0.0)
                            if 0 <= idx < len(passages):
                                scored.append((passages[idx], float(score)))
                        scored.sort(key=lambda x: x[1], reverse=True)
                        return scored
                    except Exception:
                        logger.warning("Reranker API call failed, returning passages unscored",
                                       exc_info=True)
                        return [(p, 1.0) for p in passages]

            return APIRerankerClient(api_key=api_key, base_url=base_url, model=model)

        # OpenAI-compatible chat API with logprobs (use LLM for reranking)
        logger.info("Reranker: chat-API provider=%s model=%s base=%s",
                     provider, model, base_url)
        return OpenAIRerankerClient(
            config=LLMConfig(api_key=api_key, base_url=base_url + "/v1", model=model)
        )

    # ------------------------------------------------------------------
    # Config resolution helpers
    # ------------------------------------------------------------------

    def _resolve_api_key(self, prefix: str) -> str:
        """Resolve the first API key (backward-compatible single-key path)."""
        keys = self._resolve_api_keys(prefix)
        return keys[0] if keys else ""

    def _resolve_api_keys(self, prefix: str) -> list[str]:
        """Resolve API keys as a list for LLM multi-key load distribution.

        Priority: env var {PREFIX}_API_KEY → YAML config api_key field
        Supports: comma-separated string, YAML list, single string.
        """
        if prefix != "LLM":
            # Non-LLM services only need a single key
            env_val = os.environ.get(f"{prefix}_API_KEY", "").strip()
            if env_val and env_val != "EMPTY":
                return [env_val]
            if prefix == "VECTORIZE":
                cfg = self.embedder_config.get("api_key", "")
            elif prefix == "RERANK":
                cfg = self.rerank_config.get("api_key", "")
            else:
                cfg = ""
            return [cfg] if cfg else []

        # LLM: YAML config first, env vars as fallback
        cfg_val = self.llm_config.get("api_key", "")
        if isinstance(cfg_val, list):
            return [k.strip() for k in cfg_val if k.strip()]
        if isinstance(cfg_val, str) and cfg_val.strip():
            return [k.strip() for k in cfg_val.split(",") if k.strip()]

        env_val = os.environ.get("LLM_API_KEY", "").strip()
        if env_val and env_val != "EMPTY":
            return [k.strip() for k in env_val.split(",") if k.strip()]

        # Final fallback: OPENAI_API_KEY
        fallback = os.environ.get("OPENAI_API_KEY", "")
        return [fallback] if fallback else []

    def _resolve_base_url(self, prefix: str) -> str:
        """Resolve base URL for a service prefix (LLM, VECTORIZE, RERANK).

        Returns a URL ending with /v1 for OpenAI-compatible chat/embedding APIs.
        For RERANK, returns the raw URL as-is (may end with /rerank, not /v1).
        """
        env_val = os.environ.get(f"{prefix}_BASE_URL", "").strip()
        if env_val:
            return env_val  # RERANK URLs are returned as-is

        if prefix == "LLM":
            base = self.llm_config.get("base_url", "https://api.deepseek.com/v1")
        elif prefix == "VECTORIZE":
            base = self.embedder_config.get("base_url", "")
        elif prefix == "RERANK":
            base = self.rerank_config.get("base_url", "")
        else:
            base = ""

        # RERANK URLs point to dedicated /rerank endpoints — don't append /v1
        if prefix == "RERANK":
            return base

        if base and not base.endswith("/v1"):
            base = base.rstrip("/") + "/v1"
        return base

    # ------------------------------------------------------------------
    # Lazy Graphiti client
    # ------------------------------------------------------------------

    async def _get_shared_driver(self):
        """Create or return the single shared Neo4j driver.

        All Graphiti instances share one connection pool to avoid overwhelming
        Neo4j when many samples run concurrently.
        """
        if self._shared_driver is None:
            async with self._shared_driver_lock:
                if self._shared_driver is None:
                    from graphiti_core.driver.neo4j_driver import Neo4jDriver

                    self._shared_driver = Neo4jDriver(
                        self.neo4j_uri, self.neo4j_user, self.neo4j_password,
                    )
        return self._shared_driver

    async def _get_graphiti(self, group_id: str):
        """Get or create a Graphiti client for *group_id* (lazy, one per group).

        When multiple samples run concurrently, each conversation maps to a
        different group_id.  Sharing a single Graphiti instance would cause a
        race condition because add_episode() mutates self.driver based on
        group_id.  We create one instance per group so each has its own driver
        permanently set to the right database.

        All instances share a single Neo4j driver via graph_driver= to keep
        one connection pool instead of one per instance.
        """
        if group_id not in self._graphiti_by_group:
            async with self._graphiti_init_lock:
                if group_id not in self._graphiti_by_group:
                    from graphiti_core.graphiti import Graphiti

                    # Ensure shared resources are initialized before creating
                    # the LLM client (which now depends on the shared HTTP client)
                    await self._get_shared_http_client()
                    shared_driver = await self._get_shared_driver()

                    # Round-robin key distribution: each _get_graphiti()
                    # call gets the next key, cycling through all keys.
                    # Same group across different days may use different keys
                    # but hash-based determinism isn't needed.
                    key_index = self._key_rr % 1024
                    self._key_rr += 1
                    llm_client = self._get_llm_client(key_index)
                    embedder = self._get_embedder_client()
                    cross_encoder = self._get_cross_encoder()

                    graphiti = Graphiti(
                        uri=self.neo4j_uri,
                        user=self.neo4j_user,
                        password=self.neo4j_password,
                        llm_client=llm_client,
                        embedder=embedder,
                        cross_encoder=cross_encoder,
                        max_coroutines=self.max_coroutines,
                        graph_driver=shared_driver,
                    )
                    self._graphiti_by_group[group_id] = graphiti

        return self._graphiti_by_group[group_id]

    async def close(self) -> None:
        for graphiti in self._graphiti_by_group.values():
            await graphiti.close()
        self._graphiti_by_group.clear()
        if self._shared_driver is not None:
            await self._shared_driver.close()
            self._shared_driver = None
        if self._shared_http_client is not None:
            await self._shared_http_client.aclose()
            self._shared_http_client = None

    # ------------------------------------------------------------------
    # Ingest (mirrors eval_e2e_graph_building.build_subgraph)
    # ------------------------------------------------------------------

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest messages via add_episode_bulk for combined extraction and batch dedup.

        Messages are grouped into batches (batch_size) and each batch becomes a
        RawEpisode. All episodes are processed in a single add_episode_bulk call,
        which enables cross-episode node/edge dedup and combined extraction
        (single LLM call per episode for nodes + edges).
        """
        from graphiti_core.nodes import EpisodeType
        from graphiti_core.utils.bulk_utils import RawEpisode

        total_added = 0
        total_failed = 0
        per_episode_metadata: list[dict] = []
        latest_ref_date: Dict[str, str] = {}

        # Pre-compute group_id from the first chunk (all belong to same conversation)
        group_id = ""
        for chunk in chunks:
            if chunk.messages:
                group_id = _safe_group_id(chunk.conversation_id)
                break

        if not group_id:
            return {
                "type": "graphiti_local",
                "total_chunks": len(chunks),
                "added": 0,
                "failed": 0,
                "metadata": [],
            }

        graphiti = await self._get_graphiti(group_id)

        for chunk in chunks:
            if not chunk.messages:
                continue

            # Collect valid messages (filter empty content, append image metadata)
            valid_msgs: list[tuple[Any, str, str, Any]] = []  # (msg, speaker, content, ts)
            for msg in chunk.messages:
                speaker = getattr(msg, "speaker_name", "") or "unknown"
                content = (getattr(msg, "content", "") or "").strip()
                content = self._append_image_metadata(msg, content)
                if not content:
                    continue
                ts = self._resolve_timestamp(
                    getattr(msg, "timestamp", None), chunk.timestamp,
                )
                valid_msgs.append((msg, speaker, content, ts))

            if not valid_msgs:
                continue

            # Track latest session date for answer() temporal grounding
            self._track_ref_date(chunk, latest_ref_date)

            # Build RawEpisode batches for add_episode_bulk
            raw_episodes: list[RawEpisode] = []
            batch_indices: list[tuple[int, int, datetime]] = []  # (count, ref_time)
            for batch_start in range(0, len(valid_msgs), self.batch_size):
                batch_msgs = valid_msgs[batch_start:batch_start + self.batch_size]
                lines = [f"{speaker}: {content}" for _, speaker, content, _ in batch_msgs]
                episode_body = "\n".join(lines)
                reference_time = batch_msgs[0][3]
                raw_episodes.append(RawEpisode(
                    name="",
                    content=episode_body,
                    source_description=f"conversation {chunk.conversation_id}",
                    source=EpisodeType.message,
                    reference_time=reference_time,
                ))
                batch_indices.append((len(batch_msgs), reference_time))

            if not raw_episodes:
                continue

            logger.info(
                "ADD: conversation=%s group=%s messages=%d episodes=%d batch_size=%d combined=%s",
                chunk.conversation_id, group_id, len(valid_msgs), len(raw_episodes),
                self.batch_size, self.use_combined_extraction,
            )

            for attempt in range(1, 3):
                try:
                    result = await graphiti.add_episode_bulk(
                        bulk_episodes=raw_episodes,
                        group_id=group_id,
                        use_combined_extraction=self.use_combined_extraction,
                    )
                    for i, ep in enumerate(result.episodes):
                        count, ref_time = batch_indices[i]
                        total_added += count
                        per_episode_metadata.append({
                            "batch_size": count,
                            "nodes_extracted": len(result.nodes),
                            "edges_extracted": len(result.edges),
                            "episode_uuid": ep.uuid,
                            "reference_time": ref_time.isoformat(),
                        })
                    break
                except Exception as exc:
                    delay = 5 * attempt + random.uniform(0, 5)
                    if attempt < 3:
                        logger.warning(
                            "ADD bulk failed (conversation=%s, episodes=%d, attempt=%d/3): %s — retrying in %ds",
                            chunk.conversation_id, len(raw_episodes), attempt,
                            str(exc)[:150], delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(
                            "ADD bulk failed (conversation=%s, episodes=%d): %s",
                            chunk.conversation_id, len(raw_episodes), str(exc)[:200],
                        )
                        total_failed += sum(c for c, _ in batch_indices)

        self._conversation_reference_date.update(latest_ref_date)

        return {
            "type": "graphiti_local",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
            "metadata": per_episode_metadata,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search the knowledge graph via Graphiti core.

        Uses the advanced search_() with cross-encoder hybrid search that covers
        all four graph layers — edges, entity nodes, episodes, and communities.
        Results from all layers are merged, scored, and truncated to top_k.
        """
        top_k = kwargs.get("top_k", self.top_k)

        group_id = _safe_group_id(conversation_id)
        graphiti = await self._get_graphiti(group_id)

        logger.info("SEARCH: query=%.50s... group=%s top_k=%d", query, group_id, top_k)

        try:
            from graphiti_core.search.search_config import (
                CommunityReranker,
                CommunitySearchConfig,
                CommunitySearchMethod,
                EdgeReranker,
                EdgeSearchConfig,
                EdgeSearchMethod,
                EpisodeReranker,
                EpisodeSearchConfig,
                EpisodeSearchMethod,
                NodeReranker,
                NodeSearchConfig,
                NodeSearchMethod,
                SearchConfig,
            )

            # Build a fresh config per call — never mutate the global singleton.
            # RRF (Reciprocal Rank Fusion) is pure math, zero LLM cost for reranking.
            limit = max(top_k, 30)
            config = SearchConfig(
                edge_config=EdgeSearchConfig(
                    search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                    reranker=EdgeReranker.rrf,
                ),
                node_config=NodeSearchConfig(
                    search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                    reranker=NodeReranker.rrf,
                ),
                episode_config=EpisodeSearchConfig(
                    search_methods=[EpisodeSearchMethod.bm25],
                    reranker=EpisodeReranker.rrf,
                ),
                community_config=CommunitySearchConfig(
                    search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
                    reranker=CommunityReranker.rrf,
                ),
                limit=limit,
            )
            result = await graphiti.search_(
                query=query,
                config=config,
                group_ids=[group_id],
            )

            # ---- collect scored items from all layers ----
            scored: list[tuple[float, str, dict]] = []  # (score, content, metadata)

            # 1. Edges: relationship facts (e.g. "Jon told Gina he lost his job")
            for i, edge in enumerate(result.edges):
                score = result.edge_reranker_scores[i] if i < len(result.edge_reranker_scores) else 1.0
                content = edge.fact or edge.name
                scored.append((float(score), content, {
                    "layer": "edge",
                    "uuid": edge.uuid,
                    "relation": edge.name,
                    "valid_at": str(edge.valid_at) if edge.valid_at else None,
                }))

            # 2. Entity nodes: per-person/entity summaries
            for i, node in enumerate(result.nodes):
                score = result.node_reranker_scores[i] if i < len(result.node_reranker_scores) else 1.0
                parts = [node.name, node.summary] if node.summary else [node.name]
                content = " — ".join(p for p in parts if p)
                scored.append((float(score), content, {
                    "layer": "entity",
                    "uuid": node.uuid,
                }))

            # 3. Episodes: raw conversation snippets
            for i, ep in enumerate(result.episodes):
                score = result.episode_reranker_scores[i] if i < len(result.episode_reranker_scores) else 1.0
                content = ep.content
                scored.append((float(score), content, {
                    "layer": "episode",
                    "uuid": ep.uuid,
                    "valid_at": str(ep.valid_at) if ep.valid_at else None,
                }))

            # 4. Communities: topic/theme summaries
            for i, com in enumerate(result.communities):
                score = result.community_reranker_scores[i] if i < len(result.community_reranker_scores) else 1.0
                parts = [com.name, com.summary] if com.summary else [com.name]
                content = " — ".join(p for p in parts if p)
                scored.append((float(score), content, {
                    "layer": "community",
                    "uuid": com.uuid,
                }))

            # ---- deduplicate by content, keeping highest score ----
            seen: set[str] = set()
            unique: list[tuple[float, str, dict]] = []
            for score, content, meta in sorted(scored, key=lambda x: x[0], reverse=True):
                normalized = content.strip().lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                unique.append((score, content, meta))

            # ---- truncate to top_k ----
            unique = unique[:top_k]

            facts = [
                RetrievedMemory(content=content, score=score, metadata=meta)
                for score, content, meta in unique
            ]

            layer_counts = {}
            for m in [f.metadata for f in facts]:
                layer = m.get("layer", "?")
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            logger.info(
                "SEARCH: %d total facts across layers: %s",
                len(facts),
                dict(layer_counts),
            )

            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=facts,
                retrieval_metadata={
                    "adapter": "graphiti_local",
                    "total_results": len(facts),
                    "layers": layer_counts,
                },
            )

        except Exception as exc:
            logger.error("SEARCH failed: %s", str(exc)[:200])
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"adapter": "graphiti_local", "error": str(exc)[:200]},
            )

    # ------------------------------------------------------------------
    # Answer (LOCOMO-style 7-step reasoning)
    # ------------------------------------------------------------------

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """Generate an answer via LLM with LOCOMO-style 7-step reasoning.

        When ``search_result`` is provided in kwargs, builds chronologically
        sorted memories with date prefixes (mirroring mem0 adapter). Falls
        back to the pre-formatted ``context`` string otherwise.
        """
        api_key = self._resolve_api_key("LLM")
        base_url = self._resolve_base_url("LLM")
        model = self.llm_config.get("model", os.environ.get("LLM_MODEL", "deepseek-chat"))
        temperature = self.llm_config.get("temperature", 0)
        max_tokens = self.llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        # Route through proxy when configured
        if self.llm_proxy_url:
            base_url = self.llm_proxy_url

        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        reference_date = (
            self._conversation_reference_date.get(conversation_id)
            or kwargs.get("reference_date")
            or "2023"
        )

        search_result = kwargs.get("search_result")
        if search_result and hasattr(search_result, "results"):
            memories_text = self._build_memories_text(search_result.results)
        else:
            memories_text = context

        prompt = f"""You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory -- important details are often scattered across many memories, including ones far down the list. Give equal weight to ALL memories regardless of position -- a memory near the end is just as likely to contain the answer as one near the beginning. In these memories, "User" refers to the main person whose memories these are.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity. If the question asks "What does Person A like?" and a memory says "Person B likes X", do NOT use that memory to answer about Person A. In two-person conversations, both speakers' actions are relevant -- if the question asks about person A and a memory attributes an action to person B (the other speaker), that information is still valid evidence from their shared conversations, but always check the attribution is correct.

## Step 3: COMBINE AND CROSS-REFERENCE
- COMBINE facts from multiple memories about the same topic. If one memory says "won first place" and another says "performed a piece titled X," those describe the same event -- connect them.
- For listing/counting questions, extract EVERY distinct item from ALL memories. A single memory may contain multiple items. Think about what CATEGORIES of answers the question could have, then re-scan specifically for each category.
- For counting questions ("how many times", "how many X"), enumerate each distinct instance explicitly with its date or context BEFORE giving a final count. Do not estimate -- list them out, then count the list.
- DECOMPOSE complex sentences: "an immersive X with Y, enjoys Z" contains multiple distinct facts. Each could be the answer.
- Connect related facts across memories: if one says "nearby lake" and another says "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe. If one says "bought X in Paris", infer the country is France.

## Step 4: SELECT THE BEST ANSWER
- Do NOT assume the highest-ranked memory is correct. Multiple memories may describe different events for the same topic. Compare each candidate's relevance to the SPECIFIC question, not its retrieval score. A lower-ranked memory that directly answers the question beats a higher-ranked one that is only tangentially related.
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description. Rate each candidate as HIGH specificity (name, title, number, specific activity) or LOW (generic description), and prefer HIGH.
- Report what someone actually DID, not what was offered or available to them. "Has not tried X yet" means X was NOT done -- disqualify it. "Joined X" or "has done X" means it WAS done -- prefer it.
- When multiple memories repeat the same generic fact, that repetition does NOT make it more correct than a single memory with a more specific answer.
- Photos depict what was IN the photo, not facts about someone's daily life. Prefer direct statements over photo descriptions for inferences.
- Re-read the question carefully before answering. If it asks "what aspect/type/kind", answer with the specific aspect. If it asks "what did they discover they both enjoy", answer with the specific thing, not the setting.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.
- Calculate time relative to this date, NOT today. Never output 2025 or 2026.
- Use dates explicitly stated in memory text. Do not invent or estimate dates.
- **The DATE prefix on each memory is the date the original conversation took place.** Trust it for "when did X happen" questions.
- **IGNORE any date that appears only in the "ingestion time" fallback** -- those are unreliable (they reflect when the memory was stored, not when the conversation happened).
- When a question asks what someone "shared" or "mentioned" on a date, that date is when they TALKED about it -- look for events shortly BEFORE that date.
- For "how long" questions, find the start and end dates explicitly, then compute the duration. Do not guess.
- TEMPORAL DISAMBIGUATION: When you find MULTIPLE instances of similar events at different dates, enumerate them all with their dates before picking. If the question uses past tense + "the" -> select the instance closest to (and before) the reference date. If future tense ("plans to", "going to") -> select the earliest planned date. NEVER default to the first-mentioned or highest-scored instance -- the DATE determines the answer.

## Step 6: INCLUSION CHECK (for lists and counts)
If you found items during reasoning that you're tempted to exclude from your answer -- STOP. Include them unless you have STRONG evidence they are wrong. The most common mistake is finding relevant items but then dropping them due to overly strict filtering. More items is better than fewer when there is supporting evidence.
- For counting: after enumerating, re-verify each item. Check for duplicates (same event described differently) and ensure you haven't missed items from memories late in the list.
- The question assumes something happened. Find WHAT happened, don't say nothing happened.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified", "not mentioned", "no record", or "the memories don't say" -- if ANY memory contains relevant information, give the best answer from available evidence. No hedging, no caveats. If the question asks for a list, include ALL items found. NEVER return an empty answer when relevant memories exist.
- **ANTI-HALLUCINATION for dates**: When a memory's text mentions a date (e.g. "visited Paris on May 8, 2023"), use THAT date verbatim -- DO NOT substitute the current year, today, or the ingestion date. If two memories give different dates for the same event, trust the one whose date prefix matches.
- NEVER generate specific names, titles, places, or dates that do not appear in any memory above. If no memory contains the specific detail the question asks for, answer with what the memories DO contain rather than guessing.
- For open-domain/opinion questions ("Would X do Y?", "Is X considered Z?"):
  * Follow the DIRECT causal reasoning in the memories. Do NOT construct elaborate counter-arguments.
  * "Would X still do Y without Z?" -- If memories show X does Y BECAUSE of Z, then without Z, answer "likely no."
  * "Would X do Y again soon?" -- If the most recent attempt involved a bad experience (accident, scare, trauma), answer "likely no." A recent negative experience outweighs historical positive patterns.
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_memories_text(results: list) -> str:
        """Build multi-layered memories text for the LOCOMO prompt.

        Groups results by layer type for the LLM:
          - Entity summaries first (who each person is, what they do)
          - Community summaries (thematic context)
          - Episode snippets (raw conversation excerpts)
          - Edge facts (relationship-level facts, chronologically ordered)

        Each memory is tagged with its layer type so the LLM can weigh
        summary-level vs fact-level evidence appropriately.
        """
        if not results:
            return "(No relevant memories found)"

        # Bucket by layer
        entities: list = []
        communities: list = []
        episodes: list = []
        edges: list = []

        for mem in results:
            layer = mem.metadata.get("layer", "") if mem.metadata else ""
            if layer == "entity":
                entities.append(mem)
            elif layer == "community":
                communities.append(mem)
            elif layer == "episode":
                episodes.append(mem)
            else:
                edges.append(mem)

        def _sort_key(mem):
            valid_at = mem.metadata.get("valid_at") if mem.metadata else None
            return valid_at or "z"

        episodes.sort(key=_sort_key)
        edges.sort(key=_sort_key)

        lines = []

        # Layer 1: Entity summaries — high-value overview
        if entities:
            lines.append("=== ENTITY SUMMARIES (what we know about key people) ===")
            for mem in entities:
                lines.append(f"[entity] {mem.content}")
            lines.append("")

        # Layer 2: Community summaries — thematic context
        if communities:
            lines.append("=== COMMUNITY THEMES (shared topics and context) ===")
            for mem in communities:
                lines.append(f"[community] {mem.content}")
            lines.append("")

        # Layer 3: Episode snippets — raw conversation excerpts
        if episodes:
            lines.append("=== CONVERSATION EXCERPTS (original dialogue) ===")
            for mem in episodes:
                valid_at = mem.metadata.get("valid_at") if mem.metadata else None
                date_str = GraphitiLocalAdapter._format_date(valid_at)
                lines.append(f"[episode | {date_str}] {mem.content}")
            lines.append("")

        # Layer 4: Edge facts — extracted relationship facts
        if edges:
            lines.append("=== EXTRACTED FACTS (chronological, oldest first) ===")
            for mem in edges:
                valid_at = mem.metadata.get("valid_at") if mem.metadata else None
                date_str = GraphitiLocalAdapter._format_date(valid_at)
                relation = mem.metadata.get("relation", "") if mem.metadata else ""
                tag = f"[fact | {date_str}]" if relation == "" else f"[fact: {relation} | {date_str}]"
                lines.append(f"{tag} {mem.content}")

        return "\n".join(lines)

    @staticmethod
    def _format_date(valid_at) -> str:
        """Format a valid_at timestamp into a readable date string."""
        if not valid_at:
            return "unknown date"
        try:
            dt = datetime.fromisoformat(str(valid_at))
            return dt.strftime("%B %d, %Y")
        except (ValueError, TypeError):
            return str(valid_at)[:10]

    @staticmethod
    def _resolve_timestamp(
        msg_timestamp, chunk_timestamp: Optional[int]
    ) -> datetime:
        """Resolve the best reference_time for a message.

        Handles three formats found in real data:
          - ISO 8601 string  e.g. "2023-01-20T16:04:00+00:00" (msg.timestamp)
          - Unix epoch int    e.g. 1674230640                (chunk.timestamp)
          - datetime object
        """
        ts = msg_timestamp or chunk_timestamp
        if ts is None:
            return datetime.now(timezone.utc)

        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, str):
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    dt = datetime.strptime(ts[:35], fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue

        if isinstance(ts, (int, float)):
            try:
                return datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                pass

        return datetime.now(timezone.utc)

    @staticmethod
    def _append_image_metadata(msg, content: str) -> str:
        """Append image metadata (blip_caption / query) to content.

        Mirrors the mem0 adapter's _render_message_content pattern so the
        LLM extractor sees photo descriptions as part of the episode text
        and can create entities/edges from them.
        """
        metadata = getattr(msg, "metadata", None) or {}
        if not isinstance(metadata, dict):
            return content
        blip = metadata.get("blip_caption", "") or ""
        query = metadata.get("query", "") or ""

        photo_tag = ""
        if query and blip:
            photo_tag = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo_tag = f"[Sharing image - query for: {query}]"
        elif blip:
            photo_tag = f"[Sharing image that shows: {blip}]"

        if photo_tag:
            return f"{content} {photo_tag}".strip() if content else photo_tag
        return content

    @staticmethod
    def _track_ref_date(chunk: ChunkedMessage, tracker: Dict[str, str]) -> None:
        """Record the latest session date per conversation for answer() grounding."""
        cid = chunk.conversation_id
        date_str = None

        if chunk.session_time_str:
            for fmt in (
                "%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
            ):
                try:
                    date_str = datetime.strptime(
                        chunk.session_time_str[:35], fmt
                    ).strftime("%B %d, %Y")
                    break
                except ValueError:
                    continue

        if not date_str and chunk.timestamp:
            try:
                date_str = datetime.fromtimestamp(
                    int(chunk.timestamp), tz=timezone.utc
                ).strftime("%B %d, %Y")
            except (ValueError, OverflowError, OSError):
                pass

        if date_str:
            prev = tracker.get(cid)
            if prev is None or date_str > prev:
                tracker[cid] = date_str
