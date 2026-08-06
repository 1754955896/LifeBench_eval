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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

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

        # Per-conversation reference date for answer() temporal grounding
        self._conversation_reference_date: Dict[str, str] = {}

        # Lazy init
        self._graphiti = None

    # ------------------------------------------------------------------
    # LLM / Embedder / Reranker helpers (independently configurable)
    # ------------------------------------------------------------------

    def _get_llm_client(self):
        """Create an LLM client matching the Graphiti eval pattern.

        Uses OpenAIGenericClient (json_object mode) for non-OpenAI providers
        (e.g. DeepSeek) that do not support the responses.parse API.
        """
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        api_key = self._resolve_api_key("LLM")
        base_url = self._resolve_base_url("LLM")
        model = self.llm_config.get("model", os.environ.get("LLM_MODEL", "deepseek-chat"))

        llm_cfg = LLMConfig(api_key=api_key, base_url=base_url, model=model, temperature=0)

        if "openai.com" in base_url:
            return OpenAIClient(config=llm_cfg)

        return OpenAIGenericClient(config=llm_cfg, structured_output_mode="json_object")

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
        """Resolve API key for a service prefix (LLM, VECTORIZE, RERANK).

        Priority: env var {PREFIX}_API_KEY → YAML config → env var OPENAI_API_KEY
        """
        env_val = os.environ.get(f"{prefix}_API_KEY", "").strip()
        if env_val and env_val != "EMPTY":
            return env_val

        if prefix == "LLM":
            return self.llm_config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
        if prefix == "VECTORIZE":
            return self.embedder_config.get("api_key", "")
        if prefix == "RERANK":
            return self.rerank_config.get("api_key", "")

        return ""

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

    async def _get_graphiti(self):
        """Get or create the Graphiti client (lazy init).

        LLM, embedder, and cross-encoder are each independently configurable
        via their respective env vars (LLM_*, VECTORIZE_*, RERANK_*).
        """
        if self._graphiti is None:
            from graphiti_core.graphiti import Graphiti

            llm_client = self._get_llm_client()
            embedder = self._get_embedder_client()
            cross_encoder = self._get_cross_encoder()

            self._graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )

        return self._graphiti

    async def close(self) -> None:
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None

    # ------------------------------------------------------------------
    # Ingest (mirrors eval_e2e_graph_building.build_subgraph)
    # ------------------------------------------------------------------

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest messages following the Graphiti LongMemEval eval pattern.

        Each message becomes one add_episode() call:
          episode_body = '{speaker}: {content}'
          source = EpisodeType.message
          group_id = MD5(conversation_id)
        """
        from graphiti_core.nodes import EpisodeType

        graphiti = await self._get_graphiti()

        total_added = 0
        total_failed = 0
        latest_ref_date: Dict[str, str] = {}

        for chunk in chunks:
            if not chunk.messages:
                continue

            group_id = _safe_group_id(chunk.conversation_id)
            logger.info(
                "ADD: conversation=%s group=%s messages=%d",
                chunk.conversation_id, group_id, len(chunk.messages),
            )

            for msg in chunk.messages:
                # Build episode_body mirroring the eval script's
                #   f'{msg["role"]}: {msg["content"]}'
                # Our data uses speaker_name instead of role, and may carry
                # image metadata (blip_caption / query) that should be
                # included in the episode text so the LLM can extract them.
                speaker = getattr(msg, "speaker_name", "") or "unknown"
                content = (getattr(msg, "content", "") or "").strip()
                content = self._append_image_metadata(msg, content)
                if not content:
                    continue
                episode_body = f"{speaker}: {content}"

                # Resolve reference_time (handles ISO-8601 str, unix int, datetime)
                reference_time = self._resolve_timestamp(
                    getattr(msg, "timestamp", None),
                    chunk.timestamp,
                )

                # Track latest session date for answer() temporal grounding
                self._track_ref_date(chunk, latest_ref_date)

                try:
                    await graphiti.add_episode(
                        name="",
                        episode_body=episode_body,
                        reference_time=reference_time,
                        source=EpisodeType.message,
                        source_description=f"conversation {chunk.conversation_id}",
                        group_id=group_id,
                    )
                    total_added += 1
                except Exception as exc:
                    logger.warning(
                        "ADD episode failed: %s",
                        str(exc)[:200],
                    )
                    total_failed += 1

        self._conversation_reference_date.update(latest_ref_date)

        return {
            "type": "graphiti_local",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
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

        graphiti = await self._get_graphiti()
        group_id = _safe_group_id(conversation_id)

        logger.info("SEARCH: query=%.50s... group=%s top_k=%d", query, group_id, top_k)

        try:
            from graphiti_core.search.search_config_recipes import (
                COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            )

            config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER
            # Ask each layer for more candidates so the merged top_k is diverse
            config.limit = max(top_k, 20)
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
        api_key = self.llm_config.get("api_key", "")
        base_url = self.llm_config.get("base_url", "https://api.deepseek.com/v1")
        model = self.llm_config.get("model", "deepseek-chat")
        temperature = self.llm_config.get("temperature", 0)
        max_tokens = self.llm_config.get("max_tokens", 32768)

        if not api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

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
