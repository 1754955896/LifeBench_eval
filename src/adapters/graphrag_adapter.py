"""
GraphRAG Adapter for LifeBench_eval.

Uses Microsoft GraphRAG 3.x with DRIFT search. Runs from within the graphrag
venv (Python >=3.11 required) so all imports are direct — no subprocess bridging.

Key pattern (mirrors cognee adapter):
  - add_chunks(expect_search=False): write txt file, no index
  - add_chunks(expect_search=True):  write txt + graphrag update/index
  - Only index on dates that have QA questions

Workspace layout for each conversation:
  {output_dir}/graphrag/workspaces/{conv_id}/
    settings.yaml
    input/        # session txt files
    output/       # parquet after index
    cache/        # index cache

Usage:
  cd LifeBench_eval
  systems/graphrag/.venv/Scripts/activate
  python cli.py --dataset lifebench --system graphrag --smoke
"""
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import yaml

from graphrag.api import build_index
from graphrag.api.query import drift_search
from graphrag.config.load_config import load_config

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import RetrievedMemory, SearchResult

logger = logging.getLogger(__name__)

# ── helpers ─────────────────────────────────────────────────


def _resolve_env_var(value: Any) -> str:
    if not isinstance(value, str):
        return str(value) if value else ""
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)
    return re.sub(pattern, replacer, value)


def _format_timestamp(ts: Any) -> str:
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts.strip()
    iso = getattr(ts, "isoformat", None)
    if iso is None:
        return ""
    try:
        return ts.isoformat()
    except Exception:
        return ""


def _format_messages_text(messages: List[Any]) -> List[str]:
    lines = []
    for msg in messages:
        speaker = (getattr(msg, "speaker_name", None) or "unknown").strip()
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            continue
        ts_str = _format_timestamp(getattr(msg, "timestamp", None))
        ts_prefix = f"[{ts_str}] " if ts_str else ""

        meta = getattr(msg, "metadata", None) or {}
        tags = []
        blip = (meta.get("blip_caption", "") or "").strip()
        query_tag = (meta.get("query", "") or "").strip()
        if blip:
            tags.append(f"image: {blip}")
        if query_tag:
            tags.append(f"query: {query_tag}")
        tag_str = f" | {'; '.join(tags)}" if tags else ""

        lines.append(f"{ts_prefix}{speaker}: {content}{tag_str}")
    return lines


# ── adapter ─────────────────────────────────────────────────


@register_adapter("graphrag")
class GraphRAGAdapter(BaseAdapter):
    """Microsoft GraphRAG adapter with per-conversation workspaces.

    Each conversation gets its own GraphRAG workspace so search is naturally
    scoped to a single user's memories. Indexing only runs when the pipeline
    signals expect_search=True (i.e. the current date has QA questions).
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = Path(output_dir).resolve() if output_dir else Path("results/graphrag").resolve()
        self.stats_collector = stats_collector

        # LLM
        llm_cfg = config.get("llm", {})
        self.llm_model = _resolve_env_var(llm_cfg.get("model", "deepseek-v4-flash"))
        self.llm_api_key = _resolve_env_var(llm_cfg.get("api_key", ""))
        self.llm_base_url = _resolve_env_var(llm_cfg.get("base_url", "https://api.deepseek.com"))
        self.llm_temperature = llm_cfg.get("temperature", 0)
        self.llm_max_tokens = llm_cfg.get("max_tokens", 32768)

        # Embedding
        embed_cfg = config.get("embedding", {})
        self.embed_model = _resolve_env_var(embed_cfg.get("model", "Qwen/Qwen3-Embedding-4B"))
        self.embed_api_key = _resolve_env_var(embed_cfg.get("api_key", ""))
        self.embed_base_url = _resolve_env_var(embed_cfg.get("base_url", "https://api.siliconflow.cn/v1"))
        self.embed_dimension = _resolve_env_var(embed_cfg.get("dimension", "2560"))

        # Index
        idx_cfg = config.get("index", {})
        self.chunk_size = idx_cfg.get("chunk_size", 1000)
        self.chunk_overlap = idx_cfg.get("chunk_overlap", 250)
        self.max_gleanings = idx_cfg.get("max_gleanings", 3)
        self.concurrent_requests = idx_cfg.get("concurrent_requests", 25)
        self.report_max_length = idx_cfg.get("report_max_length", 4000)
        self.report_max_input_length = idx_cfg.get("report_max_input_length", 16000)
        self.desc_max_length = idx_cfg.get("desc_max_length", 1000)
        self.claims_gleanings = idx_cfg.get("claims_gleanings", 2)

        # Search
        search_cfg = config.get("search", {})
        self.search_method = search_cfg.get("method", "drift")
        self.top_k = search_cfg.get("top_k", 40)
        self.community_level = search_cfg.get("community_level", 2)
        self.primer_folds = search_cfg.get("primer_folds", 5)
        self.drift_k_followups = search_cfg.get("drift_k_followups", 20)
        self.n_depth = search_cfg.get("n_depth", 3)
        self.local_search_text_unit_prop = search_cfg.get("local_search_text_unit_prop", 0.5)
        self.local_search_community_prop = search_cfg.get("local_search_community_prop", 0.25)
        self.local_search_top_k_mapped_entities = search_cfg.get("local_search_top_k_entities", 15)
        self.local_search_top_k_relationships = search_cfg.get("local_search_top_k_relationships", 15)

        # Answer
        ans_cfg = config.get("answer", {})
        self.answer_temperature = ans_cfg.get("temperature", 0.3)
        self.answer_max_tokens = ans_cfg.get("max_tokens", 32768)

        self._graphrag_dir = self.output_dir / "graphrag"

    # ── workspace management ──────────────────────────────────

    def _get_workspace(self, conversation_id: str) -> Path:
        return self._graphrag_dir / "workspaces" / conversation_id

    def _ensure_workspace(self, conversation_id: str) -> Path:
        ws = self._get_workspace(conversation_id)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "input").mkdir(exist_ok=True)
        (ws / "output").mkdir(exist_ok=True)
        (ws / "cache").mkdir(exist_ok=True)
        self._write_settings(ws)
        return ws

    def _write_settings(self, workspace: Path) -> None:
        base_url = self.llm_base_url.rstrip("/")
        embed_base_url = self.embed_base_url.rstrip("/")

        settings = {
            # Root-level concurrency
            "concurrent_requests": self.concurrent_requests,
            "async_mode": "asyncio",

            "completion_models": {
                "default_completion_model": {
                    "type": "litellm",
                    "model_provider": "openai",
                    "auth_type": "api_key",
                    "api_key": self.llm_api_key,
                    "model": self.llm_model,
                    "api_base": base_url,
                    "retry": {
                        "strategy": "exponential_backoff",
                        "max_retries": 3,
                    },
                }
            },
            "embedding_models": {
                "default_embedding_model": {
                    "type": "litellm",
                    "model_provider": "openai",
                    "auth_type": "api_key",
                    "api_key": self.embed_api_key,
                    "model": self.embed_model,
                    "api_base": embed_base_url,
                    "retry": {
                        "strategy": "exponential_backoff",
                        "max_retries": 3,
                    },
                    "call_args": {
                        "dimensions": int(self.embed_dimension),
                        "allowed_openai_params": ["dimensions"],
                    },
                }
            },
            "input_storage": {
                "type": "file",
                "base_dir": str((workspace / "input").resolve()),
            },
            "output_storage": {
                "type": "file",
                "base_dir": str((workspace / "output").resolve()),
            },
            "update_output_storage": {
                "type": "file",
                "base_dir": str((workspace / "output").resolve()),
            },
            "cache": {
                "type": "json",
                "storage": {
                    "type": "file",
                    "base_dir": str((workspace / "cache").resolve()),
                }
            },
            "reporting": {
                "type": "file",
                "base_dir": str((workspace / "cache").resolve()),
            },
            "vector_store": {
                "type": "lancedb",
                "db_uri": str((workspace / "output" / "lancedb").resolve()),
                "vector_size": int(self.embed_dimension) if self.embed_dimension else 3072,
            },
            "chunking": {
                "type": "tokens",
                "size": self.chunk_size,
                "overlap": self.chunk_overlap,
            },
            "extract_graph": {
                "completion_model_id": "default_completion_model",
                "max_gleanings": self.max_gleanings,
            },
            "extract_claims": {
                "enabled": self.claims_gleanings > 0,
                "max_gleanings": self.claims_gleanings,
            },
            "community_reports": {
                "completion_model_id": "default_completion_model",
                "max_length": self.report_max_length,
                "max_input_length": self.report_max_input_length,
            },
            "summarize_descriptions": {
                "max_length": self.desc_max_length,
            },
            "embed_text": {
                "embedding_model_id": "default_embedding_model",
                "names": [
                    "entity_description",
                    "community_full_content",
                    "text_unit_text",
                ],
            },
            "drift_search": {
                "completion_model_id": "default_completion_model",
                "embedding_model_id": "default_embedding_model",
                "primer_folds": self.primer_folds,
                "drift_k_followups": self.drift_k_followups,
                "n_depth": self.n_depth,
                "local_search_text_unit_prop": self.local_search_text_unit_prop,
                "local_search_community_prop": self.local_search_community_prop,
                "local_search_top_k_mapped_entities": self.local_search_top_k_mapped_entities,
                "local_search_top_k_relationships": self.local_search_top_k_relationships,
            },
        }
        settings_path = workspace / "settings.yaml"
        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(settings, f, allow_unicode=True, sort_keys=False)

    def _is_indexed(self, workspace: Path) -> bool:
        return (workspace / "output" / "entities.parquet").exists()

    def _has_input_files(self, workspace: Path) -> bool:
        input_dir = workspace / "input"
        return input_dir.exists() and any(input_dir.glob("*.txt"))

    def _has_unindexed_input(self, workspace: Path) -> bool:
        """Check if any input files are newer than the index (or index missing)."""
        index_file = workspace / "output" / "entities.parquet"
        if not index_file.exists():
            return self._has_input_files(workspace)
        index_mtime = index_file.stat().st_mtime
        input_dir = workspace / "input"
        if not input_dir.exists():
            return False
        for txt_file in input_dir.glob("*.txt"):
            if txt_file.stat().st_mtime > index_mtime:
                return True
        return False

    async def _ensure_index(self, conversation_id: str) -> None:
        """Auto-index or update if needed. Called before search to handle
        the case where QA asktime is after the last session date."""
        ws = self._get_workspace(conversation_id)
        if not self._has_input_files(ws):
            return
        if self._has_unindexed_input(ws):
            logger.info("Auto-indexing %s (unindexed input detected)", conversation_id)
            await self._run_index(conversation_id)

    # ── BaseAdapter ───────────────────────────────────────────

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        expect_search: bool = kwargs.get("expect_search", True)

        total_added = 0
        workspaces_touched: set[str] = set()

        for chunk in chunks:
            if not chunk.messages:
                continue
            conv_id = chunk.conversation_id or "default"
            ws = self._ensure_workspace(conv_id)

            text = self._format_chunk(chunk)
            if not text:
                continue

            session_label = chunk.session_id or "session_0"
            safe_label = re.sub(r'[\\/:*?"<>|]', '_', str(session_label))
            txt_path = ws / "input" / f"{safe_label}.txt"
            txt_path.write_text(text, encoding="utf-8")
            total_added += 1
            workspaces_touched.add(conv_id)

        if expect_search:
            for conv_id in workspaces_touched:
                await self._run_index(conv_id)

        return {
            "type": "graphrag",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": 0,
            "indexed": len(workspaces_touched) if expect_search else 0,
        }

    async def _run_index(self, conversation_id: str) -> None:
        ws = self._get_workspace(conversation_id)
        if not ws.exists():
            logger.warning("Workspace %s does not exist, skipping index", ws)
            return

        is_update = self._is_indexed(ws)
        cmd = "update" if is_update else "index"
        logger.info("GraphRAG %s: %s", cmd, ws)

        try:
            config = load_config(ws)
            await build_index(config, is_update_run=is_update)
            logger.info("graphrag %s OK for %s", cmd, conversation_id)
        except Exception as exc:
            logger.error("graphrag %s error for %s: %s", cmd, conversation_id, exc)

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        ws = self._get_workspace(conversation_id)
        question_id = kwargs.get("question_id", "")
        top_k = kwargs.get("top_k", self.top_k)

        if not self._is_indexed(ws) or self._has_unindexed_input(ws):
            await self._ensure_index(conversation_id)

        if not self._is_indexed(ws):
            logger.warning("No index for %s after auto-index, search returns empty", conversation_id)
            return SearchResult(
                question_id=question_id, query=query,
                conversation_id=conversation_id, results=[],
                retrieval_metadata={"adapter": "graphrag", "error": "not_indexed"},
            )

        try:
            config = load_config(ws)
            output_dir = ws / "output"

            entities_df = pd.read_parquet(output_dir / "entities.parquet")
            communities_df = pd.read_parquet(output_dir / "communities.parquet")
            reports_df = pd.read_parquet(output_dir / "community_reports.parquet")
            text_units_df = pd.read_parquet(output_dir / "text_units.parquet")
            relationships_df = pd.read_parquet(output_dir / "relationships.parquet")

            response, _context_data = await drift_search(
                config=config,
                entities=entities_df,
                communities=communities_df,
                community_reports=reports_df,
                text_units=text_units_df,
                relationships=relationships_df,
                community_level=self.community_level,
                response_type="multiple paragraphs",
                query=query,
            )

            response_text = response if isinstance(response, str) else str(response)
            paragraphs = [p.strip() for p in response_text.split("\n\n") if p.strip()]

            memories = [
                RetrievedMemory(
                    content=para,
                    score=1.0 - (i * 0.05),
                    metadata={"index": i, "layer": "drift"},
                )
                for i, para in enumerate(paragraphs[:top_k])
            ]
            return SearchResult(
                question_id=question_id, query=query,
                conversation_id=conversation_id, results=memories,
                retrieval_metadata={
                    "adapter": "graphrag",
                    "total_results": len(memories),
                    "method": self.search_method,
                },
            )

        except Exception as exc:
            logger.error("search failed for %s: %s", conversation_id, exc)
            return SearchResult(
                question_id=question_id, query=query,
                conversation_id=conversation_id, results=[],
                retrieval_metadata={"adapter": "graphrag", "error": str(exc)[:200]},
            )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        if not self.llm_api_key:
            return "Error: No LLM API key configured"

        reference_date = kwargs.get("reference_date", "2023")

        prompt = f"""You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory -- important details are often scattered across many memories.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity.

## Step 3: COMBINE AND CROSS-REFERENCE
Combine facts from multiple memories about the same topic. For listing/counting questions, extract EVERY distinct item.

## Step 4: SELECT THE BEST ANSWER
Choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.

## Step 6: INCLUSION CHECK
If you found items during reasoning that you're tempted to exclude -- STOP. Include them unless you have STRONG evidence they are wrong.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified" or "no record" -- if ANY memory contains relevant information, give the best answer.

{context if context else "(No relevant memories found)"}

Question: {query}

Work through Steps 1-7, then give your final answer after "ANSWER:"."""

        base_url = self.llm_base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        url = f"{base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.answer_temperature,
            "max_tokens": self.answer_max_tokens,
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

    async def close(self) -> None:
        pass

    def get_system_info(self) -> Dict[str, Any]:
        import graphrag
        return {
            "name": "GraphRAGAdapter",
            "search_method": self.search_method,
            "graphrag_version": getattr(graphrag, "__version__", "unknown"),
        }

    # ── internal ─────────────────────────────────────────────

    def _format_chunk(self, chunk: ChunkedMessage) -> str:
        header = f"[{chunk.conversation_id} {chunk.session_id}]"
        if hasattr(chunk, 'session_time_str') and chunk.session_time_str:
            header += f" ({chunk.session_time_str})"
        body = _format_messages_text(chunk.messages)
        if not body:
            return ""
        return header + "\n" + "\n".join(body)
