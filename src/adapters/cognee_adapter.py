"""
Cognee Adapter for LifeBench_eval.

In-process cognee SDK adapter that mirrors the cognee BEAM eval flow:
    add(corpus)  ->  cognify()  ->  cognee.search(GRAPH_COMPLETION, datasets=[user])

Replaces the previous HTTP REST client against cognee-server (Docker). The cognee
package is vendored at LifeBench_eval/systems/cognee and is consumed via direct
Python imports instead of going through a separate server.

Reference: cognee/eval_framework/{corpus_builder,answer_generation}/...
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import RetrievedMemory, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-load environment patches BEFORE `import cognee` below.
#
# Two platform hazards block cognee from loading on Windows when this file
# sits next to a vendored `systems/cognee/.env`:
#
#   1. cognee/__init__.py runs `dotenv.load_dotenv(override=True)` and reads
#      that vendored `.env`, which carries Docker-only paths
#      (`/app/data/.cognee_data`, `/app/data/.cognee_system`). Those values
#      become BaseConfig fields and fail Windows absolute-path validation,
#      raising `pydantic.ValidationError` during import — cognee ends up as
#      None in `cognee_adapter` and `add_chunks` blows up at runtime.
#   2. cognee's BaseConfig requires DATA/SYSTEM_ROOT_DIRECTORY to be set to
#      absolute paths. The harness already loaded the real `.env` above the
#      adapter import, so we just set safe defaults if missing.
#
# We replace `dotenv.load_dotenv` with a no-op for the rest of the process
# so cognee's internal call becomes harmless. The harness's earlier explicit
# `load_dotenv(project_root / ".env")` already ran before this module loaded,
# so harness-side `.env` semantics are preserved.
# ---------------------------------------------------------------------------
import dotenv as _cognee_dotenv_module
_orig_load_dotenv = _cognee_dotenv_module.load_dotenv
_cognee_dotenv_module.load_dotenv = lambda *a, **kw: False  # noqa: E731

# Provide absolute cognee storage paths. Falls back to `.cognee_data` /
# `.cognee_system` next to the cognee submodule if cognee's defaults aren't
# already in os.environ. create the dirs so cognee can write to them.
_cognee_submodule_root = (
    Path(__file__).resolve().parents[2] / "systems" / "cognee"
)
_cognee_data_root = _cognee_submodule_root / ".cognee_data"
_cognee_system_root = _cognee_submodule_root / ".cognee_system"
try:
    _cognee_data_root.mkdir(parents=True, exist_ok=True)
    _cognee_system_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DATA_ROOT_DIRECTORY", str(_cognee_data_root))
    os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(_cognee_system_root))
except OSError:
    pass  # permission issues shouldn't block import — cognee will error at use


# Defensive imports — adapter module must load even if cognee isn't installed so
# the registry's lazy import path stays healthy. Method calls fail loudly at use.
#
# All cognee API calls (add / cognify / search) read the LLM and embedding model
# from `os.environ` (set by `CogneeBuilder.build()` via the harness `.env` and
# the YAML's `llm`/`embedding` blocks). The adapter does NOT instantiate or
# pass per-call LLMConfig / EmbeddingConfig objects — keeping search, add,
# and answer on a single, consistent configuration.
try:
    import cognee
    from cognee.modules.search.types import SearchType
    from cognee.modules.retrieval.utils.completion import (
        generate_completion as _cognee_generate_completion,
    )

    _COGNEE_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - import-time only
    cognee = None  # type: ignore[assignment]
    SearchType = None  # type: ignore[assignment]
    _cognee_generate_completion = None  # type: ignore[assignment]
    _COGNEE_IMPORT_ERROR = exc


# Mapping of adapter config string -> cognee SearchType. Mirrors the eval_framework's
# retriever_options dict for the non-beam graph retrievers.
_RETRIEVER_TYPE_MAP: Dict[str, "SearchType"] = {
    "graph_completion": "GRAPH_COMPLETION",
    "graph_completion_cot": "GRAPH_COMPLETION_COT",
    "graph_completion_context_extension": "GRAPH_COMPLETION_CONTEXT_EXTENSION",
    "graph_summary_completion": "GRAPH_SUMMARY_COMPLETION",
}


def _resolve_search_type(retriever_type: str):
    """Resolve a config-string retriever type to a cognee SearchType enum member."""
    if SearchType is None:  # cognee missing
        return None
    name = _RETRIEVER_TYPE_MAP.get(retriever_type, "GRAPH_COMPLETION")
    return getattr(SearchType, name)


def _payload_to_memories(payload: Any, dataset_name: str) -> List[RetrievedMemory]:
    """Convert a cognee SearchResultPayload.context into RetrievedMemory items."""
    if payload is None:
        return []

    context = getattr(payload, "context", None)
    if context is None:
        return []

    items: List[str]
    if isinstance(context, str):
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        items = lines if lines else [context]
    elif isinstance(context, list):
        items = []
        for entry in context:
            if isinstance(entry, str):
                items.append(entry)
            elif isinstance(entry, dict):
                txt = entry.get("text") or entry.get("content") or str(entry)
                items.append(str(txt))
            else:
                items.append(str(entry))
    else:
        items = [str(context)]

    memories: List[RetrievedMemory] = []
    total = len(items)
    for idx, content in enumerate(items):
        # Higher-ranked items get higher pseudo-scores (1.0 -> 0.0 linearly).
        score = 1.0 - (idx / max(total, 1))
        memories.append(
            RetrievedMemory(
                content=content,
                score=score,
                metadata={"dataset": dataset_name},
            )
        )
    return memories


def _format_timestamp(ts: Any) -> str:
    """Render a `datetime` (or ISO string) as a precision-preserving prefix.

    Seconds and timezone offset are kept so temporal-reasoning questions like
    "what happened a few minutes after X" can be answered against the ingested
    text. Falls back to the raw string when `ts` isn't a datetime/ISO-parsable.
    """
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts.strip()
    iso = getattr(ts, "isoformat", None)
    if iso is None:
        return ""
    try:
        s = ts.isoformat()
    except Exception:
        return ""
    # datetime.isoformat() already gives full precision (incl. seconds + tz);
    # nothing else to do. Trimming to minutes would lose information.
    return s


def _format_messages_text(messages: List[Any]) -> List[str]:
    """Render each message as one line, preserving order and message metadata.

    Format: `[<iso-timestamp>] <Speaker>: <content>[ | caption=<...>][, query=<...>]`

    - Timestamp: full ISO 8601 (seconds + tz offset) — keeps temporal precision.
    - Speaker: `msg.speaker_name` or `User` as fallback.
    - Content: `msg.content`.
    - Metadata: when present, `blip_caption` / `query` fields are appended as
      pipe-delimited tags so cognee's `extract_graph_from_data` LLM picks them
      up instead of dropping them on the floor.
    """
    lines: List[str] = []
    for msg in messages:
        speaker = (msg.speaker_name or "User").strip() or "User"
        content = (msg.content or "").strip()
        meta = getattr(msg, "metadata", None) or {}

        ts_str = _format_timestamp(getattr(msg, "timestamp", None))
        ts_prefix = f"[{ts_str}] " if ts_str else ""

        meta_parts: List[str] = []
        blip = (meta.get("blip_caption") or "").strip()
        query = (meta.get("query") or "").strip()
        if blip:
            meta_parts.append(f"caption={blip}")
        if query:
            meta_parts.append(f"query={query}")
        meta_suffix = f" | {', '.join(meta_parts)}" if meta_parts else ""

        if not content and not meta_parts:
            continue
        body = content if content else "(no text)"
        lines.append(f"{ts_prefix}{speaker}: {body}{meta_suffix}")
    return lines


@register_adapter("cognee")
class CogneeAdapter(BaseAdapter):
    """In-process cognee SDK adapter — mirrors cognee's BEAM eval-framework flow.

    Configuration (all optional):
        dataset_prefix (str):       Namespace for per-user cognee datasets.
                                    Final dataset name = f"{prefix}{conversation_id}".
                                    NOTE: cleanup (resetting cognee state between runs)
                                    is the **builder's** concern — set
                                    `prune_on_init: true` on `cognee.yaml`, not here.
        top_k (int):                Default top_k passed to cognee.search. Default 10.
        retriever_type (str):       One of graph_completion, graph_completion_cot,
                                    graph_completion_context_extension,
                                    graph_summary_completion. Default graph_completion.
        system_prompt_path (str):   cognee prompt filename; default
                                    "answer_simple_question.txt".
        wide_search_top_k (int):    Forwarded to cognee.search. Default 100.
        chunk_size (int):           Forwarded to cognee.cognify chunker. Default 1024.
    """

    def __init__(
        self,
        config: dict,
        output_dir=None,
        stats_collector=None,
    ):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        self.dataset_prefix: str = config.get("dataset_prefix", "locomo_")
        self.top_k: int = config.get("top_k", 10)
        self.retriever_type: str = config.get("retriever_type", "graph_completion")
        self.system_prompt_path: str = config.get(
            "system_prompt_path", "answer_simple_question.txt"
        )
        self.wide_search_top_k: int = config.get("wide_search_top_k", 100)
        self.chunk_size: int = config.get("chunk_size", 1024)

        self.search_type = _resolve_search_type(self.retriever_type)

        # LLM and embedding models are NOT constructed here. cognee.add /
        # cognee.cognify / cognee.search all read LLMConfig and EmbeddingConfig
        # from os.environ (populated by CogneeBuilder.build() from the YAML's
        # `llm`/`embedding` blocks AND/OR the harness .env). Keeping a single
        # source of truth means search, add, and answer always agree on which
        # model is in use — there is no per-call-override pathway that could
        # silently drift them apart.

    # ----- internal helpers ----------------------------------------------------

    def _require_cognee(self) -> None:
        if cognee is None:
            raise NotImplementedError(
                f"cognee SDK unavailable — cannot use CogneeAdapter. "
                f"Import error: {_COGNEE_IMPORT_ERROR}"
            )

    def _dataset_name(self, conversation_id: str) -> str:
        return f"{self.dataset_prefix}{conversation_id or 'default'}"

    def _format_chunk(self, chunk: ChunkedMessage) -> str:
        """Render a ChunkedMessage as one text string — mirrors BEAMAdapter._flatten_chat."""
        header = f"--- Session {chunk.session_id or '0'} ---"
        body = _format_messages_text(chunk.messages)
        if not body:
            return ""
        return header + "\n" + "\n".join(body)

    # ----- BaseAdapter contract ------------------------------------------------

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Batch ingest, optionally followed by cognify, grouped per-user.

        Within a single `add_chunks` call we aggregate all chunks that share a
        `conversation_id` (i.e. a single user) into one cognee.add + (optionally)
        one cognee.cognify pair. The harness may pass `expect_search=False`
        for session-only dates whose data will never be queried — we then skip
        the cognify call entirely.

        Recognized kwargs:
            expect_search (bool): Default True. When False, only `cognee.add`
                runs; `cognee.cognify` is skipped (saves the 5-task pipeline
                + LLM extraction cost for data that won't be searched).
        """
        self._require_cognee()
        expect_search: bool = kwargs.get("expect_search", True)

        # Bucket chunks by conversation_id. Order preserved per-user via list.
        texts_by_user: Dict[str, List[str]] = {}
        n_input_chunks = 0
        for chunk in chunks:
            if not chunk.messages:
                continue
            text = self._format_chunk(chunk)
            if not text:
                continue
            texts_by_user.setdefault(chunk.conversation_id or "default", []).append(text)
            n_input_chunks += 1

        added = 0
        failed = 0
        for conv_id, texts in texts_by_user.items():
            ds = self._dataset_name(conv_id)
            try:
                # cognee reads LLM/embedding config from os.environ (set by
                # the builder). Keeping the API calls keyword-only here
                # ensures search/add/answer share one configuration.
                await cognee.add(texts, dataset_name=ds)
                if expect_search:
                    await cognee.cognify(
                        datasets=[ds],
                        chunk_size=self.chunk_size,
                        run_in_background=False,
                    )
                added += len(texts)
            except Exception as exc:  # noqa: BLE001 — per-user isolation
                failed += len(texts)
                logger.error(
                    "cognee add%s failed for dataset %s: %s",
                    "+cognify" if expect_search else "",
                    ds,
                    exc,
                    exc_info=True,
                )

        return {
            "type": "cognee",
            "total_chunks": n_input_chunks,
            "added": added,
            "failed": failed,
            "users": len(texts_by_user),
        }

    async def search(
        self,
        query: str,
        conversation_id: str,
        index: Any = None,
        **kwargs,
    ) -> SearchResult:
        self._require_cognee()

        ds = self._dataset_name(conversation_id)
        question_id = kwargs.get("question_id", "")
        top_k = kwargs.get("top_k", self.top_k)
        search_type = kwargs.get("search_type", self.search_type)

        try:
            results = await cognee.search(
                query_text=query,
                query_type=search_type,
                datasets=[ds],
                system_prompt_path=self.system_prompt_path,
                top_k=top_k,
                wide_search_top_k=self.wide_search_top_k,
                only_context=True,  # retrieve, don't LLM-complete yet
            )
        except Exception as exc:
            logger.warning(
                "cognee.search failed for dataset %s: %s", ds, exc, exc_info=True
            )
            results = []

        memories: List[RetrievedMemory] = []
        dataset_seen = ds
        for r in results or []:
            payload = getattr(r, "search_result", None)
            dataset_seen = getattr(r, "dataset_name", None) or ds
            memories.extend(_payload_to_memories(payload, dataset_seen))

        formatted_context = "\n\n".join(m.content for m in memories)
        return SearchResult(
            question_id=question_id,
            query=query,
            conversation_id=conversation_id,
            results=memories,
            retrieval_metadata={
                "adapter": "cognee",
                "dataset": dataset_seen,
                "retriever_type": self.retriever_type,
                "total_results": len(memories),
                "formatted_context": formatted_context,
                # Marker consumed by `answer()` to skip the second graph traversal.
                # Set whenever `only_context=True` retrieved successfully; missing
                # means caller will fall back to a fresh `cognee.search` inside
                # `answer()`.
                "context_only": True,
            },
        )

    async def answer(
        self,
        query: str,
        context: str,
        conversation_id: str,
        **kwargs,
    ) -> str:
        """Generate the final answer; reuse cached retrieval from `search()`.

        Fast path: when the harness calls `adapter.answer(query, context, conv_id,
        search_result=sr)`, we read `sr.retrieval_metadata["formatted_context"]`
        (the same context the harness already formatted) and skip the second
        graph traversal. The LLM call only — equivalent to BEAM's third step
        (`GraphCompletionRetriever.get_completion_from_context`) — runs via
        cognee's `generate_completion` helper.

        Fallback path: when no `search_result` is provided, repeat `cognee.search`
        with `only_context=False` so caller-side usage without prior
        `adapter.search()` still works.
        """
        self._require_cognee()

        # ---- Fast path: harness passes the prior SearchResult via kwargs ------
        cached_sr: Optional[SearchResult] = kwargs.get("search_result")
        cached_context: Optional[str] = None
        if cached_sr is not None:
            meta = getattr(cached_sr, "retrieval_metadata", None) or {}
            if meta.get("context_only") and meta.get("formatted_context"):
                cached_context = str(meta["formatted_context"])

        if cached_context is not None and _cognee_generate_completion is not None:
            try:
                completion = await _cognee_generate_completion(
                    query=query,
                    context=cached_context,
                    user_prompt_path="graph_context_for_question.txt",
                    system_prompt_path=self.system_prompt_path,
                )
                if isinstance(completion, str):
                    if completion:
                        return completion
                elif completion:
                    return str(completion)
                logger.warning(
                    "cognee generate_completion returned empty for q=%r — "
                    "falling back to context",
                    query[:60],
                )
            except Exception as exc:
                logger.warning(
                    "cognee generate_completion failed: %s — falling back to "
                    "cognee.search(only_context=False)",
                    exc,
                    exc_info=True,
                )

        # ---- Slow path: no cached retrieval available, redo the search ----------
        ds = self._dataset_name(conversation_id)
        top_k = kwargs.get("top_k", self.top_k)
        search_type = kwargs.get("search_type", self.search_type)

        try:
            results = await cognee.search(
                query_text=query,
                query_type=search_type,
                datasets=[ds],
                system_prompt_path=self.system_prompt_path,
                top_k=top_k,
                wide_search_top_k=self.wide_search_top_k,
                only_context=False,
            )
        except Exception as exc:
            logger.warning(
                "cognee answer fallback failed for dataset %s: %s — returning "
                "raw context",
                ds,
                exc,
                exc_info=True,
            )
            return context or ""

        for r in results or []:
            payload = getattr(r, "search_result", None)
            if payload is None:
                continue
            completion = getattr(payload, "completion", None)
            if completion:
                if isinstance(completion, list):
                    completion = next(
                        (c for c in completion if c), completion[0] if completion else ""
                    )
                if isinstance(completion, str):
                    return completion
                return str(completion)

        return context or ""

    # ----- introspection / lifecycle ------------------------------------------

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "CogneeAdapter",
            "config": self.config,
            "retriever_type": self.retriever_type,
            "dataset_prefix": self.dataset_prefix,
            "cognee_available": cognee is not None,
        }

    async def close(self) -> None:  # noqa: D401 — no resources to release
        """No-op. Cognee manages its own engines; nothing to close on this side."""
        return None
