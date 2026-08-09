"""
Cognee-specific resource tracker.

Extends DefaultTracker with cognee's three-tier storage measurement
(relational / graph / vector / raw data), knowledge-graph structural
metrics (node/edge counts, density, connectivity), and dataset-level
statistics (count, per-dataset data items).

Cognee storage architecture
----------------------------
  Relational (SQLite)   …  metadata, Data/Dataset records, pipeline state
  Graph (Kuzu/Ladybug)  …  knowledge graph nodes + edges (entity extraction)
  Vector (LanceDB)      …  chunk / entity / triplet embeddings
  Raw data              …  original ingested files (DATA_ROOT_DIRECTORY)

All file-size metrics are purely filesystem operations (always available).
Graph and dataset metrics use an async→sync bridge and are best-effort —
failures are silently ignored so `snapshot()` never blocks.
"""

import asyncio
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.trackers.system_trackers.base import SystemSnapshot, register_tracker
from src.trackers.system_trackers.default import DefaultTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM proxy helpers — TOOLS mode ↔ JSON mode conversion
# ---------------------------------------------------------------------------
# DeepSeek ignores OpenAI-style tool_calls and returns plain text. instructor
# (TOOLS mode) then fails because it expects exactly 1 tool call.  We convert:
#
#   Request:  tools + tool_choice  →  response_format: json_object
#   Response: JSON text content    →  synthetic tool_call wrapper
#
# The proxy calls transform_llm_request() before forwarding, and
# _wrap_response_as_tool_call() after receiving the DeepSeek response.

def _convert_tools_to_json_mode(data: dict, _json) -> None:
    """Convert instructor TOOLS-mode params to JSON mode (in-place on *data*).

    Stashes tool metadata in ``data["_proxy_tool_info"]`` so the response
    wrapper can reconstruct the tool-call envelope.
    """
    tools = data.get("tools")
    tool_choice = data.get("tool_choice")

    if not tools or not tool_choice:
        return

    # Extract the target function schema
    tool_func = tool_choice.get("function") if isinstance(tool_choice, dict) else None
    tool_name = (tool_func or {}).get("name", "")
    schema = None
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        if fn.get("name") == tool_name:
            schema = fn.get("parameters", {})
            break

    if not schema:
        return

    data["_proxy_tool_info"] = {"name": tool_name, "schema": schema}
    data.pop("tools", None)
    data.pop("tool_choice", None)
    data["response_format"] = {"type": "json_object"}

    # Inject schema instruction so the model knows the expected shape
    schema_str = _json.dumps(schema, ensure_ascii=False)
    messages = data.setdefault("messages", [])
    messages.append({
        "role": "user",
        "content": (
            "You must respond with exactly one JSON object matching this schema:\n"
            + schema_str
            + "\n\nOutput ONLY the JSON object. No markdown, no explanation."
        ),
    })
    data["messages"] = messages


def _wrap_response_as_tool_call(response_dict: dict,
                                tool_info: dict | None) -> dict:
    """Wrap a JSON-mode response as a TOOLS-mode response for instructor.

    Returns *response_dict* unchanged when *tool_info* is ``None`` or the
    response already contains tool calls.
    """
    if tool_info is None:
        return response_dict

    choices = response_dict.get("choices", [])
    if not choices:
        return response_dict

    message = choices[0].get("message", {})
    # Don't double-wrap
    if message.get("tool_calls"):
        return response_dict

    content = message.get("content", "")
    if not content:
        return response_dict

    # Strip markdown code fences that some providers wrap around JSON
    stripped = content.strip()
    for fence in ("```json", "```"):
        if stripped.startswith(fence):
            stripped = stripped[len(fence):].lstrip("\n\r")
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip("\n\r")

    # Move text content into a synthetic tool call
    tool_name = tool_info["name"]
    import uuid as _uuid
    return {
        **response_dict,
        "choices": [{
            **choices[0],
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_" + _uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": stripped,
                    },
                }],
            },
        }],
    }


# ---------------------------------------------------------------------------
# async → sync bridge
# ---------------------------------------------------------------------------


def _run_async_safe(coro, *, timeout: float = 5.0):
    """Run an async coroutine synchronously, returning its result or ``None``.

    Handles both contexts safely:

    * No running event loop (e.g. ``GlobalMonitor`` daemon thread) →
      ``asyncio.run()``.
    * Running event loop (e.g. inside ``Pipeline.run()``) →
      offloads to a temporary thread with its own fresh loop.
    """
    if coro is None:
        return None

    async def _guarded():
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except Exception:
            return None

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — simplest path.
        try:
            return asyncio.run(_guarded())
        except Exception:
            return None
    else:
        # A loop is already running; run in a fresh thread.
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, _guarded())
                return fut.result(timeout=timeout + 2)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# CogneeTracker
# ---------------------------------------------------------------------------


@register_tracker("cognee")
class CogneeTracker(DefaultTracker):
    """Cognee-specific resource tracker.

    Adds three dimensions on top of ``DefaultTracker`` (process RSS/CPU
    + LLM token usage):

    * **Storage breakdown** — sizes of the relational, graph, vector, and
      raw-data directories (purely filesystem-based, always collected).
    * **Graph metrics** — node/edge counts, mean degree, density,
      connected components (best-effort, via cognee's graph engine).
    * **Dataset stats** — count and per-dataset data-item counts
      (best-effort, via ``cognee.datasets``).

    All async-derived metrics carry a 5-second timeout; failures are
    swallowed so ``snapshot()`` latency stays predictable.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        pid: Optional[int] = None,
        llm_proxy_url: Optional[str] = None,
    ):
        super().__init__(config, pid, llm_proxy_url)
        self._config = config or {}
        self._system_root: Optional[Path] = None
        self._data_root: Optional[Path] = None
        self._cognee_available: Optional[bool] = None

    # -- public markers -------------------------------------------------------

    @property
    def system_name(self) -> str:
        return "cognee"

    # -- LLM request transform ------------------------------------------------

    # Known OpenAI SDK top-level parameters. Anything else gets funneled into
    # ``extra_body`` so the SDK call doesn't choke on unknown kwargs (e.g.
    # liteLLM expanding ``extra_body.thinking`` as a top-level ``thinking`` key).
    _SDK_TOP_LEVEL = frozenset({
        "model", "messages",
        "max_tokens", "max_completion_tokens",
        "temperature", "top_p", "top_k",
        "stream", "stream_options",
        "stop", "n",
        "frequency_penalty", "presence_penalty",
        "logprobs", "top_logprobs", "logit_bias",
        "user", "seed",
        "response_format",
        "tools", "tool_choice",
        "parallel_tool_calls",
        "functions", "function_call",
        "metadata", "store",
        "reasoning_effort",
        "service_tier",
        "modalities", "audio",
    })

    @staticmethod
    def transform_llm_request(data: dict) -> dict:
        """Normalize instructor TOOLS-mode requests for DeepSeek compatibility.

        DeepSeek models ignore OpenAI-style ``tools`` / ``tool_choice`` and
        return plain text instead of tool calls.  instructor then fails with
        *"Instructor does not support multiple tool calls"* because it sees
        zero tool calls.

        This method converts TOOLS-mode requests into JSON-mode requests
        (``response_format: json_object``) which DeepSeek honours.  The
        companion response wrapper in the proxy (``_wrap_response_as_tool_call``)
        converts the JSON text back into a synthetic tool-call so instructor
        can parse it transparently.

        Metadata (tool name, original schema) is stashed under
        ``_proxy_tool_info`` for the response wrapper.
        """
        import json as _json

        data = dict(data)  # shallow copy

        # ── Step 1: TOOLS mode → JSON mode ──────────────────────────────
        _convert_tools_to_json_mode(data, _json)

        # ── Step 2: fold unknown keys into extra_body ───────────────────
        extra_body: dict = {}
        if "extra_body" in data:
            eb = data.pop("extra_body")
            if isinstance(eb, dict):
                extra_body.update(eb)
            elif isinstance(eb, str):
                try:
                    extra_body.update(_json.loads(eb))
                except (TypeError, ValueError):
                    pass

        for key in list(data.keys()):
            if key in ("_proxy_tool_info",):
                continue
            if key not in CogneeTracker._SDK_TOP_LEVEL:
                extra_body[key] = data.pop(key)

        if extra_body:
            data["extra_body"] = extra_body

        return data

    @staticmethod
    def wrap_llm_response(response_dict: dict,
                          tool_info: dict | None) -> dict:
        """Wrap JSON-mode response as TOOLS-mode for instructor."""
        return _wrap_response_as_tool_call(response_dict, tool_info)

    # -- lazy import guard ----------------------------------------------------

    def _check_cognee(self) -> bool:
        """``True`` if the cognee SDK is importable.  Cached after first call."""
        if self._cognee_available is None:
            try:
                import cognee  # noqa: F401
                self._cognee_available = True
            except ImportError:
                self._cognee_available = False
        return self._cognee_available

    # -- path resolution (3-tier: cognee config → env → default) --------------

    def _get_system_root(self) -> Optional[Path]:
        """Resolve ``SYSTEM_ROOT_DIRECTORY`` (databases live here)."""
        if self._system_root is not None:
            return self._system_root

        # Tier 1 — cognee's own BaseConfig (most authoritative)
        if self._check_cognee():
            try:
                from cognee.base_config import get_base_config
                cfg = get_base_config()
                path = cfg.system_root_directory
                if path:
                    self._system_root = Path(path)
                    return self._system_root
            except Exception:
                pass

        # Tier 2 — environment variable
        for env_var in ("SYSTEM_ROOT_DIRECTORY",):
            val = os.environ.get(env_var)
            if val:
                p = Path(val)
                if p.exists():
                    self._system_root = p
                    return self._system_root

        # Tier 3 — derive from known locations relative to systems/cognee
        base = Path(__file__).resolve().parents[3] / "systems" / "cognee"
        for candidate in (
            base / ".cognee" / "system",
            base / ".cognee_system",
        ):
            if candidate.exists():
                self._system_root = candidate
                return self._system_root

        return None

    def _get_data_root(self) -> Optional[Path]:
        """Resolve ``DATA_ROOT_DIRECTORY`` (raw ingested files live here)."""
        if self._data_root is not None:
            return self._data_root

        # Tier 1 — cognee BaseConfig
        if self._check_cognee():
            try:
                from cognee.base_config import get_base_config
                cfg = get_base_config()
                path = cfg.data_root_directory
                if path:
                    self._data_root = Path(path)
                    return self._data_root
            except Exception:
                pass

        # Tier 2 — environment variable
        for env_var in ("DATA_ROOT_DIRECTORY",):
            val = os.environ.get(env_var)
            if val:
                p = Path(val)
                if p.exists():
                    self._data_root = p
                    return self._data_root

        # Tier 3 — derive from known locations
        base = Path(__file__).resolve().parents[3] / "systems" / "cognee"
        for candidate in (
            base / ".cognee" / "data",
            base / ".cognee_data",
        ):
            if candidate.exists():
                self._data_root = candidate
                return self._data_root

        return None

    # -- directory / file size helpers ----------------------------------------

    @staticmethod
    def _get_dir_size_mb(path: Path) -> float:
        """Walk *path* and return total size in MB."""
        if not path or not path.exists() or not path.is_dir():
            return 0.0
        total = 0
        try:
            for dirpath, _dirnames, filenames in os.walk(path):
                for f in filenames:
                    try:
                        total += (Path(dirpath) / f).stat().st_size
                    except OSError:
                        pass
        except Exception:
            pass
        return total / (1024 * 1024)

    @staticmethod
    def _get_file_size_mb(path: Path) -> float:
        """Return size of a single file in MB."""
        try:
            return path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    # -- storage breakdown (sync – filesystem only) ---------------------------

    def _get_relational_db_size(self) -> float:
        """SQLite database(s) size under ``<system_root>/databases/``."""
        system_root = self._get_system_root()
        if not system_root:
            return 0.0
        db_dir = system_root / "databases"
        if not db_dir.exists():
            return 0.0
        total = 0.0
        for db_file in db_dir.glob("*.db"):
            total += self._get_file_size_mb(db_file)
        # Some configurations use "cognee_db" (no extension)
        cognee_db = db_dir / "cognee_db"
        total += self._get_file_size_mb(cognee_db)
        return total

    def _get_graph_db_size(self) -> float:
        """Kuzu / Ladybug graph storage size (file *or* directory)."""
        system_root = self._get_system_root()
        if not system_root:
            return 0.0
        db_dir = system_root / "databases"
        if not db_dir.exists():
            return 0.0
        total = 0.0
        for graph_path in db_dir.glob("cognee_graph_*"):
            if graph_path.is_dir():
                total += self._get_dir_size_mb(graph_path)
            else:
                total += self._get_file_size_mb(graph_path)
        return total

    def _get_vector_db_size(self) -> float:
        """LanceDB vector directory / file size."""
        system_root = self._get_system_root()
        if not system_root:
            return 0.0
        db_dir = system_root / "databases"
        if not db_dir.exists():
            return 0.0
        total = 0.0
        for vec_path in db_dir.glob("cognee.lancedb*"):
            if vec_path.is_dir():
                total += self._get_dir_size_mb(vec_path)
            else:
                total += self._get_file_size_mb(vec_path)
        return total

    def _get_raw_data_size(self) -> float:
        """Raw ingested files under ``DATA_ROOT_DIRECTORY``."""
        data_root = self._get_data_root()
        if not data_root:
            return 0.0
        return self._get_dir_size_mb(data_root)

    def _db_ready(self) -> bool:
        """True if the databases directory exists and has been set up."""
        system_root = self._get_system_root()
        if not system_root:
            return False
        db_dir = system_root / "databases"
        return db_dir.exists() and any(db_dir.iterdir())

    # -- graph metrics (async → sync, best-effort) ---------------------------

    def _collect_graph_metrics(self) -> Optional[Dict[str, Any]]:
        """Graph structure metrics via ``get_graph_engine().get_graph_metrics()``."""
        if not self._check_cognee() or not self._db_ready():
            return None

        async def _get():
            from cognee.infrastructure.databases.graph import get_graph_engine
            engine = await get_graph_engine()
            return await engine.get_graph_metrics(include_optional=False)

        return _run_async_safe(_get(), timeout=5.0)

    # -- dataset stats (async → sync, best-effort) ----------------------------

    def _collect_dataset_stats(self) -> Optional[Dict[str, Any]]:
        """Per-dataset data-item counts via ``cognee.datasets``."""
        if not self._check_cognee() or not self._db_ready():
            return None

        async def _get():
            import cognee
            datasets = await cognee.datasets.list_datasets()
            if not datasets:
                return {"count": 0, "details": []}

            details: list[dict] = []
            for ds in datasets:
                ds_id = getattr(ds, "id", None)
                ds_name = getattr(
                    ds, "name", getattr(ds, "dataset_name", str(ds))
                )
                data_items = 0
                if ds_id:
                    try:
                        data = await cognee.datasets.list_data(ds_id)
                        data_items = len(data) if data else 0
                    except Exception:
                        pass
                details.append({
                    "name": str(ds_name),
                    "data_items": data_items,
                })

            return {"count": len(datasets), "details": details}

        return _run_async_safe(_get(), timeout=5.0)

    # -- main snapshot --------------------------------------------------------

    def snapshot(self) -> SystemSnapshot:
        """Take a resource snapshot with cognee-specific layers.

        Returns a ``SystemSnapshot`` whose ``extra`` dict carries:

        * ``storage_breakdown`` — relational / graph / vector / raw_data (MB)
        * ``graph_metrics`` — node/edge counts, density, connectivity
          (best-effort)
        * ``datasets`` — count + per-dataset data-item counts (best-effort)
        * plus all parent ``DefaultTracker`` fields (RSS, CPU, LLM tokens)
        """
        snapshot = super().snapshot()

        # ── storage (always available) ──────────────────────────────────
        rel_mb = self._get_relational_db_size()
        graph_mb = self._get_graph_db_size()
        vec_mb = self._get_vector_db_size()
        data_mb = self._get_raw_data_size()

        snapshot.extra["storage_breakdown"] = {
            "relational_db_mb": round(rel_mb, 3),
            "graph_db_mb": round(graph_mb, 3),
            "vector_db_mb": round(vec_mb, 3),
            "raw_data_mb": round(data_mb, 3),
        }
        # Override DefaultTracker's 0.0 — cognee has measurable on-disk state.
        snapshot.storage_mb = round(rel_mb + graph_mb + vec_mb + data_mb, 3)

        # ── graph structure (best-effort) ───────────────────────────────
        graph_metrics = self._collect_graph_metrics()
        if graph_metrics:
            snapshot.extra["graph_metrics"] = graph_metrics

        # ── datasets (best-effort) ──────────────────────────────────────
        dataset_stats = self._collect_dataset_stats()
        if dataset_stats:
            snapshot.extra["datasets"] = dataset_stats

        return snapshot

    # -- backend summary ------------------------------------------------------

    def backend_specific_stats(self) -> Dict[str, Any]:
        """Return cognee backend summary (storage + graph metrics)."""
        stats = super().backend_specific_stats()

        stats.update({
            "relational_db_mb": round(self._get_relational_db_size(), 3),
            "graph_db_mb": round(self._get_graph_db_size(), 3),
            "vector_db_mb": round(self._get_vector_db_size(), 3),
            "raw_data_mb": round(self._get_raw_data_size(), 3),
        })

        graph_metrics = self._collect_graph_metrics()
        if graph_metrics:
            stats["graph_metrics"] = graph_metrics

        return stats