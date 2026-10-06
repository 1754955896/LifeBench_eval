"""
GraphRAG-specific resource tracker.

Extends DefaultTracker (process RSS/CPU + LLM token usage) with:

* **Storage breakdown** — input / output / cache sizes summed across all
  conversation workspaces under ``{output_dir}/graphrag/workspaces/``
  (purely filesystem operations, always collected).
* **Graph metrics** — entity / relationship / text-unit / community /
  community-report counts from the indexed parquet outputs (best-effort;
  parquet files may be mid-write while indexing runs, so every read is
  guarded and row counts are cached by (mtime, size)).

Workspace layout measured (one dir per conversation):
  {workspace_root}/{conv_id}/{input,output,cache}/
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.trackers.system_trackers.base import SystemSnapshot, register_tracker
from src.trackers.system_trackers.default import DefaultTracker

logger = logging.getLogger(__name__)

# parquet table -> filename (as written by graphrag's index pipeline)
_PARQUET_TABLES = (
    ("entities", "entities.parquet"),
    ("relationships", "relationships.parquet"),
    ("text_units", "text_units.parquet"),
    ("communities", "communities.parquet"),
    ("community_reports", "community_reports.parquet"),
)


@register_tracker("graphrag")
class GraphRAGTracker(DefaultTracker):
    """GraphRAG-specific resource tracker."""

    def __init__(
        self,
        config: Optional[dict] = None,
        pid: Optional[int] = None,
        llm_proxy_url: Optional[str] = None,
    ):
        super().__init__(config, pid, llm_proxy_url)
        self._config = config or {}
        self._workspace_root: Optional[Path] = None
        # path -> ((mtime_ns, size), row_count)
        self._parquet_cache: Dict[str, tuple] = {}

    @property
    def system_name(self) -> str:
        return "graphrag"

    # -- LLM request transform ----------------------------------------------

    @staticmethod
    def transform_llm_request(data: dict) -> dict:
        """Inject thinking-disabled extra_body into forwarded chat requests.

        graphrag 2.7.2's litellm calls cannot disable the model's default
        thinking mode, so every extraction/community-report call wastes
        tokens on chain-of-thought (observed 10K+ completion tokens per
        call). Same fix as cognee.yaml's ``llm_args``: send
        ``extra_body={"thinking": {"type": "disabled"}}``.
        """
        data = dict(data)
        extra_body: dict = {}
        if "extra_body" in data:
            eb = data["extra_body"]
            if isinstance(eb, dict):
                extra_body.update(eb)
            elif isinstance(eb, str):
                import json as _json
                try:
                    extra_body.update(_json.loads(eb))
                except (TypeError, ValueError):
                    pass
        extra_body["thinking"] = {"type": "disabled"}
        data["extra_body"] = extra_body
        return data

    # -- workspace root resolution ----------------------------------------

    def _get_workspace_root(self) -> Optional[Path]:
        """Resolve the graphrag workspaces root directory.

        Priority: config output_dir (set by cli.py) → env
        GRAPHRAG_WORKSPACE_ROOT → default results/graphrag/workspaces.
        """
        if self._workspace_root is not None:
            return self._workspace_root

        output_dir = self._config.get("output_dir")
        if output_dir:
            candidate = Path(output_dir) / "graphrag" / "workspaces"
            if candidate.is_dir():
                self._workspace_root = candidate
                return self._workspace_root

        env_root = os.environ.get("GRAPHRAG_WORKSPACE_ROOT")
        if env_root:
            p = Path(env_root)
            if p.is_dir():
                self._workspace_root = p
                return self._workspace_root

        base = Path(__file__).resolve().parents[3]
        candidate = base / "results" / "graphrag" / "workspaces"
        if candidate.is_dir():
            self._workspace_root = candidate
            return self._workspace_root

        return None

    # -- size helpers ------------------------------------------------------

    @staticmethod
    def _get_dir_size_mb(path: Path) -> float:
        """Walk *path* and return total size in MB."""
        if not path or not path.is_dir():
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

    def _get_storage_breakdown(self) -> Dict[str, float]:
        """Total + per-subdir sizes summed across all conversation workspaces."""
        root = self._get_workspace_root()
        if not root:
            return {}

        sizes = {"input": 0.0, "output": 0.0, "cache": 0.0}
        for conv_dir in root.iterdir():
            if not conv_dir.is_dir():
                continue
            for sub in sizes:
                sizes[sub] += self._get_dir_size_mb(conv_dir / sub)

        total = round(sizes["input"] + sizes["output"] + sizes["cache"], 3)
        return {
            "total_mb": total,
            "input_mb": round(sizes["input"], 3),
            "output_mb": round(sizes["output"], 3),
            "cache_mb": round(sizes["cache"], 3),
        }

    # -- graph metrics (best-effort) ---------------------------------------

    def _count_parquet_rows(self, path: Path) -> Optional[int]:
        """Row count of a parquet file, cached by (mtime_ns, size)."""
        try:
            st = path.stat()
            stat_key = (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

        cached = self._parquet_cache.get(str(path))
        if cached is not None and cached[0] == stat_key:
            return cached[1]

        try:
            import pandas as pd
            row_count = len(pd.read_parquet(path))
        except Exception:
            return None

        self._parquet_cache[str(path)] = (stat_key, row_count)
        return row_count

    def _collect_graph_metrics(self) -> Optional[Dict[str, Any]]:
        """Aggregate table row counts across all indexed conversations."""
        root = self._get_workspace_root()
        if not root:
            return None

        totals = {name: 0 for name, _ in _PARQUET_TABLES}
        any_data = False
        indexed_convs = 0

        for conv_dir in root.iterdir():
            if not conv_dir.is_dir():
                continue
            out = conv_dir / "output"
            if not out.is_dir():
                continue
            indexed_convs += 1
            for name, fname in _PARQUET_TABLES:
                count = self._count_parquet_rows(out / fname)
                if count is not None:
                    totals[name] += count
                    any_data = True

        if not any_data:
            return None
        totals["conversations_indexed"] = indexed_convs
        return totals

    # -- main snapshot ------------------------------------------------------

    def snapshot(self) -> SystemSnapshot:
        """Take a resource snapshot with graphrag-specific layers.

        ``extra`` carries ``storage_breakdown`` (input/output/cache MB) and
        ``graph_metrics`` (entity/relationship/text-unit/community counts),
        plus all parent DefaultTracker fields (RSS, CPU, LLM tokens).
        """
        snapshot = super().snapshot()

        storage = self._get_storage_breakdown()
        if storage:
            snapshot.extra["storage_breakdown"] = storage
            snapshot.storage_mb = storage.get("total_mb", 0.0)

        graph_metrics = self._collect_graph_metrics()
        if graph_metrics:
            snapshot.extra["graph_metrics"] = graph_metrics

        return snapshot

    # -- backend summary ------------------------------------------------------

    def backend_specific_stats(self) -> Dict[str, Any]:
        """Return graphrag backend summary (storage + graph metrics)."""
        stats = super().backend_specific_stats()

        storage = self._get_storage_breakdown()
        if storage:
            stats["storage_mb"] = storage.get("total_mb", 0.0)
            stats["storage_breakdown"] = storage

        graph_metrics = self._collect_graph_metrics()
        if graph_metrics:
            stats["graph_metrics"] = graph_metrics

        return stats