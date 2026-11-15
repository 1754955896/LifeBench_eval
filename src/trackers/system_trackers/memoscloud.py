"""
Memos Cloud resource tracker.

Remote API backend (memos.memtensor.cn) — the memory system runs on the
server, so CPU/RAM/storage of the memory system itself are NOT visible
from the client.  This tracker records what is measurable locally:

* Local test process load (inherited from DefaultTracker: psutil RSS/CPU)
* LLM token usage for the answer / evaluate phases (via llm_proxy, when
  those phases route their /chat/completions calls through the proxy)
* API latency aggregates from the evaluation result files
  (add_latency.json / search_latency.json)

Server-side LLM tokens consumed by add/search are invisible: the API
responses carry no usage field.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.trackers.system_trackers.base import register_tracker
from src.trackers.system_trackers.default import DefaultTracker

logger = logging.getLogger(__name__)


def _aggregate_latency(output_dir: Path, filename: str) -> Optional[Dict[str, Any]]:
    """Aggregate latency stats from a results file (e.g. add_latency.json)."""
    path = output_dir / filename
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.debug("Failed to load %s: %s", filename, exc)
        return None
    if not isinstance(data, list) or not data:
        return None

    times = sorted(float(d.get("latency_seconds", 0.0)) for d in data)
    n = len(times)
    total = sum(times)
    stats: Dict[str, Any] = {
        "count": n,
        "total_seconds": round(total, 3),
        "avg_seconds": round(total / n, 3),
        "p50_seconds": round(times[n // 2], 3),
        "p95_seconds": round(times[int(n * 0.95)], 3),
        "max_seconds": round(times[-1], 3),
    }
    if all("added" in d and "failed" in d for d in data):
        stats["chunks_added"] = sum(int(d.get("added", 0)) for d in data)
        stats["chunks_failed"] = sum(int(d.get("failed", 0)) for d in data)
    return stats


@register_tracker("memos_cloud")
class MemosCloudTracker(DefaultTracker):
    """Remote-API resource tracker for Memos Cloud.

    ``snapshot()`` is inherited from ``DefaultTracker`` (process RSS/CPU +
    LLM token totals polled from the llm_proxy); ``backend_specific_stats()``
    additionally aggregates add/search API latency from the evaluation
    output files.  Requires ``output_dir`` in the tracker config.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        pid: Optional[int] = None,
        llm_proxy_url: Optional[str] = None,
    ):
        super().__init__(config, pid, llm_proxy_url)
        self._output_dir: Optional[Path] = None
        if config and config.get("output_dir"):
            self._output_dir = Path(config["output_dir"])

    @property
    def system_name(self) -> str:
        return "memos_cloud"

    def _latency_stats(self) -> Dict[str, Any]:
        if not self._output_dir:
            return {}
        stats: Dict[str, Any] = {}
        for filename in ("add_latency.json", "search_latency.json"):
            agg = _aggregate_latency(self._output_dir, filename)
            if agg:
                stats[filename.removesuffix(".json")] = agg
        return stats

    def backend_specific_stats(self) -> Dict[str, Any]:
        """Return process + LLM token stats plus API latency aggregates."""
        stats = super().backend_specific_stats()
        latency = self._latency_stats()
        if latency:
            stats["latency"] = latency
        return stats
