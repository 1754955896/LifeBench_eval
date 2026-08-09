"""
MindMemOS-specific resource tracker.

MindMemOS runs locally (FastAPI server + Qdrant/Neo4j/Kafka docker stack), so
unlike cloud-API systems its backend resources ARE visible from the eval host.
Extends ``DefaultTracker`` with:

* MindMemOS API server process (uvicorn ``mindmemos.api.app``) CPU/RSS via
  psutil — matched once by command line and cached.
* Qdrant collection point counts (memory_item_v1 / entity_item_v1 /
  source_item_v1) via the Qdrant REST API (best-effort, TTL-cached).
* Docker container stats for the ``mindmemos-*`` stack (qdrant/neo4j/kafka)
  via ``docker stats --no-stream`` (best-effort, TTL-cached).
* Docker volume storage for the mindmemos stack via ``docker system df -v``.
* API latency aggregates from add_latency.json / search_latency.json
  (requires ``output_dir`` in the tracker config).

Parent ``DefaultTracker`` supplies the eval process RSS/CPU and LLM token
totals polled from the llm_proxy (answer/evaluate phases routed through the
proxy). All backend reads run on the GlobalMonitor sampling thread, so they
fail fast / degrade to empty rather than block.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.trackers.system_trackers.base import SystemSnapshot, register_tracker
from src.trackers.system_trackers.default import DefaultTracker

logger = logging.getLogger(__name__)

# Volume name suffixes from systems/MindMemOS/dockers/docker-compose.memory.yml.
# docker compose prefixes them with the project name, so match by suffix.
_VOLUME_SUFFIXES = ("qdrant_storage", "neo4j_data", "neo4j_logs", "kafka_data")

# Qdrant collections created by the MindMemOS schema (config/database.qdrant).
_QDRANT_COLLECTIONS = ("memory_item_v1", "entity_item_v1", "source_item_v1")

# Containers started by the MindMemOS compose file.
_CONTAINER_NAMES = ("mindmemos-qdrant", "mindmemos-neo4j", "mindmemos-kafka")

# Match the uvicorn command line of the MindMemOS API server.
_SERVER_CMDLINE_MARKER = "mindmemos.api.app"

_QDRANT_URL = "http://localhost:6333"

# Longest units first so '1.5GB' matches GB, not the trailing 'B'.
# docker stats reports binary units (MiB/GiB), docker system df uses decimal.
_SIZE_UNITS = {
    "TiB": 1 << 40, "TB": 1e12,
    "GiB": 1 << 30, "GB": 1e9,
    "MiB": 1 << 20, "MB": 1e6,
    "KiB": 1 << 10, "kB": 1e3,
    "B": 1.0,
}

_STORAGE_TTL = 30.0        # cache successful `docker system df -v` results
_STATS_TTL = 15.0          # cache successful `docker stats` results
_COUNTS_TTL = 15.0         # cache successful qdrant count reads
_RETRY_AFTER = 30.0        # wait before retrying after a backend failure


def _parse_size(text: str) -> float:
    """Parse a docker human-readable size ('1.5GB', '12.3MiB', '0B') to bytes."""
    text = (text or "").strip()
    if not text or text == "N/A":
        return 0.0
    for unit, factor in _SIZE_UNITS.items():
        if text.endswith(unit):
            try:
                return float(text[: -len(unit)]) * factor
            except ValueError:
                return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_mem_usage(text: str) -> float:
    """Parse docker MemUsage '123.4MiB / 4GiB' into the used bytes."""
    used = (text or "").split("/")[0].strip()
    return _parse_size(used)


class _TTLCache:
    """Caches a producer's result for ``ttl`` seconds; after a failure,
    retries are throttled to ``retry_after`` seconds."""

    def __init__(self, ttl: float, retry_after: float):
        self.ttl = ttl
        self.retry_after = retry_after
        self._value: Optional[Any] = None
        self._ts: float = 0.0
        self._fail_ts: float = 0.0

    def get(self, producer) -> Optional[Any]:
        now = time.time()
        if self._value is not None and now - self._ts < self.ttl:
            return self._value
        if self._fail_ts and now - self._fail_ts < self.retry_after:
            return None
        try:
            value = producer()
        except Exception as exc:
            logger.debug("TTL cache producer failed: %s", exc)
            self._fail_ts = now
            return None
        self._value = value
        self._ts = now
        self._fail_ts = 0.0
        return value


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


@register_tracker("mindmemos")
class MindMemosTracker(DefaultTracker):
    """MindMemOS-specific resource tracker.

    ``snapshot()`` inherits ``DefaultTracker`` (eval process RSS/CPU + LLM
    tokens from the llm_proxy) and adds the MindMemOS server process load,
    Qdrant record counts, docker container stats and volume storage.
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
        self._server_pid: Optional[int] = None
        self._server_proc = None
        self._stats_cache = _TTLCache(_STATS_TTL, _RETRY_AFTER)
        self._counts_cache = _TTLCache(_COUNTS_TTL, _RETRY_AFTER)
        self._storage_cache = _TTLCache(_STORAGE_TTL, _RETRY_AFTER)

    @property
    def system_name(self) -> str:
        return "mindmemos"

    # -- MindMemOS API server process ----------------------------------------

    def _find_server_process(self):
        """Locate the MindMemOS uvicorn worker once (match by cmdline).

        ``uv run uvicorn`` spawns a parent wrapper plus the real worker; both
        carry the marker in their cmdline. The worker is the one actually
        serving, so among matches we keep the process with the largest RSS.
        """
        if self._server_proc is not None or self._server_pid is not None:
            return self._server_proc
        try:
            import psutil

            best = None
            best_rss = -1.0
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    rss = proc.memory_info().rss
                except Exception:
                    continue
                if _SERVER_CMDLINE_MARKER in cmdline and "uvicorn" in cmdline and rss > best_rss:
                    best = proc
                    best_rss = rss
            if best is not None:
                self._server_pid = best.pid
                self._server_proc = best
                return self._server_proc
        except Exception as exc:
            logger.debug("MindMemOS server process lookup failed: %s", exc)
        return None

    def _server_stats(self) -> Dict[str, Any]:
        """CPU/RSS of the MindMemOS API server process (best-effort)."""
        proc = self._find_server_process()
        if proc is None:
            return {}
        try:
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            return {
                "server_pid": proc.pid,
                "server_memory_rss_mb": round(rss_mb, 2),
                "server_cpu_percent": round(proc.cpu_percent(interval=None), 2),
            }
        except Exception as exc:
            logger.debug("MindMemOS server stats failed: %s", exc)
            return {}

    # -- Qdrant record counts -------------------------------------------------

    def _get_qdrant_counts(self) -> Optional[Dict[str, Any]]:
        return self._counts_cache.get(self._read_qdrant_counts)

    def _read_qdrant_counts(self) -> Dict[str, Any]:
        import httpx

        counts: Dict[str, Any] = {}
        with httpx.Client(timeout=2.0) as client:
            for name in _QDRANT_COLLECTIONS:
                try:
                    resp = client.get(f"{_QDRANT_URL}/collections/{name}")
                    resp.raise_for_status()
                    data = resp.json()
                    counts[name] = data.get("result", {}).get("points_count", 0)
                except Exception as exc:
                    logger.debug("Qdrant collection %s failed: %s", name, exc)
                    raise
        if not counts:
            raise RuntimeError("Qdrant unreachable")
        return counts

    # -- Docker containers & volumes ------------------------------------------

    def _get_container_stats(self) -> Optional[Dict[str, Any]]:
        return self._stats_cache.get(self._read_container_stats)

    def _read_container_stats(self) -> Dict[str, Any]:
        result = subprocess.run(
            [
                "docker", "stats", "--no-stream", "--format",
                "{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"docker exit code {result.returncode}")

        stats: Dict[str, Any] = {}
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name, mem_usage, cpu_perc = parts[0], parts[1], parts[2]
            if name in _CONTAINER_NAMES:
                stats[name] = {
                    "memory_mb": round(_parse_mem_usage(mem_usage) / (1024 * 1024), 3),
                    "cpu_percent": round(float(cpu_perc.rstrip("%") or 0.0), 2),
                }
        if not stats:
            raise RuntimeError("no mindmemos containers found")
        return stats

    def _get_storage(self) -> Dict[str, float]:
        return self._storage_cache.get(self._read_storage) or {}

    def _read_storage(self) -> Dict[str, float]:
        result = subprocess.run(
            ["docker", "system", "df", "-v"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"docker exit code {result.returncode}")
        return self._parse_docker_df(result.stdout)

    @staticmethod
    def _parse_docker_df(output: str) -> Dict[str, float]:
        """Extract named-volume sizes from ``docker system df -v`` output."""
        sizes: Dict[str, float] = {}
        in_table = False
        for line in output.splitlines():
            if "VOLUME NAME" in line:
                in_table = True
                continue
            if not in_table or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name, size_text = parts[0], parts[-1]
            if name.endswith(_VOLUME_SUFFIXES):
                sizes[name] = _parse_size(size_text)
        return sizes

    # -- main snapshot --------------------------------------------------------

    def snapshot(self) -> SystemSnapshot:
        """Take a resource snapshot with MindMemOS backend metrics.

        ``extra`` carries the parent ``DefaultTracker`` fields plus:
        * ``mindmemos_server`` — API server process CPU/RSS
        * ``qdrant`` — collection point counts
        * ``docker_containers`` — container memory/CPU
        * ``storage_breakdown`` — per-backend docker volume sizes (MB)
        """
        snapshot = super().snapshot()

        server_stats = self._server_stats()
        if server_stats:
            snapshot.extra["mindmemos_server"] = server_stats

        counts = self._get_qdrant_counts()
        if counts:
            snapshot.extra["qdrant"] = counts

        container_stats = self._get_container_stats()
        if container_stats:
            snapshot.extra["docker_containers"] = container_stats

        breakdown = self._get_storage()
        if breakdown:
            snapshot.extra["storage_breakdown"] = {
                name: round(size / (1024 * 1024), 3) for name, size in breakdown.items()
            }
            snapshot.storage_mb = round(sum(breakdown.values()) / (1024 * 1024), 3)

        return snapshot

    # -- backend summary ------------------------------------------------------

    def backend_specific_stats(self) -> Dict[str, Any]:
        """Return MindMemOS backend summary + latency aggregates."""
        stats = super().backend_specific_stats()

        server_stats = self._server_stats()
        if server_stats:
            stats["mindmemos_server"] = server_stats
        counts = self._get_qdrant_counts()
        if counts:
            stats["qdrant"] = counts
        container_stats = self._get_container_stats()
        if container_stats:
            stats["docker_containers"] = container_stats
        breakdown = self._get_storage()
        if breakdown:
            stats["storage_breakdown"] = {
                name: round(size / (1024 * 1024), 3) for name, size in breakdown.items()
            }

        if self._output_dir:
            latency: Dict[str, Any] = {}
            for filename in ("add_latency.json", "search_latency.json"):
                agg = _aggregate_latency(self._output_dir, filename)
                if agg:
                    latency[filename.removesuffix(".json")] = agg
            if latency:
                stats["latency"] = latency
        return stats
