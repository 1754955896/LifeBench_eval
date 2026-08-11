"""
Mem0-specific resource tracker.

Extends DefaultTracker with Docker-based mem0 deployment metrics:
- Container CPU/memory/IO usage via docker stats
- PostgreSQL/pgvector storage size
- LLM token usage via llm_proxy (inherited from DefaultTracker)
"""
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from src.trackers.system_trackers.base import SystemSnapshot, register_tracker
from src.trackers.system_trackers.default import DefaultTracker

logger = logging.getLogger(__name__)


@register_tracker("mem0")
class Mem0Tracker(DefaultTracker):
    """Mem0-specific resource tracker.

    Extends DefaultTracker with:
    - Docker container CPU/memory/IO metrics
    - PostgreSQL/pgvector storage size and row count
    - LLM token usage via llm_proxy (inherited)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        pid: Optional[int] = None,
        llm_proxy_url: Optional[str] = None,
    ):
        super().__init__(config, pid, llm_proxy_url)
        self._config = config or {}
        self._compose_file = self._config.get("docker_compose")
        self._compose_dir: Optional[Path] = None

    @property
    def system_name(self) -> str:
        return "mem0"

    # -- compose path resolution --------------------------------------------

    def _get_compose_dir(self) -> Optional[Path]:
        """Resolve the directory containing docker-compose.yaml."""
        if self._compose_dir is not None:
            return self._compose_dir

        compose_file = self._compose_file
        if not compose_file:
            return None

        base = Path(__file__).resolve().parents[3]
        candidate = base / compose_file
        if candidate.exists():
            self._compose_dir = candidate.parent
            return self._compose_dir

        return None

    def _get_compose_path(self) -> Optional[Path]:
        compose_dir = self._get_compose_dir()
        if not compose_dir:
            return None
        path = compose_dir / "docker-compose.yaml"
        return path if path.exists() else None

    # -- container stats ----------------------------------------------------

    @staticmethod
    def _parse_mem_usage_mb(mem_usage: str) -> float:
        """Parse docker mem_usage like '271.8MiB / 7.69GiB' to MB."""
        if not mem_usage:
            return 0.0
        part = mem_usage.split("/")[0].strip()
        try:
            if part.endswith("GiB"):
                return float(part[:-3].strip()) * 1024
            elif part.endswith("MiB"):
                return float(part[:-3].strip())
            elif part.endswith("KiB"):
                return float(part[:-3].strip()) / 1024
            elif part.endswith("GB"):
                return float(part[:-2].strip()) * 1000
            elif part.endswith("MB"):
                return float(part[:-2].strip())
            elif part.endswith("kB"):
                return float(part[:-2].strip()) / 1000
        except ValueError:
            pass
        return 0.0

    def _get_container_stats(self) -> Dict[str, Any]:
        """Get CPU/memory/IO for each mem0-related docker container.

        Uses ``docker compose ps -q`` to discover container IDs, then
        ``docker stats --no-stream`` for live resource usage.
        """
        compose_file = self._get_compose_path()
        if not compose_file:
            return {}

        stats: Dict[str, Any] = {}
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "ps", "-q"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return {}

            container_ids = [
                c.strip() for c in result.stdout.strip().split("\n") if c.strip()
            ]
            if not container_ids:
                return {}

            for cid in container_ids:
                try:
                    r = subprocess.run(
                        [
                            "docker", "stats", "--no-stream",
                            "--format",
                            "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t"
                            "{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}",
                            cid,
                        ],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r.returncode != 0 or not r.stdout.strip():
                        continue
                    parts = r.stdout.strip().split("\t")
                    if len(parts) < 6:
                        continue
                    name = parts[0]
                    cpu = parts[1].rstrip("%")
                    mem_perc = parts[3].rstrip("%")
                    mem_usage_str = parts[2]
                    stats[name] = {
                        "container": name,
                        "cpu_percent": float(cpu) if cpu else 0.0,
                        "mem_usage": mem_usage_str,
                        "mem_usage_mb": round(self._parse_mem_usage_mb(mem_usage_str), 2),
                        "mem_percent": float(mem_perc) if mem_perc else 0.0,
                        "net_io": parts[4],
                        "block_io": parts[5],
                    }
                except (ValueError, subprocess.TimeoutExpired):
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return stats

    # -- PostgreSQL storage -------------------------------------------------

    def _pg_query(self, sql: str) -> str:
        """Run a SQL query against the postgres container, return stdout."""
        compose_file = self._get_compose_path()
        if not compose_file:
            return ""

        try:
            result = subprocess.run(
                [
                    "docker", "compose", "-f", str(compose_file),
                    "exec", "-T", "postgres",
                    "psql", "-U", "postgres", "-d", "postgres",
                    "-c", sql,
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return ""

    def _get_pg_storage_mb(self) -> float:
        """PostgreSQL total database size in MB."""
        stdout = self._pg_query(
            "SELECT pg_database_size('postgres') / (1024.0 * 1024) AS size_mb;"
        )
        for line in stdout.split("\n"):
            try:
                return float(line.strip())
            except ValueError:
                pass
        return 0.0

    def _get_pgvector_stats(self) -> Dict[str, Any]:
        """Get memories table row count, table size, and index size."""
        stdout = self._pg_query(
            "SELECT "
            "  COALESCE((SELECT COUNT(*) FROM memories), 0) AS row_count, "
            "  COALESCE((SELECT pg_total_relation_size('memories') "
            "           / (1024.0 * 1024)), 0) AS table_mb, "
            "  COALESCE((SELECT pg_indexes_size('memories') "
            "           / (1024.0 * 1024)), 0) AS index_mb;"
        )
        for line in stdout.split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    return {
                        "row_count": int(parts[0]),
                        "table_size_mb": float(parts[1]),
                        "index_size_mb": float(parts[2]),
                    }
                except (ValueError, TypeError):
                    pass
        return {}

    # -- snapshot -----------------------------------------------------------

    def snapshot(self) -> SystemSnapshot:
        """Take a resource snapshot with mem0-specific metrics.

        Returns a ``SystemSnapshot`` whose ``extra`` dict carries:

        * ``containers`` — per-container CPU/memory/IO from docker stats
        * ``total_container_cpu_percent`` / ``total_container_mem_percent``
        * ``pg_storage_mb`` — PostgreSQL database size on disk
        * ``pgvector`` — memories row count, table size, index size
        * plus all parent ``DefaultTracker`` fields (CPU, memory, LLM tokens)
        """
        snapshot = super().snapshot()

        # Container metrics
        container_stats = self._get_container_stats()
        if container_stats:
            snapshot.extra["containers"] = container_stats

            total_mem_percent = sum(
                c.get("mem_percent", 0.0) for c in container_stats.values()
            )
            total_cpu = sum(
                c.get("cpu_percent", 0.0) for c in container_stats.values()
            )
            total_mem_mb = sum(
                c.get("mem_usage_mb", 0.0) for c in container_stats.values()
            )
            snapshot.extra["total_container_mem_percent"] = round(total_mem_percent, 2)
            snapshot.extra["total_container_cpu_percent"] = round(total_cpu, 2)
            snapshot.container_memory_rss_mb = round(total_mem_mb, 2)
            snapshot.container_cpu_percent = round(total_cpu, 2)

        # PostgreSQL storage
        pg_size_mb = self._get_pg_storage_mb()
        if pg_size_mb > 0:
            snapshot.extra["pg_storage_mb"] = round(pg_size_mb, 3)
            snapshot.storage_mb = round(pg_size_mb, 3)

        # pgvector table stats
        pgv_stats = self._get_pgvector_stats()
        if pgv_stats:
            snapshot.extra["pgvector"] = pgv_stats

        return snapshot

    # -- backend summary ----------------------------------------------------

    def backend_specific_stats(self) -> Dict[str, Any]:
        """Return mem0 backend summary for final report."""
        stats = super().backend_specific_stats()

        pg_size_mb = self._get_pg_storage_mb()
        if pg_size_mb > 0:
            stats["pg_storage_mb"] = round(pg_size_mb, 3)

        pgv_stats = self._get_pgvector_stats()
        if pgv_stats:
            stats["memories_row_count"] = pgv_stats.get("row_count", 0)

        return stats
