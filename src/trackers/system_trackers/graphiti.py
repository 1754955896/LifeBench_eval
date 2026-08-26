"""
Graphiti-specific resource tracker.

Extends DefaultTracker with Neo4j knowledge-graph metrics:
- Node counts (Entity, Episode, Community) via Cypher
- Edge count (EntityEdge) via Cypher
- Neo4j database store size on disk (via docker exec du or HTTP API)
"""
import base64
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import urllib.request
import urllib.error

from src.trackers.system_trackers.base import SystemSnapshot, register_tracker
from src.trackers.system_trackers.default import DefaultTracker

logger = logging.getLogger(__name__)


@register_tracker("graphiti_local")
class GraphitiTracker(DefaultTracker):
    """Graphiti-specific resource tracker.

    Extends DefaultTracker with:
    - Neo4j node/edge counts via HTTP Cypher API
    - Neo4j store size on disk
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        pid: Optional[int] = None,
        llm_proxy_url: Optional[str] = None,
    ):
        super().__init__(config, pid, llm_proxy_url)
        cfg = config or {}
        self._config = cfg
        self._neo4j_user = cfg.get("neo4j_user", "neo4j")
        self._neo4j_password = cfg.get("neo4j_password", "password")
        self._neo4j_database = cfg.get("neo4j_database", "neo4j")
        self._neo4j_http = self._resolve_http_uri(cfg)
        self._compose_file = cfg.get("docker_compose")
        self._compose_dir: Optional[Path] = None

    @staticmethod
    def _resolve_http_uri(cfg: dict) -> str:
        """Resolve the Neo4j HTTP base URL.

        Preference order: explicit ``neo4j_http_uri`` → derived from
        ``neo4j_uri`` (bolt://host:7687 → http://host:7474) → localhost default.
        """
        http_uri = cfg.get("neo4j_http_uri")
        if http_uri:
            return str(http_uri).rstrip("/")
        bolt = cfg.get("neo4j_uri", "bolt://localhost:7687")
        if bolt.startswith("bolt://"):
            host = bolt[len("bolt://"):].rsplit(":", 1)[0]
            return f"http://{host}:7474"
        return "http://localhost:7474"

    @property
    def system_name(self) -> str:
        return "graphiti_local"

    # -- compose helpers ---------------------------------------------------

    def _get_compose_dir(self) -> Optional[Path]:
        if self._compose_dir is not None:
            return self._compose_dir
        if not self._compose_file:
            return None
        base = Path(__file__).resolve().parents[3]
        candidate = base / self._compose_file
        if candidate.exists():
            self._compose_dir = candidate.parent
            return self._compose_dir
        return None

    def _get_compose_path(self) -> Optional[Path]:
        compose_dir = self._get_compose_dir()
        if not compose_dir:
            return None
        path = compose_dir / "docker-compose.yml"
        return path if path.exists() else None

    # -- Neo4j Cypher via HTTP API -----------------------------------------

    def _cypher_query(self, statement: str) -> Optional[list]:
        """Run a Cypher query against Neo4j HTTP API, return rows as list of dicts."""
        url = f"{self._neo4j_http}/db/{self._neo4j_database}/tx/commit"
        token = base64.b64encode(
            f"{self._neo4j_user}:{self._neo4j_password}".encode()
        ).decode()

        try:
            payload = json.dumps({"statements": [{"statement": statement}]}).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Basic {token}",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            if data.get("errors"):
                logger.debug("Cypher query error: %s", data["errors"])
                return None
            results = data.get("results", [])
            if not results:
                return None
            columns = results[0].get("columns", [])
            rows = []
            for row_data in results[0].get("data", []):
                row = dict(zip(columns, row_data.get("row", [])))
                rows.append(row)
            return rows
        except Exception:
            return None

    def _get_neo4j_node_counts(self) -> Dict[str, int]:
        """Get node counts per label from Neo4j."""
        rows = self._cypher_query(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS cnt"
        )
        if not rows:
            return {}
        return {
            r.get("label"): r.get("cnt", 0)
            for r in rows
            if r.get("label")
        }

    def _get_neo4j_edge_count(self) -> int:
        """Get total relationship count from Neo4j."""
        rows = self._cypher_query("MATCH ()-[r]->() RETURN count(r) AS cnt")
        if not rows:
            return 0
        return rows[0].get("cnt", 0)

    def _get_neo4j_store_size_mb(self) -> float:
        """Get Neo4j store size in MB.

        Tries docker exec du on /data first (when Neo4j is in Docker),
        falls back to querying dbms information.
        """
        compose_path = self._get_compose_path()
        if compose_path:
            try:
                result = subprocess.run(
                    [
                        "docker", "compose", "-f", str(compose_path),
                        "exec", "-T", "neo4j",
                        "du", "-sm", f"/data/databases/{self._neo4j_database}",
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    size_mb = float(result.stdout.strip().split()[0])
                    return size_mb
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
                pass

        return 0.0

    # -- container stats ---------------------------------------------------

    @staticmethod
    def _parse_mem_mb(mem_usage: str) -> float:
        """Parse a docker-stats MemUsage string (e.g. '3.022GiB / 7.69GiB') into MB."""
        if not mem_usage:
            return 0.0
        token = mem_usage.split("/")[0].strip()
        m = re.match(r"([\d.]+)\s*([KMGT]?i?B)", token, re.IGNORECASE)
        if not m:
            return 0.0
        value = float(m.group(1))
        unit = m.group(2).lower()
        factors = {
            "b": 1 / (1024 * 1024),
            "kb": 1 / 1024, "kib": 1 / 1024,
            "mb": 1.0, "mib": 1.0,
            "gb": 1024.0, "gib": 1024.0,
            "tb": 1024 * 1024, "tib": 1024 * 1024,
        }
        return value * factors.get(unit, 0.0)

    def _get_container_stats(self) -> Dict[str, Any]:
        """Get CPU/memory for the Neo4j container."""
        compose_path = self._get_compose_path()
        if not compose_path:
            return {}

        stats: Dict[str, Any] = {}
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "-q", "neo4j"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return {}
            container_ids = [c.strip() for c in result.stdout.strip().split("\n") if c.strip()]
            if not container_ids:
                return {}

            for cid in container_ids:
                try:
                    r = subprocess.run(
                        ["docker", "stats", "--no-stream",
                         "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}",
                         cid],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r.returncode != 0 or not r.stdout.strip():
                        continue
                    parts = r.stdout.strip().split("\t")
                    if len(parts) < 4:
                        continue
                    name = parts[0]
                    cpu = parts[1].rstrip("%")
                    mem_usage = parts[2]
                    mem_perc = parts[3].rstrip("%")
                    stats[name] = {
                        "container": name,
                        "cpu_percent": float(cpu) if cpu else 0.0,
                        "mem_usage": mem_usage,
                        "mem_usage_mb": self._parse_mem_mb(mem_usage),
                        "mem_percent": float(mem_perc) if mem_perc else 0.0,
                    }
                except (ValueError, subprocess.TimeoutExpired):
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return stats

    # -- snapshot -----------------------------------------------------------

    def snapshot(self) -> SystemSnapshot:
        snapshot = super().snapshot()

        # Node and edge counts — always present, even when empty
        node_counts = self._get_neo4j_node_counts()
        edge_count = self._get_neo4j_edge_count()
        total_nodes = sum(node_counts.values())
        snapshot.extra["neo4j_nodes"] = node_counts
        snapshot.extra["neo4j_total_nodes"] = total_nodes
        snapshot.extra["neo4j_edges"] = edge_count

        # Store size — measure actual database dir, not whole Docker volume
        store_mb = self._get_neo4j_store_size_mb()
        snapshot.extra["neo4j_store_mb"] = round(store_mb, 3)
        if store_mb > 0:
            snapshot.storage_mb = round(store_mb, 3)

        # Container metrics
        container_stats = self._get_container_stats()
        if container_stats:
            snapshot.extra["containers"] = container_stats
            total_cpu = sum(
                c.get("cpu_percent", 0.0) for c in container_stats.values()
            )
            total_mem_mb = sum(
                c.get("mem_usage_mb", 0.0) for c in container_stats.values()
            )
            total_mem_percent = sum(
                c.get("mem_percent", 0.0) for c in container_stats.values()
            )
            snapshot.extra["total_container_cpu_percent"] = round(total_cpu, 2)
            snapshot.extra["total_container_mem_percent"] = round(total_mem_percent, 2)
            # Fill the standard SystemSnapshot container fields so the
            # GlobalMonitor timeline carries container CPU/memory (previously
            # these stayed at their 0.0 defaults).
            snapshot.container_cpu_percent = round(total_cpu, 2)
            snapshot.container_memory_rss_mb = round(total_mem_mb, 2)

        return snapshot

    # -- backend summary ----------------------------------------------------

    def backend_specific_stats(self) -> Dict[str, Any]:
        stats = super().backend_specific_stats()

        node_counts = self._get_neo4j_node_counts()
        if node_counts:
            stats["neo4j_nodes"] = node_counts
            stats["neo4j_total_nodes"] = sum(node_counts.values())

        edge_count = self._get_neo4j_edge_count()
        if edge_count >= 0:
            stats["neo4j_edges"] = edge_count

        store_mb = self._get_neo4j_store_size_mb()
        if store_mb > 0:
            stats["neo4j_store_mb"] = round(store_mb, 3)

        return stats
