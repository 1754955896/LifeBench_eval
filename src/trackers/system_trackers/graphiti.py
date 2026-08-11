"""
Graphiti-specific resource tracker.

Extends DefaultTracker with Neo4j knowledge-graph metrics:
- Node counts (Entity, Episode, Community) via Cypher
- Edge count (EntityEdge) via Cypher
- Neo4j database store size on disk (via docker exec du or HTTP API)
"""
import json
import logging
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
        self._config = config or {}
        self._neo4j_http = "http://localhost:7474"
        self._neo4j_user = config.get("neo4j_user", "neo4j") if config else "neo4j"
        self._neo4j_password = config.get("neo4j_password", "password") if config else "password"
        self._compose_file = config.get("docker_compose") if config else None
        self._compose_dir: Optional[Path] = None

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
        url = f"{self._neo4j_http}/db/neo4j/tx/commit"
        credentials = f"{self._neo4j_user}:{self._neo4j_password}"
        auth = urllib.request.HTTPBasicAuthHandler()
        auth.add_password(
            realm="Neo4j", uri=self._neo4j_http,
            user=self._neo4j_user, passwd=self._neo4j_password
        )

        try:
            payload = json.dumps({"statements": [{"statement": statement}]}).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            opener = urllib.request.build_opener(auth)
            with opener.open(req, timeout=10) as resp:
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
            "MATCH (n) RETURN DISTINCT labels(n) AS label, count(n) AS cnt"
        )
        if not rows:
            return {}
        return {
            ", ".join(r.get("label", [])): r.get("cnt", 0)
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
                        "du", "-sm", "/data/databases/neo4j",
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
                    mem_perc = parts[3].rstrip("%")
                    stats[name] = {
                        "container": name,
                        "cpu_percent": float(cpu) if cpu else 0.0,
                        "mem_usage": parts[2],
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
            total_mem = sum(
                c.get("mem_percent", 0.0) for c in container_stats.values()
            )
            snapshot.extra["total_container_cpu_percent"] = round(total_cpu, 2)
            snapshot.extra["total_container_mem_percent"] = round(total_mem, 2)

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
