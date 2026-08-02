"""
Graphiti Local Builder — ensures Neo4j is accessible for direct-connect mode.

The graphiti_local adapter connects to Neo4j directly (bolt://) rather than
through the HTTP API server. This builder's role is:
  1. Verify Neo4j is reachable (bolt + HTTP health checks)
  2. Optionally start Neo4j via docker compose if configured
  3. Tear down docker services on cleanup (if it started them)
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder

logger = logging.getLogger(__name__)

# Default Neo4j ports
BOLT_PORT = 7687
HTTP_PORT = 7474


def _load_env(env_path: Path) -> dict:
    """Parse a .env file into a dict."""
    cfg: dict = {}
    if not env_path.exists():
        return cfg
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            cfg[key] = value
    return cfg


@register_builder("graphiti_local")
class GraphitiLocalBuilder(BaseBuilder):
    """Graphiti local builder that ensures Neo4j is running.

    graphiti_local connects directly to Neo4j via bolt:// — it does NOT use
    the Graphiti HTTP API server. This builder only manages the Neo4j dependency.

    Configuration:
        neo4j_uri:      Neo4j bolt URI  (default: bolt://localhost:7687)
        neo4j_http_uri: Neo4j HTTP URI  (default: http://localhost:7474)
        neo4j_user:     Neo4j username  (default: neo4j)
        neo4j_password: Neo4j password  (default: password)

        docker_compose: Path to docker-compose file that includes a neo4j
                        service (relative to project_root).
                        Default: docker-compose.yml (Graphiti's own)
        start_docker:   Whether to attempt docker compose up (default: True)
        docker_wait:    Max seconds to wait for neo4j to become healthy
                        (default: 120)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self.neo4j_uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_http_uri = config.get("neo4j_http_uri", "http://localhost:7474")
        self.neo4j_user = config.get("neo4j_user", "neo4j")
        self.neo4j_password = config.get("neo4j_password", "password")
        self.docker_compose = config.get("docker_compose", "docker-compose.yml")
        self.start_docker = config.get("start_docker", True)
        self.docker_wait = config.get("docker_wait", 120)
        self._started_by_us = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(self) -> bool:
        """Ensure Neo4j is healthy and reachable.

        Returns True if Neo4j is ready, False otherwise.
        """
        # 1. Already healthy?
        if await self._check_neo4j():
            logger.info("Neo4j is already running and healthy")
            return True

        # 2. Try docker compose if configured
        if self.start_docker:
            project_root = self._resolve_project_root()
            if project_root:
                compose_path = project_root / self.docker_compose
                if compose_path.exists():
                    logger.info("Neo4j not running — starting via docker compose: %s", compose_path)
                    if await self._docker_up(compose_path):
                        self._started_by_us = True
                        return True

        # 3. Give up with helpful message
        logger.error(
            "Neo4j is not reachable at %s / %s. "
            "Start Neo4j manually or set start_docker=True with a valid docker_compose path.",
            self.neo4j_uri, self.neo4j_http_uri,
        )
        return False

    async def cleanup(self) -> bool:
        """Stop docker services if we started them."""
        if not self._started_by_us:
            logger.info("GraphitiLocalBuilder: nothing to clean up (Neo4j was not started by us)")
            return True

        project_root = self._resolve_project_root()
        if not project_root:
            return True

        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.warning("Docker compose file not found: %s", compose_path)
            return True

        try:
            logger.info("Stopping docker services: %s", self.docker_compose)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "down"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("docker compose down failed: %s", result.stderr.strip())
                return False
            logger.info("Docker services stopped")
            self._started_by_us = False
            return True
        except Exception as exc:
            logger.error("Failed to stop docker services: %s", exc)
            return False

    def get_status(self):
        return {
            "name": self.__class__.__name__,
            "started_by_us": self._started_by_us,
            "neo4j_uri": self.neo4j_uri,
            "docker_compose": self.docker_compose,
        }

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def _check_neo4j(self) -> bool:
        """Check Neo4j HTTP interface (port 7474).

        The HTTP endpoint is the most reliable signal that Neo4j is fully up.
        """
        import urllib.request

        try:
            req = urllib.request.Request(self.neo4j_http_uri, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Docker management
    # ------------------------------------------------------------------

    async def _docker_up(self, compose_path: Path) -> bool:
        """Start docker compose services and wait for Neo4j to be healthy."""
        try:
            # Write .env so docker compose picks up API keys
            self._write_docker_env(compose_path)

            # Check if already running
            already = await self._docker_ps(compose_path)

            if already:
                logger.info("Docker services already running, waiting for health...")
                return await self._wait_healthy()

            # Start fresh
            logger.info("docker compose up -d ...")
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "up", "-d", "neo4j"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("docker compose up failed: %s", result.stderr.strip())
                return False

            logger.info("Docker services started, waiting for Neo4j to become healthy...")
            return await self._wait_healthy()

        except Exception as exc:
            logger.error("Failed to start docker services: %s", exc)
            return False

    async def _docker_ps(self, compose_path: Path) -> bool:
        """Return True if docker compose services are already running."""
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_path), "ps", "-q"],
            cwd=str(compose_path.parent),
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    async def _wait_healthy(self, timeout: int | None = None) -> bool:
        """Poll Neo4j HTTP endpoint until healthy or timeout."""
        if timeout is None:
            timeout = self.docker_wait

        start = time.time()
        interval = 5

        while time.time() - start < timeout:
            elapsed = int(time.time() - start)
            if await self._check_neo4j():
                logger.info("Neo4j is healthy (%ds)", elapsed)
                return True
            logger.info("Waiting for Neo4j... (%ds/%ds)", elapsed, timeout)
            await asyncio.sleep(interval)

        logger.error("Neo4j failed to become healthy within %ds", timeout)
        return False

    def _write_docker_env(self, compose_path: Path) -> None:
        """Write a .env file next to the compose file so docker picks up keys."""
        project_root = self._resolve_project_root()
        env_config: dict = {}
        if project_root:
            env_config = _load_env(project_root / ".env")

        api_key = env_config.get("LLM_API_KEY", env_config.get("OPENAI_API_KEY", ""))
        base_url = env_config.get("LLM_BASE_URL", env_config.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        env_path = compose_path.parent / ".env"
        content = (
            f"OPENAI_API_KEY={api_key}\n"
            f"OPENAI_BASE_URL={base_url}\n"
            f"NEO4J_USER={self.neo4j_user}\n"
            f"NEO4J_PASSWORD={self.neo4j_password}\n"
        )
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(content)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_project_root(self) -> Optional[Path]:
        """Resolve project_root from config or infer from this file's location."""
        if self.project_root:
            return Path(self.project_root)
        # src/builders/graphiti_local_builder.py -> src/ -> LifeBench_eval/
        candidate = Path(__file__).resolve().parent.parent.parent
        if candidate.exists():
            return candidate
        return None
