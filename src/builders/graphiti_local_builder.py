"""
Graphiti Local Builder - starts Graphiti API server via docker compose (local standalone mode).

This builder uses docker-compose.local.yml which combines Neo4j and Graphiti API
in a single standalone deployment, simpler than the full distributed setup.
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


def _load_env_config(project_env: Path) -> dict:
    """Load configuration from project .env file.

    Args:
        project_env: Path to LifeBench_eval/.env

    Returns:
        Dict with api_key, base_url, model, vectorize_api_key, embedding settings
    """
    config = {}
    if project_env.exists():
        with open(project_env, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key] = value

    llm_base = config.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    if not llm_base.endswith("/v1"):
        llm_base = llm_base.rstrip("/") + "/v1"

    config["openai_api_key"] = config.get("LLM_API_KEY", "")
    config["openai_base_url"] = llm_base
    config["chat_model"] = config.get("LLM_MODEL", "deepseek-v4-flash")

    return config


@register_builder("graphiti_local")
class GraphitiLocalBuilder(BaseBuilder):
    """Graphiti local builder that starts/stops local Graphiti API via docker compose.

    Uses docker-compose.local.yml which combines Neo4j + Graphiti API in one file.
    Simpler than the full distributed setup.

    Configuration:
        docker_compose: Path to docker-compose.local.yml (relative to project root)
        docker_wait: Seconds to wait after starting services (default 120)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self.docker_compose = config.get("docker_compose", "local/docker-compose.local.yml")
        self.docker_wait = config.get("docker_wait", 120)
        self._started = False

    async def build(self) -> bool:
        """
        Start Neo4j via docker if not running, then verify health.

        graphiti_local connects directly to Neo4j, not via HTTP API.

        Returns:
            True if Neo4j is running, False on failure
        """
        # Check if Neo4j is already healthy
        if await self._check_health():
            logger.info("Neo4j is already running at localhost:7474")
            self._started = True
            return True

        # Try to start docker services
        project_root = self._get_project_root()
        if project_root:
            logger.info("Neo4j not running, attempting to start docker services...")
            if await self._start_docker(project_root):
                self._started = True
                return True

        logger.error("Neo4j is not running at localhost:7474")
        logger.info("Please start Neo4j first")
        return False

    async def cleanup(self) -> bool:
        """
        Stop Graphiti API server via docker compose.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("Graphiti local builder not started, nothing to cleanup")
            return True

        project_root = Path(self.project_root) if self.project_root else self._get_project_root()
        if not project_root or not self.docker_compose:
            return True

        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.warning(f"Docker compose file not found: {compose_path}")
            return True

        try:
            logger.info(f"Stopping docker services: {self.docker_compose}")
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "down"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Docker stop failed: {result.stderr}")
                return False
            logger.info("Docker services stopped")
            self._started = False
            return True
        except Exception as e:
            logger.error(f"Failed to stop docker services: {e}")
            return False

    def _get_project_root(self) -> Optional[Path]:
        """Get project root from project_root config or this file's location."""
        if self.project_root:
            return Path(self.project_root)
        # Try to infer from this file's location
        builder_path = Path(__file__).parent
        # src/builders/graphiti_local_builder.py -> src/ -> LifeBench_eval/
        project_root = builder_path.parent.parent
        if project_root.exists():
            return project_root.resolve()
        return None

    async def _start_docker(self, project_root: Path) -> bool:
        """Start docker compose services."""
        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.error(f"Docker compose file not found: {compose_path}")
            return False

        try:
            # Check if containers are already running
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "-q"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            already_running = bool(result.stdout.strip())

            if already_running:
                logger.info("Services already running, checking health...")
                if await self._check_health():
                    logger.info("Services already healthy")
                    return True
                else:
                    logger.warning("Services running but not healthy, restarting...")

            # Start services fresh
            logger.info(f"Starting docker services: {self.docker_compose}")

            # Create .env file from LifeBench_eval/.env
            env_config = _load_env_config(project_root / ".env")
            env_path = compose_path.parent / ".env"
            env_content = f"""OPENAI_API_KEY={env_config.get('openai_api_key', '')}
OPENAI_BASE_URL={env_config.get('openai_base_url', 'https://api.deepseek.com/v1')}
"""
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)

            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "up", "-d"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Docker start failed: {result.stderr}")
                return False
            logger.info("Docker services started")

            # Wait for services to be healthy
            if not await self._wait_for_healthy(compose_path):
                logger.error("Services failed to become healthy")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to start docker services: {e}")
            return False

    async def _wait_for_healthy(self, compose_path: Path, timeout: int = 180) -> bool:
        """Wait for graphiti API and neo4j to be fully healthy."""
        import urllib.request

        start = time.time()
        check_interval = 10

        while time.time() - start < timeout:
            elapsed = int(time.time() - start)

            # Check 1: graphiti API is responding
            try:
                req = urllib.request.Request(
                    "http://localhost:8000/healthcheck",
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info(f"Graphiti API responding ({elapsed}s)")
            except Exception:
                logger.info(f"Waiting for graphiti API... ({elapsed}s/{timeout}s)")
                await asyncio.sleep(check_interval)
                continue

            # Check 2: neo4j is responding
            try:
                req = urllib.request.Request(
                    "http://localhost:7474",
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logger.info("Neo4j is responding")
                    logger.info("Graphiti API and neo4j are fully healthy")
                    return True
            except Exception:
                logger.info(f"Waiting for neo4j... ({elapsed}s/{timeout}s)")
                await asyncio.sleep(check_interval)
                continue

        return False

    async def _check_health(self) -> bool:
        """Check if Neo4j is healthy at localhost:7687."""
        import urllib.request

        try:
            # Check Neo4j HTTP interface
            req = urllib.request.Request("http://localhost:7474")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "docker_compose": self.docker_compose,
        }
