"""
Graphiti builder - starts Graphiti API server via docker compose.
"""

import asyncio
import logging
import os
import subprocess
import sys
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
                    os.environ.setdefault(key, value)

    llm_base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    if not llm_base.endswith("/v1"):
        llm_base = llm_base.rstrip("/") + "/v1"

    config["openai_api_key"] = os.environ.get("LLM_API_KEY", "")
    config["openai_base_url"] = llm_base
    config["chat_model"] = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

    vectorize_api_key = os.environ.get("VECTORIZE_API_KEY", "")
    vectorize_base_url = os.environ.get("VECTORIZE_BASE_URL", "")
    vectorize_model = os.environ.get("VECTORIZE_MODEL", "")
    vectorize_dimensions = os.environ.get("VECTORIZE_DIMENSIONS", "1024")

    if vectorize_api_key and vectorize_base_url and vectorize_model:
        config["embedding_api_key"] = vectorize_api_key
        base = vectorize_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        config["embedding_base_url"] = base
        config["embedding_model"] = vectorize_model
        config["embedding_dimension"] = vectorize_dimensions
    else:
        config["embedding_api_key"] = os.environ.get("EMBEDDING_API_KEY", config["openai_api_key"])
        config["embedding_base_url"] = "https://api.voyageai.com/v1"
        config["embedding_model"] = "voyage-3.5-lite"
        config["embedding_dimension"] = "1024"

    return config


@register_builder("graphiti")
class GraphitiBuilder(BaseBuilder):
    """Graphiti builder that starts/stops Graphiti API via docker compose.

    Configuration:
        docker_compose: Path to docker-compose.yaml (relative to project root)
        env_template: Path to .env.example template
        env_defaults: Dict of env var defaults to set
        docker_wait: Seconds to wait after starting services (default 60)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self.docker_compose = config.get("docker_compose")
        self.env_template = config.get("env_template")
        self.env_defaults = config.get("env_defaults", {})
        self.docker_wait = config.get("docker_wait", 60)
        self.post_start_script = config.get("post_start_script")
        self._started = False

    async def build(self) -> bool:
        """
        Start Graphiti API server via docker compose.

        Returns:
            True if successful or already running, False on failure
        """
        project_root = Path(self.project_root) if self.project_root else self._get_project_root()
        if not project_root:
            logger.error("Cannot determine project root")
            return False

        # Check if containers are actually healthy before returning "already started"
        if self._started and self.docker_compose:
            compose_path = project_root / self.docker_compose
            if compose_path.exists():
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "ps", "-q"],
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    # Containers are running, check API health
                    try:
                        import urllib.request
                        req = urllib.request.Request("http://localhost:8000/healthcheck")
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            if resp.status == 200:
                                logger.info("Graphiti builder already started and healthy")
                                return True
                    except Exception:
                        logger.warning("Graphiti builder _started=True but API not healthy, restarting...")
                        self._started = False

        if self._started:
            logger.info("Graphiti builder already started")
            return True

        # 1. Prepare .env file
        if not await self._prepare_env_file(project_root):
            logger.error("Failed to prepare .env file")
            return False

        # 2. Start docker services
        if self.docker_compose:
            if not await self._start_docker(project_root):
                return False

        # 3. Run post-start script
        if self.post_start_script:
            if not await self._run_post_start(project_root):
                logger.warning("Post-start script failed, continuing anyway")

        self._started = True
        logger.info("Graphiti builder completed successfully")
        return True

    async def cleanup(self) -> bool:
        """
        Stop Graphiti API server via docker compose.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("Graphiti builder not started, nothing to cleanup")
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
        """Get project root from project_root config or cli.py location."""
        if self.project_root:
            return Path(self.project_root)
        # Try to infer from cli.py location
        cli_path = Path(__file__).parent.parent.parent / "cli.py"
        if cli_path.exists():
            return cli_path.parent.resolve()
        return None

    async def _prepare_env_file(self, project_root: Path) -> bool:
        """Prepare .env file from LifeBench_eval/.env configuration."""
        compose_path = project_root / self.docker_compose
        env_path = compose_path.parent / ".env"

        # Load config from LifeBench_eval/.env
        project_env = project_root / ".env"
        env_config = _load_env_config(project_env)

        env_content = f"""# Graphiti Environment Configuration
# Generated by GraphitiBuilder from LifeBench_eval/.env

# LLM settings (DeepSeek)
OPENAI_API_KEY={env_config['openai_api_key']}
OPENAI_BASE_URL={env_config['openai_base_url']}

# Neo4j settings
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Embedding settings (SiliconFlow Qwen3-Embedding-4B)
EMBEDDING_API_KEY={env_config.get('embedding_api_key', '')}
EMBEDDING_BASE_URL={env_config.get('embedding_base_url', '')}
EMBEDDING_MODEL={env_config.get('embedding_model', '')}
EMBEDDING_DIMENSION={env_config.get('embedding_dimension', '1024')}
"""
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info(f"Created .env file: {env_path}")
            logger.info(f"  LLM: {env_config['chat_model']} @ {env_config['openai_base_url']}")
            logger.info(f"  Embedding: {env_config.get('embedding_model')} @ {env_config.get('embedding_base_url')}")
            return True
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False

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
                capture_output=True,
                text=True,
            )
            already_running = bool(result.stdout.strip())

            if already_running:
                # Containers already running and healthy, don't restart
                logger.info("Services already running, skipping restart...")
                return True

            # Start services fresh
            logger.info(f"Starting docker services: {self.docker_compose}")
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
        import time
        import urllib.request
        import urllib.error

        start = time.time()
        check_interval = 5

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

            # Check 2: graphiti can actually talk to neo4j via /clear endpoint
            try:
                req = urllib.request.Request(
                    "http://localhost:8000/clear",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info("Graphiti API and neo4j are fully healthy")
                        return True
            except urllib.error.HTTPError as e:
                if e.code == 500:
                    logger.debug(f"neo4j not ready yet: {e}")
                    await asyncio.sleep(check_interval)
                    continue
            except Exception:
                await asyncio.sleep(check_interval)
                continue

            logger.info(f"Waiting for graphiti to connect to neo4j... ({elapsed}s/{timeout}s)")
            await asyncio.sleep(check_interval)

        return False

    async def _run_post_start(self, project_root: Path) -> bool:
        """Run post-start script."""
        script_path = project_root / self.post_start_script
        if not script_path.exists():
            logger.warning(f"Post-start script not found: {script_path}")
            return False

        try:
            logger.info(f"Running post-start script: {self.post_start_script}")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(f"Post-start script failed: {result.stderr}")
                return False
            logger.info("Post-start script completed")
            return True
        except Exception as e:
            logger.warning(f"Failed to run post-start script: {e}")
            return False

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "docker_compose": self.docker_compose,
        }
