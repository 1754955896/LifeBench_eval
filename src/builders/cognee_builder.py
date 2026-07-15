"""
Cognee builder - starts Cognee Docker container via docker compose.
"""

import logging
import os
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
        Dict with LLM and embedding configuration
    """
    config = {}
    if project_env.exists():
        with open(project_env, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)

    config["llm_api_key"] = os.environ.get("LLM_API_KEY", "")
    config["llm_base_url"] = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    config["llm_model"] = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
    config["llm_provider"] = os.environ.get("LLM_PROVIDER", "openai")

    # Embedding config
    config["embedding_api_key"] = os.environ.get("VECTORIZE_API_KEY", os.environ.get("LLM_API_KEY", ""))
    config["embedding_base_url"] = os.environ.get("VECTORIZE_BASE_URL", "https://api.siliconflow.cn/v1")
    config["embedding_model"] = os.environ.get("VECTORIZE_MODEL", "Qwen/Qwen3-Embedding-4B")
    config["embedding_dimensions"] = os.environ.get("VECTORIZE_DIMENSIONS", "1024")

    # Rerank config (optional)
    config["rerank_api_key"] = os.environ.get("RERANK_API_KEY", os.environ.get("VECTORIZE_API_KEY", ""))
    config["rerank_base_url"] = os.environ.get("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
    config["rerank_model"] = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    return config


@register_builder("cognee")
class CogneeBuilder(BaseBuilder):
    """Cognee builder that starts/stops Cognee Docker via docker compose.

    Configuration:
        docker_compose: Path to docker-compose.yaml (relative to project root)
        docker_wait: Seconds to wait after starting services (default 60)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self.docker_compose = config.get("docker_compose")
        self.docker_wait = config.get("docker_wait", 60)
        self._started = False

    async def build(self) -> bool:
        """
        Start Cognee Docker services via docker compose.

        Returns:
            True if successful or already running, False on failure
        """
        if self._started:
            logger.info("Cognee builder already started")
            return True

        project_root = Path(self.project_root) if self.project_root else self._get_project_root()
        if not project_root:
            logger.error("Cannot determine project root")
            return False

        # 1. Prepare .env file
        if not await self._prepare_env_file(project_root):
            logger.error("Failed to prepare .env file")
            return False

        # 2. Start docker services
        if self.docker_compose:
            if not await self._start_docker(project_root):
                return False

            # Wait for services to be ready
            logger.info(f"Waiting {self.docker_wait}s for services to be ready...")
            time.sleep(self.docker_wait)

            # 3. Wait for API to be healthy
            if not await self._wait_for_api():
                return False

        self._started = True
        logger.info("Cognee builder completed successfully")
        return True

    async def cleanup(self) -> bool:
        """
        Stop Cognee Docker services via docker compose.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("Cognee builder not started, nothing to cleanup")
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
                ["docker", "compose", "-f", str(compose_path), "down", "--remove-orphans"],
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
        cli_path = Path(__file__).parent.parent.parent / "cli.py"
        if cli_path.exists():
            return cli_path.parent.resolve()
        return None

    async def _prepare_env_file(self, project_root: Path) -> bool:
        """Prepare .env file for Cognee Docker."""
        compose_path = project_root / self.docker_compose
        env_path = compose_path.parent / ".env"

        # Load config from LifeBench_eval/.env
        project_env = project_root / ".env"
        env_config = _load_env_config(project_env)

        # Build LLM_ARGS to disable DeepSeek thinking mode
        llm_args_json = '{"extra_body": {"thinking": {"type": "disabled"}}}'

        env_content = f"""# Cognee Server Environment Configuration
# Generated by CogneeBuilder from LifeBench_eval/.env

# =================== LLM Config ===================
LLM_API_KEY={env_config['llm_api_key']}
# Use deepseek-v4-flash with thinking mode disabled via extra_body
LLM_MODEL=openai/{env_config['llm_model']}
# Use openai provider with custom endpoint for DeepSeek (OpenAI-compatible API)
LLM_PROVIDER={env_config['llm_provider']}
# LLM_ENDPOINT maps to llm_endpoint in config
LLM_ENDPOINT={env_config['llm_base_url']}
# Disable DeepSeek thinking mode to avoid tool_choice incompatibility
LLM_ARGS={llm_args_json}

# =================== Embedding Config ===================
# Use openai_compatible provider for SiliconFlow (uses openai SDK directly)
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL={env_config['embedding_model']}
EMBEDDING_DIMENSIONS={env_config['embedding_dimensions']}
EMBEDDING_API_KEY={env_config['embedding_api_key']}
EMBEDDING_ENDPOINT={env_config['embedding_base_url']}

# =================== Database Config (PostgreSQL) ===================
DB_PROVIDER=postgres
DB_HOST=postgres
DB_PORT=5432
DB_USERNAME=cognee
DB_PASSWORD=cognee
DB_NAME=cognee_db

# =================== Vector DB (pgvector) ===================
VECTOR_DB_PROVIDER=pgvector

# =================== Graph DB ===================
GRAPH_DATABASE_PROVIDER=postgres

# =================== Storage ===================
SYSTEM_ROOT_DIRECTORY=/app/data/.cognee_system
DATA_ROOT_DIRECTORY=/app/data/.cognee_data

# =================== API Config ===================
ENV=local
LOG_LEVEL=INFO
CORS_ALLOWED_ORIGINS=*

# Disable multi-user auth for local testing
ENABLE_BACKEND_ACCESS_CONTROL=false

# Skip LLM connection test (for testing with external APIs)
COGNEE_SKIP_CONNECTION_TEST=true
"""
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info(f"Created .env file: {env_path}")
            logger.info(f"  LLM: {env_config['llm_model']} @ {env_config['llm_base_url']}")
            logger.info(f"  Embedding: {env_config['embedding_model']} ({env_config['embedding_dimensions']} dims) @ {env_config['embedding_base_url']}")
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
            if result.stdout.strip():
                logger.info("Docker services already running")
                return True

            # Start postgres first, then cognee
            logger.info(f"Starting postgres: {self.docker_compose}")
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "up", "-d", "postgres"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Docker start postgres failed: {result.stderr}")
                return False

            # Wait for postgres to be ready
            if not await self._wait_for_postgres(project_root):
                logger.error("Postgres did not become ready in time")
                return False

            # Then start cognee
            logger.info(f"Starting cognee: {self.docker_compose}")
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "up", "-d", "cognee"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Docker start cognee failed: {result.stderr}")
                return False

            logger.info("Docker services started")
            return True

        except Exception as e:
            logger.error(f"Failed to start docker services: {e}")
            return False

    async def _wait_for_postgres(self, project_root: Path) -> bool:
        """Wait for postgres container to be healthy."""
        import urllib.request
        import urllib.error

        max_retries = 30  # 30 * 2s = 60s
        for attempt in range(max_retries):
            try:
                # Check postgres health via docker inspect
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Health.Status}}", "postgres"],
                    capture_output=True,
                    text=True,
                )
                health_status = result.stdout.strip()
                if health_status == "healthy":
                    logger.info("Postgres is healthy")
                    return True
                logger.debug(f"Postgres status: {health_status}, attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                logger.debug(f"Postgres check attempt {attempt + 1}/{max_retries}: {e}")

            if attempt < max_retries - 1:
                time.sleep(2)

        logger.error("Postgres did not become healthy in 60s")
        return False

    async def _wait_for_api(self) -> bool:
        """Wait for Cognee API and container to be healthy."""
        import urllib.request
        import urllib.error

        max_retries = 96  # 96 * 5s = 480s
        for attempt in range(max_retries):
            try:
                # Check container health first
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Health.Status}}", "cognee"],
                    capture_output=True,
                    text=True,
                )
                container_health = result.stdout.strip()

                # Check API health
                url = "http://localhost:8000/health"
                req = urllib.request.Request(url, method="GET")
                response = urllib.request.urlopen(req, timeout=5)
                if response.status == 200 and container_health == "healthy":
                    result_text = response.read().decode()
                    logger.info(f"Cognee API is ready: {result_text}")
                    return True
            except urllib.error.HTTPError as e:
                if e.code == 200:
                    logger.info(f"Cognee API is ready (attempt {attempt + 1})")
                    return True
                logger.debug(f"API responded with status {e.code}, retrying...")
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}: waiting... container_health={container_health if 'container_health' in dir() else 'unknown'}")

            if attempt < max_retries - 1:
                time.sleep(5)

        logger.error("Cognee API not ready after 480s")
        return False

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "docker_compose": self.docker_compose,
        }
