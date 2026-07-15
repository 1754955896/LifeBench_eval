"""
memU-server Builder - starts/stops memU-server via docker compose.
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
    """Load configuration from project .env file."""
    config = {}
    if project_env.exists():
        with open(project_env, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)

    # LLM config
    llm_base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
    if not llm_base.endswith("/v1"):
        llm_base = llm_base.rstrip("/") + "/v1"
    config["openai_api_key"] = os.environ.get("LLM_API_KEY", "")
    config["openai_base_url"] = llm_base
    config["chat_model"] = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

    # Embedding config
    vectorize_api_key = os.environ.get("VECTORIZE_API_KEY", "")
    vectorize_base_url = os.environ.get("VECTORIZE_BASE_URL", "")
    vectorize_model = os.environ.get("VECTORIZE_MODEL", "")
    if vectorize_api_key and vectorize_base_url and vectorize_model:
        base = vectorize_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        config["embedding_api_key"] = vectorize_api_key
        config["embedding_base_url"] = base
        config["embedding_model"] = vectorize_model
    else:
        config["embedding_api_key"] = os.environ.get("EMBEDDING_API_KEY", config["openai_api_key"])
        config["embedding_base_url"] = "https://api.voyageai.com/v1"
        config["embedding_model"] = "voyage-3.5-lite"

    return config


@register_builder("memu_server")
class MemUServerBuilder(BaseBuilder):
    """memU-server builder that starts/stops Docker services.

    Configuration:
        docker_compose: Path to docker-compose.yaml (relative to project root)
        docker_wait: Seconds to wait after starting services (default: 15)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self.docker_compose = config.get("docker_compose", "systems/memU-server/docker-compose.yml")
        self.docker_wait = config.get("docker_wait", 15)
        self._started = False

    async def build(self) -> bool:
        """
        Start memU-server via docker compose.

        Returns:
            True if successful or already running, False on failure
        """
        if self._started:
            logger.info("memU-server already started")
            return True

        project_root = self._get_project_root()
        if not project_root:
            logger.error("Cannot determine project root")
            return False

        # 1. Prepare .env file
        if not await self._prepare_env_file(project_root):
            logger.error("Failed to prepare .env file")
            return False

        # 2. Start docker services
        if not await self._start_docker(project_root):
            return False

        # 3. Wait for services to be ready
        logger.info(f"Waiting {self.docker_wait}s for services to be ready...")
        time.sleep(self.docker_wait)

        self._started = True
        logger.info("memU-server builder completed successfully")
        return True

    async def cleanup(self) -> bool:
        """
        Stop memU-server via docker compose.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("memU-server not started, nothing to cleanup")
            return True

        project_root = self._get_project_root()
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
        """Get project root from config or infer from location."""
        if self.project_root:
            return Path(self.project_root)
        cli_path = Path(__file__).parent.parent.parent / "cli.py"
        if cli_path.exists():
            return cli_path.parent.resolve()
        return None

    async def _prepare_env_file(self, project_root: Path) -> bool:
        """Prepare .env file from LifeBench_eval/.env configuration."""
        compose_path = project_root / self.docker_compose
        env_path = compose_path.parent / ".env"

        project_env = project_root / ".env"
        env_config = _load_env_config(project_env)

        env_content = f"""OPENAI_API_KEY={env_config['openai_api_key']}
OPENAI_BASE_URL={env_config['openai_base_url']}
DEFAULT_LLM_MODEL={env_config['chat_model']}
EMBEDDING_API_KEY={env_config['embedding_api_key']}
EMBEDDING_BASE_URL={env_config['embedding_base_url']}
EMBEDDING_MODEL={env_config['embedding_model']}
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=memu
POSTGRES_HOST=postgres
TEMPORAL_HOST=temporal
TEMPORAL_PORT=7233
TEMPORAL_DB=temporal
"""
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info(f"Created .env file: {env_path}")
            logger.info(f"  Chat Model: {env_config['chat_model']} @ {env_config['openai_base_url']}")
            logger.info(f"  Embedding: {env_config['embedding_model']} @ {env_config['embedding_base_url']}")
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

            # Start services
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
            return True

        except Exception as e:
            logger.error(f"Failed to start docker services: {e}")
            return False

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "docker_compose": self.docker_compose,
        }
