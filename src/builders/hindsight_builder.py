"""
Hindsight builder - starts Hindsight server via docker compose.
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
        Dict with LLM, embedding, rerank settings
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
    config["vectorize_api_key"] = os.environ.get("VECTORIZE_API_KEY", "")
    config["vectorize_base_url"] = os.environ.get("VECTORIZE_BASE_URL", "https://api.siliconflow.cn/v1")
    config["vectorize_model"] = os.environ.get("VECTORIZE_MODEL", "Qwen/Qwen3-Embedding-4B")
    config["vectorize_dimensions"] = os.environ.get("VECTORIZE_DIMENSIONS", "1024")
    config["rerank_api_key"] = os.environ.get("RERANK_API_KEY", os.environ.get("VECTORIZE_API_KEY", ""))
    config["rerank_base_url"] = os.environ.get("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
    config["rerank_model"] = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    return config


@register_builder("hindsight")
class HindsightBuilder(BaseBuilder):
    """Hindsight builder that starts/stops Hindsight via docker compose.

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
        """Start Hindsight server via docker compose."""
        if self._started:
            logger.info("Hindsight builder already started")
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

        self._started = True
        logger.info("Hindsight builder completed successfully")
        return True

    async def cleanup(self) -> bool:
        """Stop Hindsight server via docker compose."""
        if not self._started:
            logger.info("Hindsight builder not started, nothing to cleanup")
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
        """Get project root from project_root config or script location."""
        if self.project_root:
            return Path(self.project_root)
        cli_path = Path(__file__).parent.parent.parent / "cli.py"
        if cli_path.exists():
            return cli_path.parent.resolve()
        return None

    async def _prepare_env_file(self, project_root: Path) -> bool:
        """Prepare .env file for Hindsight docker-compose."""
        compose_path = project_root / self.docker_compose
        env_path = compose_path.parent / ".env"

        # Load config from LifeBench_eval/.env
        project_env = project_root / ".env"
        env_config = _load_env_config(project_env)

        env_content = f"""# Hindsight Server Environment Configuration
# Generated by HindsightBuilder from LifeBench_eval/.env

# Database settings
HINDSIGHT_DB_USER=postgres
HINDSIGHT_DB_PASSWORD=hindsight_dev
HINDSIGHT_DB_NAME=hindsight_db

# LLM config (OpenAI-compatible)
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_API_KEY={env_config['llm_api_key']}
HINDSIGHT_API_LLM_BASE_URL={env_config['llm_base_url']}
HINDSIGHT_API_LLM_MODEL={env_config['llm_model']}

# Embedding config (OpenAI-compatible, SiliconFlow)
HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY={env_config['vectorize_api_key']}
HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL={env_config['vectorize_base_url']}
HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL={env_config['vectorize_model']}
HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS={env_config['vectorize_dimensions']}

# Reranker config (SiliconFlow)
HINDSIGHT_API_RERANKER_PROVIDER=siliconflow
HINDSIGHT_API_RERANKER_SILICONFLOW_API_KEY={env_config['rerank_api_key']}
HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL={env_config['rerank_model']}

# Vector and Text Search Extensions
HINDSIGHT_API_VECTOR_EXTENSION=pgvector
HINDSIGHT_API_TEXT_SEARCH_EXTENSION=pg_textsearch
"""
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info(f"Created .env file: {env_path}")
            logger.info(f"  LLM: {env_config['llm_model']} @ {env_config['llm_base_url']}")
            logger.info(f"  Embedding: {env_config['vectorize_model']} @ {env_config['vectorize_base_url']}")
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
            # Check if ALL required services are running (not just any container)
            required_services = ["db", "hindsight"]
            all_running = True
            for svc in required_services:
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "ps", "-q", svc],
                    cwd=str(compose_path.parent),
                    capture_output=True,
                    text=True,
                )
                if not result.stdout.strip():
                    all_running = False
                    logger.info(f"Service {svc} not running, will start all")
                    break

            if all_running:
                logger.info("All Docker services already running")
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
