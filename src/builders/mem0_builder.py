"""
Mem0 builder - starts Mem0 OSS server via docker compose.
"""

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


@register_builder("mem0")
class Mem0Builder(BaseBuilder):
    """Mem0 builder that starts/stops Mem0 OSS via docker compose.

    Configuration:
        docker_compose: Path to docker-compose.yaml (relative to project root)
        env_template: Path to .env.example template
        env_defaults: Dict of env var defaults to set
        docker_wait: Seconds to wait after starting services (default 30)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self.docker_compose = config.get("docker_compose")
        self.env_template = config.get("env_template")
        self.env_defaults = config.get("env_defaults", {})
        self.docker_wait = config.get("docker_wait", 30)
        self.post_start_script = config.get("post_start_script")
        self._started = False

    async def build(self) -> bool:
        """
        Start Mem0 OSS server via docker compose.

        Returns:
            True if successful or already running, False on failure
        """
        if self._started:
            logger.info("Mem0 builder already started")
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

        # 3. Run post-start script
        if self.post_start_script:
            if not await self._run_post_start(project_root):
                logger.warning("Post-start script failed, continuing anyway")

        self._started = True
        logger.info("Mem0 builder completed successfully")
        return True

    async def cleanup(self) -> bool:
        """
        Stop Mem0 OSS server via docker compose.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("Mem0 builder not started, nothing to cleanup")
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

        # DashScope API key from environment
        dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")

        # Default to SiliconFlow embedding, use DashScope if explicitly disabled
        use_dashscope_embedding = os.environ.get("MEM0_USE_DASHSCOPE_EMBEDDING", "false").lower() == "true"

        if use_dashscope_embedding:
            if not dashscope_api_key:
                logger.error("DASHSCOPE_API_KEY environment variable is not set")
                raise ValueError("DASHSCOPE_API_KEY is required when MEM0_USE_DASHSCOPE_EMBEDDING=true")
            # DashScope embedding
            embedding_config = f"""# DashScope Embedding settings
DASHSCOPE_API_KEY={dashscope_api_key}
DASHSCOPE_EMBEDDING_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3"""
            logger.info(f"  Embedding: DashScope text-embedding-v3 @ dashscope.aliyuncs.com")
        else:
            # SiliconFlow embedding (default, OpenAI-compatible API)
            embedding_config = f"""# SiliconFlow Embedding settings (OpenAI-compatible)
DASHSCOPE_API_KEY={env_config['vectorize_api_key']}
DASHSCOPE_EMBEDDING_URL={env_config['vectorize_base_url']}
DASHSCOPE_EMBEDDING_MODEL={env_config['vectorize_model']}"""
            logger.info(f"  Embedding: SiliconFlow {env_config['vectorize_model']} @ {env_config['vectorize_base_url']}")

        env_content = f"""# Mem0 Server Environment Configuration
# Generated by Mem0Builder from LifeBench_eval/.env

# DeepSeek LLM settings (native provider)
DEEPSEEK_API_KEY={env_config['llm_api_key']}
DEEPSEEK_API_BASE={env_config['llm_base_url']}

{embedding_config}

# Qdrant Vector Store (for hybrid search with BM25)
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=memories
QDRANT_ON_DISK=true

# PostgreSQL settings (for app data)
POSTGRES_USER=postgres
POSTGRES_DB=postgres
POSTGRES_PASSWORD=mem0dev

# Application database
APP_DB_NAME=mem0_app

# Auth disabled for development
AUTH_DISABLED=true

# Default models
MEM0_DEFAULT_LLM_MODEL={env_config['llm_model']}
MEM0_DEFAULT_EMBEDDER_MODEL=openai

# Security settings
JWT_SECRET=test-secret-key-for-dev
ADMIN_API_KEY=admin123

# Telemetry
MEM0_TELEMETRY=true

# Reranker settings (SiliconFlow llm_reranker)
MEM0_RERANKER_ENABLED=true
MEM0_RERANKER_PROVIDER=llm_reranker
MEM0_RERANKER_MODEL={env_config['rerank_model']}
MEM0_RERANKER_API_KEY={env_config['rerank_api_key']}
MEM0_RERANKER_BASE_URL={env_config['rerank_base_url']}
MEM0_RERANKER_TOP_K=10
"""
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info(f"Created .env file: {env_path}")
            logger.info(f"  LLM: {env_config['llm_model']} @ {env_config['llm_base_url']}")
            logger.info(f"  Reranker: {env_config['rerank_model']} @ {env_config['rerank_base_url']}")
            return True
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False

    async def _start_docker(self, project_root: Path) -> bool:
        """Start docker compose services, checking each one individually."""
        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.error(f"Docker compose file not found: {compose_path}")
            return False

        try:
            # Start postgres first (needed by mem0)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "postgres"],
                capture_output=True,
                text=True,
            )
            if "Up" not in result.stdout:
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
            else:
                logger.info("Postgres already running")

            # Start qdrant (vector DB, needed by mem0)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "qdrant"],
                capture_output=True,
                text=True,
            )
            if "Up" not in result.stdout:
                logger.info(f"Starting qdrant: {self.docker_compose}")
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "up", "-d", "qdrant"],
                    cwd=str(compose_path.parent),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error(f"Docker start qdrant failed: {result.stderr}")
                    return False
                logger.info("Qdrant started")
            else:
                logger.info("Qdrant already running")

            # Start mem0 main app (depends on postgres and qdrant)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "mem0"],
                capture_output=True,
                text=True,
            )
            if "Up" not in result.stdout:
                logger.info(f"Starting mem0: {self.docker_compose}")
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "up", "-d", "mem0"],
                    cwd=str(compose_path.parent),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error(f"Docker start mem0 failed: {result.stderr}")
                    return False
                logger.info("Mem0 started")
            else:
                logger.info("Mem0 already running")

            # Start mem0-dashboard (optional UI)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "mem0-dashboard"],
                capture_output=True,
                text=True,
            )
            if "Up" not in result.stdout:
                logger.info(f"Starting mem0-dashboard: {self.docker_compose}")
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "up", "-d", "mem0-dashboard"],
                    cwd=str(compose_path.parent),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error(f"Docker start mem0-dashboard failed: {result.stderr}")
                    return False
                logger.info("Mem0-dashboard started")
            else:
                logger.info("Mem0-dashboard already running")

            logger.info("Docker services ready")
            return True

        except Exception as e:
            logger.error(f"Failed to start docker services: {e}")
            return False

    async def _wait_for_postgres(self, project_root: Path) -> bool:
        """Wait for postgres container to be healthy."""
        compose_path = project_root / self.docker_compose
        max_retries = 90  # 90 * 2s = 180s
        for attempt in range(max_retries):
            try:
                # Use docker compose ps to check postgres status (works regardless of actual container name)
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "ps", "postgres"],
                    capture_output=True,
                    text=True,
                )
                # Check if postgres service is "Up" in the output
                if "Up" in result.stdout or "healthy" in result.stdout.lower():
                    logger.info("Postgres is healthy")
                    return True
                logger.debug(f"Postgres status: {result.stdout.strip()}, attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                logger.debug(f"Postgres check attempt {attempt + 1}/{max_retries}: {e}")

            if attempt < max_retries - 1:
                time.sleep(2)

        logger.error("Postgres did not become healthy in 180s")
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