"""
TencentDB-Agent-Memory builder - starts Gateway via docker.
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
        Dict with api_key, base_url, model settings
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

    return config


@register_builder("tencentdb")
class TencentDBBuilder(BaseBuilder):
    """TencentDB-Agent-Memory builder that starts Gateway via docker.

    Configuration:
        dockerfile: Path to Dockerfile (relative to project root)
        image_name: Docker image name (default: hermes-memory)
        container_name: Container name (default: hermes-memory)
        gateway_port: Host port for Gateway (default: 8420)
        docker_wait: Seconds to wait after starting container (default: 30)
        memory_config: Dict with memory plugin settings (embedding, recall, etc.)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self.dockerfile = config.get("dockerfile", "systems/TencentDB-Agent-Memory/docker/opensource/Dockerfile.hermes")
        self.image_name = config.get("image_name", "hermes-memory")
        self.container_name = config.get("container_name", "hermes-memory")
        self.gateway_port = config.get("gateway_port", 8420)
        self.docker_wait = config.get("docker_wait", 30)
        self.memory_config = config.get("memory", {})
        self._started = False

    async def build(self) -> bool:
        """
        Build and start TencentDB-Agent-Memory Gateway via docker.

        Returns:
            True if successful or already running, False on failure
        """
        if self._started:
            logger.info("TencentDB builder already started")
            return True

        project_root = Path(self.project_root) if self.project_root else self._get_project_root()
        if not project_root:
            logger.error("Cannot determine project root")
            return False

        # 1. Build Docker image
        if not await self._build_image(project_root):
            logger.error("Failed to build Docker image")
            return False

        # 2. Start container
        if not await self._start_container(project_root):
            logger.error("Failed to start container")
            return False

        # 3. Wait for Gateway to be ready
        logger.info(f"Waiting {self.docker_wait}s for Gateway to be ready...")
        time.sleep(self.docker_wait)

        # 4. Verify Gateway health
        if not await self._wait_for_gateway():
            logger.error("Gateway not ready after waiting")
            return False

        self._started = True
        logger.info("TencentDB builder completed successfully")
        return True

    async def cleanup(self) -> bool:
        """
        Stop and remove TencentDB container.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("TencentDB builder not started, nothing to cleanup")
            return True

        project_root = Path(self.project_root) if self.project_root else self._get_project_root()
        if not project_root:
            return True

        try:
            # Stop and remove container
            logger.info(f"Stopping container: {self.container_name}")
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                text=True,
            )
            logger.info("Container stopped and removed")
            self._started = False
            return True
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
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

    async def _build_image(self, project_root: Path) -> bool:
        """Build Docker image."""
        dockerfile_path = project_root / self.dockerfile
        if not dockerfile_path.exists():
            logger.error(f"Dockerfile not found: {dockerfile_path}")
            return False

        # Build context should be the directory containing src/, package.json
        # dockerfile is like "systems/TencentDB-Agent-Memory/docker/opensource/Dockerfile.hermes"
        # so build_context = project_root / "systems/TencentDB-Agent-Memory"
        dockerfile_relative = self.dockerfile  # e.g. "systems/TencentDB-Agent-Memory/docker/opensource/Dockerfile.hermes"
        build_context = project_root / Path(*dockerfile_relative.split("/")[:2])  # systems/TencentDB-Agent-Memory

        logger.info(f"Building Docker image: {self.image_name} (context={build_context})")
        try:
            result = subprocess.run(
                ["docker", "build", "-f", str(dockerfile_path), "-t", self.image_name, "."],
                cwd=str(build_context),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Docker build failed: {result.stderr[:500]}")
                return False

            logger.info("Docker image built successfully")
            return True

        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            return False

    async def _start_container(self, project_root: Path) -> bool:
        """Start Docker container."""
        # Load env config
        project_env = project_root / ".env"
        env_config = _load_env_config(project_env)

        # Stop existing container if present
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            text=True,
        )

        # Generate Gateway config file
        gateway_config = self._generate_gateway_config(env_config)
        config_dir = project_root / "systems" / "TencentDB-Agent-Memory"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "tdai-gateway.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(gateway_config)
        logger.info(f"Generated Gateway config: {config_file}")

        # Build docker run command
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--restart", "unless-stopped",
            "-p", f"{self.gateway_port}:{self.gateway_port}",
            "-v", "hermes_memory_data:/opt/data",
            "-v", f"{config_file}:/opt/data/tdai-gateway.yaml",
        ]

        # Add environment variables for model config
        if env_config["llm_api_key"]:
            cmd.extend(["-e", f"MODEL_API_KEY={env_config['llm_api_key']}"])
        if env_config["llm_base_url"]:
            cmd.extend(["-e", f"MODEL_BASE_URL={env_config['llm_base_url']}"])
        if env_config["llm_model"]:
            cmd.extend(["-e", f"MODEL_NAME={env_config['llm_model']}"])
        else:
            cmd.extend(["-e", "MODEL_NAME=deepseek-v4-flash"])

        # Default provider
        cmd.extend(["-e", "MODEL_PROVIDER=custom"])

        # Environment variables for embedding (if configured)
        memory_cfg = self.memory_config
        embedding_cfg = memory_cfg.get("embedding", {})
        if embedding_cfg.get("enabled", True):
            embed_base_url = embedding_cfg.get("baseUrl", "")
            embed_api_key = embedding_cfg.get("apiKey", "")
            embed_model = embedding_cfg.get("model", "")
            embed_dims = embedding_cfg.get("dimensions", 1024)

            if embed_base_url:
                cmd.extend(["-e", f"TDAI_EMBEDDING_BASE_URL={embed_base_url}"])
            if embed_api_key:
                cmd.extend(["-e", f"TDAI_EMBEDDING_API_KEY={embed_api_key}"])
            if embed_model:
                cmd.extend(["-e", f"TDAI_EMBEDDING_MODEL={embed_model}"])
            if embed_dims:
                cmd.extend(["-e", f"TDAI_EMBEDDING_DIMENSIONS={embed_dims}"])

        cmd.append(self.image_name)

        logger.info(f"Starting container: {self.container_name}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Container start failed: {result.stderr[:500]}")
                return False

            logger.info("Container started successfully")
            return True

        except Exception as e:
            logger.error(f"Container start failed: {e}")
            return False

    def _generate_gateway_config(self, env_config: dict) -> str:
        """Generate Gateway config YAML from memory_config."""
        memory_cfg = self.memory_config

        # Extract recall settings
        recall_cfg = memory_cfg.get("recall", {})
        recall_strategy = recall_cfg.get("strategy", "hybrid")
        recall_max_results = recall_cfg.get("maxResults", 5)
        recall_timeout = recall_cfg.get("timeoutMs", 5000)

        # Extract embedding settings
        embed_cfg = memory_cfg.get("embedding", {})
        embed_enabled = embed_cfg.get("enabled", True)
        embed_provider = embed_cfg.get("provider", "openai")
        embed_base_url = embed_cfg.get("baseUrl", "")
        embed_api_key = embed_cfg.get("apiKey", "")
        embed_model = embed_cfg.get("model", "")
        embed_dims = embed_cfg.get("dimensions", 1024)
        embed_send_dims = embed_cfg.get("sendDimensions", True)

        # Extract BM25 settings
        bm25_cfg = memory_cfg.get("bm25", {})
        bm25_enabled = bm25_cfg.get("enabled", True)
        bm25_lang = bm25_cfg.get("language", "zh")

        # Extract LLM settings for memory extraction
        llm_cfg = memory_cfg.get("llm", {})
        llm_enabled = llm_cfg.get("enabled", False)
        llm_base_url = llm_cfg.get("baseUrl", "")
        llm_api_key = llm_cfg.get("apiKey", "")
        llm_model = llm_cfg.get("model", "")
        llm_max_tokens = llm_cfg.get("maxTokens", 4096)
        llm_timeout = llm_cfg.get("timeoutMs", 120000)

        # Build config
        config_lines = [
            "# TencentDB-Agent-Memory Gateway Config",
            "# Generated by TencentDBBuilder",
            "",
            "data:",
            "  baseDir: /opt/data/tdai-memory",
            "",
            "llm:",
            f"  baseUrl: {llm_base_url or 'https://api.deepseek.com'}",
            f"  apiKey: {llm_api_key}",
            f"  model: {llm_model or 'deepseek-chat'}",
            f"  maxTokens: {llm_max_tokens}",
            f"  timeoutMs: {llm_timeout}",
            "",
            "memory:",
            "  storeBackend: sqlite",
            "",
            "  recall:",
            f"    strategy: {recall_strategy}",
            f"    maxResults: {recall_max_results}",
            f"    timeoutMs: {recall_timeout}",
            "",
            "  embedding:",
            f"    enabled: {str(embed_enabled).lower()}",
            f"    provider: {embed_provider}",
            f"    baseUrl: {embed_base_url}",
            f"    apiKey: {embed_api_key}",
            f"    model: {embed_model}",
            f"    dimensions: {embed_dims}",
            f"    sendDimensions: {str(embed_send_dims).lower()}",
            "",
            "  bm25:",
            f"    enabled: {str(bm25_enabled).lower()}",
            f"    language: {bm25_lang}",
        ]

        if llm_enabled:
            config_lines.extend([
                "",
                "  llm:",
                f"    enabled: true",
                f"    baseUrl: {llm_base_url}",
                f"    apiKey: {llm_api_key}",
                f"    model: {llm_model}",
                f"    maxTokens: {llm_max_tokens}",
                f"    timeoutMs: {llm_timeout}",
            ])

        return "\n".join(config_lines)

    async def _wait_for_gateway(self, max_retries=12) -> bool:
        """Wait for Gateway to be ready."""
        import urllib.request
        import urllib.error

        gateway_url = f"http://localhost:{self.gateway_port}/health"

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(gateway_url, method="GET")
                response = urllib.request.urlopen(req, timeout=5)
                if response.status == 200:
                    logger.info(f"Gateway is ready (attempt {attempt + 1})")
                    return True
            except (urllib.error.HTTPError, Exception):
                pass

            if attempt < max_retries - 1:
                time.sleep(5)

        return False

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "image_name": self.image_name,
            "container_name": self.container_name,
            "gateway_port": self.gateway_port,
        }
