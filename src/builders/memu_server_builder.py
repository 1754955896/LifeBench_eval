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

        # 2. Start docker services (with per-service health checks)
        if not await self._start_docker(project_root):
            return False

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

    def _run_docker(self, compose_dir: Path, args: list) -> subprocess.CompletedProcess:
        """Run a docker compose command.

        Args:
            compose_dir: Directory containing docker-compose.yml
            args: Command arguments (e.g. ["up", "-d", "postgres"])
        """
        return subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yml"] + args,
            cwd=str(compose_dir),
            capture_output=True,
            text=True,
        )

    def _get_service_status(self, project_root: Path) -> dict[str, dict]:
        """Get status of all services.

        Returns:
            Dict mapping service name to {"running": bool, "healthy": bool|None}
        """
        compose_path = project_root / self.docker_compose
        compose_dir = compose_path.parent
        result = self._run_docker(compose_dir, ["ps", "--format", "json"])

        services = {}
        if result.returncode != 0:
            logger.warning(f"Failed to get service status: {result.stderr}")
            return services

        # Parse JSON output (one JSON object per line)
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                import json
                info = json.loads(line)
                name = info.get("Service", "")
                state = info.get("State", "").lower()
                health = info.get("Health", "").lower() if info.get("Health") else None
                services[name] = {
                    "running": state == "running",
                    "healthy": health == "healthy" if health else None,
                }
            except json.JSONDecodeError:
                continue

        return services

    def _start_service(self, project_root: Path, service: str) -> bool:
        """Start a specific service.

        Returns:
            True if service started successfully or port already in use (external service).
        """
        compose_dir = (project_root / self.docker_compose).parent
        logger.info(f"Starting service: {service}")
        result = self._run_docker(compose_dir, ["up", "-d", service])
        if result.returncode != 0:
            # Check if port already allocated (service may be running externally)
            stderr = result.stderr.lower()
            if "port is already allocated" in stderr or "bind for" in stderr:
                logger.warning(f"Port for {service} already in use - assuming external service")
                return True
            logger.error(f"Failed to start {service}: {result.stderr}")
            return False
        return True

    def _wait_for_service(
        self, project_root: Path, service: str, timeout: int = 60
    ) -> bool:
        """Wait for a service to be running and healthy (if applicable).

        Args:
            project_root: Project root path
            service: Service name
            timeout: Max seconds to wait

        Returns:
            True if service is running (and healthy if healthcheck defined)
        """
        compose_dir = (project_root / self.docker_compose).parent
        start = time.time()

        # For memu-api, we need to do HTTP health check since it has no docker healthcheck
        is_http_service = service == "memu-api"

        while time.time() - start < timeout:
            status = self._get_service_status(project_root)
            svc_info = status.get(service, {})

            # Service not found in docker compose ps output - might be external
            if not svc_info:
                logger.info(f"Service {service} not found in docker compose - may be external")
                return True

            if svc_info.get("running"):
                # If no health check defined, consider it ready
                if svc_info.get("healthy") is None:
                    # For HTTP services, verify the API is actually responding
                    if is_http_service:
                        if self._check_http_health("http://localhost:8000/"):
                            logger.info(f"Service {service} is running and responding to HTTP")
                            return True
                        logger.info(f"Service {service} is running but not responding to HTTP yet...")
                    else:
                        logger.info(f"Service {service} is running")
                        return True
                # If health check exists, require healthy status
                if svc_info.get("healthy"):
                    logger.info(f"Service {service} is healthy")
                    return True
                logger.info(f"Service {service} is running but not healthy yet...")
            else:
                logger.info(f"Service {service} is not running yet...")

            time.sleep(2)

        logger.error(f"Service {service} failed to become ready within {timeout}s")
        return False

    def _check_http_health(self, url: str, timeout: int = 10) -> bool:
        """Check if an HTTP endpoint is responding."""
        try:
            import urllib.request
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _start_docker(self, project_root: Path) -> bool:
        """Start docker compose services, checking each one individually."""
        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.error(f"Docker compose file not found: {compose_path}")
            return False

        try:
            # Define required services in order (dependencies first)
            required_services = ["postgres", "temporal", "temporal-ui", "memu-api"]

            # Check current status of all services
            current_status = self._get_service_status(project_root)
            logger.info(f"Current service status: {current_status}")

            for service in required_services:
                svc_info = current_status.get(service, {})

                # For memu-api, always verify HTTP health even if running
                if service == "memu-api" and svc_info.get("running"):
                    if self._check_http_health("http://localhost:8000/"):
                        logger.info(f"Service {service} is already ready (HTTP healthy)")
                        continue
                    else:
                        logger.info(f"Service {service} is running but not responding to HTTP, restarting...")
                        # Force restart the service
                        compose_dir = (project_root / self.docker_compose).parent
                        self._run_docker(compose_dir, ["restart", service])
                        if not self._wait_for_service(project_root, service):
                            logger.error(f"Service {service} failed to start properly")
                            return False
                        continue

                if svc_info.get("running") and (
                    svc_info.get("healthy") is True or svc_info.get("healthy") is None
                ):
                    logger.info(f"Service {service} is already ready")
                    continue

                # Service not running or not healthy - start/restart it
                if not self._start_service(project_root, service):
                    return False

                # Wait for this service to be ready
                if not self._wait_for_service(project_root, service):
                    logger.error(f"Service {service} failed to start properly")
                    return False

            logger.info("All services are ready")
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
