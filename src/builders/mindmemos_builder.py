"""
MindMemOS Builder for LifeBench_eval.

Handles environment setup and Docker service startup for MindMemOS.
MindMemOS requires: Qdrant, Neo4j, Kafka (for async processing).
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


@register_builder("mindmemos")
class MindMemOSBuilder(BaseBuilder):
    """Builder for MindMemOS memory system."""

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        # Use project_root if provided, otherwise derive from file location
        if project_root:
            self.system_dir = Path(project_root) / "systems" / "MindMemOS"
        else:
            self.system_dir = Path(__file__).parent.parent.parent / "systems" / "MindMemOS"
        self.config_name = config.get("config_name", "dev")
        self.api_key = config.get("api_key", "dev-api-key-001")
        self.api_key_002 = config.get("api_key_002", "dev-api-key-002")  # schema memory
        self.docker_compose_file = self.system_dir / "dockers" / "docker-compose.memory.yml"
        self.service_url = config.get("service_url", "http://127.0.0.1:8000")

    def setup_environment(self) -> None:
        """Set up environment variables for MindMemOS."""
        # Set config file
        os.environ["MINDMEMOS_CONFIG_NAME"] = self.config_name

        # Set service URL
        os.environ["MINDMEMOS_SERVICE_URL"] = self.service_url

        # Copy .env file if it doesn't exist
        env_file = self.system_dir / ".env"
        env_example = self.system_dir / ".env.example"
        if not env_file.exists() and env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)

        # Copy config file if dev.yaml doesn't exist
        config_dir = self.system_dir / "config" / "mindmemos"
        dev_config = config_dir / f"{self.config_name}.yaml"
        dev_example = config_dir / f"{self.config_name}.example.yaml"
        if not dev_config.exists() and dev_example.exists():
            import shutil
            shutil.copy(dev_example, dev_config)

        # Ensure MindMemOS is in sys.path
        mindmemos_src = self.system_dir / "src"
        if str(mindmemos_src) not in sys.path:
            sys.path.insert(0, str(mindmemos_src))

        logger.info(f"MindMemOS environment configured")
        logger.info(f"  Config: {self.config_name}")
        logger.info(f"  Service URL: {self.service_url}")
        logger.info(f"  System dir: {self.system_dir}")

    def _check_docker_service(self, container_name: str) -> bool:
        """Check if a Docker container is running."""
        try:
            # MindMemOS containers are prefixed with "mindmemos-"
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=mindmemos-" + container_name, "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return f"mindmemos-{container_name}" in result.stdout
        except Exception:
            return False

    def _wait_for_service(self, url: str, timeout: int = 60) -> bool:
        """Wait for MindMemOS service to be ready."""
        import urllib.request

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                req = urllib.request.Request(url)
                urllib.request.urlopen(req, timeout=5)
                return True
            except Exception:
                time.sleep(2)
        return False

    def start_docker_services(self) -> None:
        """Start Docker services for MindMemOS (Qdrant, Neo4j, Kafka)."""
        logger.info("Starting Docker services for MindMemOS...")

        if not self.docker_compose_file.exists():
            logger.warning(f"Docker compose file not found: {self.docker_compose_file}")
            return

        try:
            # Check if services are already running
            if self._check_docker_service("qdrant") and self._check_docker_service("neo4j") and self._check_docker_service("kafka"):
                logger.info("Docker services already running")
            else:
                # Start services
                subprocess.run(
                    [
                        "docker", "compose",
                        "--env-file", str(self.system_dir / ".env"),
                        "-f", str(self.docker_compose_file),
                        "up", "-d", "--wait",
                        "qdrant", "neo4j", "kafka", "kafka-ui", "kafka-exporter"
                    ],
                    cwd=str(self.system_dir),
                    check=True,
                    timeout=120
                )
                logger.info("Docker services started")

            # Wait for Qdrant
            logger.info("Waiting for Qdrant...")
            qdrant_ready = self._wait_for_service("http://localhost:6333", timeout=30)
            if qdrant_ready:
                logger.info("Qdrant is ready")
            else:
                logger.warning("Qdrant not ready after 30s")

            # Wait for Neo4j
            logger.info("Waiting for Neo4j...")
            neo4j_ready = self._wait_for_service("http://localhost:7474", timeout=30)
            if neo4j_ready:
                logger.info("Neo4j is ready")
            else:
                logger.warning("Neo4j not ready after 30s")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker services: {e}")
        except Exception as e:
            logger.error(f"Error starting Docker services: {e}")

    def start_api_server(self) -> Optional[subprocess.Popen]:
        """Start MindMemOS FastAPI server."""
        logger.info("Starting MindMemOS API server...")

        # Use the pre-installed .venv - check both Windows and Unix paths
        venv_python = self.system_dir / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = self.system_dir / ".venv" / "bin" / "python"  # Linux fallback
        if not venv_python.exists():
            logger.error(f"MindMemOS .venv not found at {self.system_dir / '.venv'}")
            return None

        # Set environment for API
        env = os.environ.copy()
        env["MINDMEMOS_CONFIG_NAME"] = self.config_name

        # Start API server using the venv python
        # On Windows, use shell=True with string command
        import platform
        is_windows = platform.system() == "Windows"

        if is_windows:
            cmd = f'"{venv_python}" -m uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000'
            process = subprocess.Popen(
                cmd,
                cwd=str(self.system_dir),
                env=env,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            cmd = [str(venv_python), "-m", "uvicorn", "mindmemos.api.app:app", "--host", "127.0.0.1", "--port", "8000"]
            process = subprocess.Popen(
                cmd,
                cwd=str(self.system_dir),
                env=env,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

        logger.info("MindMemOS API server starting...")

        # Wait a bit and check if process died
        time.sleep(10)
        if process.poll() is not None:
            # Process died, get output
            output = process.stdout.read() if process.stdout else ""
            logger.error(f"API server process died. Output: {output}")
            return None

        # Wait for API to be ready
        if self._wait_for_service(f"{self.service_url}/healthz", timeout=60):
            logger.info(f"MindMemOS API ready at {self.service_url}")
            return process
        else:
            # Check if still running
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                logger.error(f"API server died during startup. Output: {output}")
            else:
                logger.warning("MindMemOS API not ready after 60s")
            return process

    async def build(self) -> bool:
        """Build the MindMemOS environment."""
        self.setup_environment()
        self.start_docker_services()

        # Verify Docker services are running
        logger.info("Checking Docker services...")
        if not self._check_docker_service("qdrant"):
            logger.error("Qdrant is not running. Please fix Docker setup.")
            return False
        if not self._check_docker_service("neo4j"):
            logger.error("Neo4j is not running. Please fix Docker setup.")
            return False
        logger.info("Docker services are running")

        # Start the API server
        api_process = self.start_api_server()
        if api_process is None:
            logger.error("Failed to start API server")
            return False
        return True

    async def cleanup(self) -> bool:
        """Clean up MindMemOS resources."""
        logger.info("Cleaning up MindMemOS resources...")
        # Note: We don't stop Docker services as they may be used by other systems
        return True
