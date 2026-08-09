"""
EverMemOS Builder - manages EverMemOS HTTP service lifecycle.

Responsibilities:
1. Start docker-compose services (MongoDB, Elasticsearch, Milvus, Redis)
2. Start the FastAPI HTTP server in background
3. Wait for service readiness
4. Cleanup: stop services

Note: This builder manages the HTTP API server, not Python imports.
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


@register_builder("evermemos")
class EverMemOSBuilder(BaseBuilder):
    """
    EverMemOS builder - manages HTTP service lifecycle.

    Responsibilities:
    - Start docker-compose services
    - Start FastAPI server
    - Wait for readiness
    - Cleanup on shutdown
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self._started = False
        self._api_process: Optional[subprocess.Popen] = None

        # Config
        self.docker_compose = config.get("docker_compose", "systems/EverMemOS_bz/docker-compose.yaml")
        self.api_url = config.get("api_url", "http://localhost:8001")
        self.api_wait = config.get("api_wait", 60)
        self.env_file = config.get("env_file", ".env")

    def _get_project_root(self) -> Optional[Path]:
        """Get project root from project_root config or cli.py location."""
        if self.project_root:
            return Path(self.project_root)
        cli_path = Path(__file__).parent.parent.parent / "cli.py"
        if cli_path.exists():
            return cli_path.parent.resolve()
        return None

    async def build(self) -> bool:
        """
        Start EverMemOS HTTP service.

        Returns:
            True if successful, False otherwise
        """
        if self._started:
            logger.info("EverMemOS builder already started")
            return True

        project_root = self._get_project_root()
        if not project_root:
            logger.error("Cannot determine project root")
            return False

        # 1. Start docker-compose services
        containers_existed = await self._start_docker(project_root)
        if containers_existed is None:
            logger.error("Failed to start docker services")
            return False

        if containers_existed:
            logger.info("Docker services already running")
        else:
            logger.info("Waiting 120s for docker services to be ready...")
            await asyncio.sleep(120)

        # 2. Start FastAPI server
        if not await self._start_api_server(project_root):
            logger.error("Failed to start API server")
            await self._stop_docker(project_root)
            return False

        # 3. Wait for API to be ready
        if not await self._wait_for_api():
            logger.error("API server not ready after %ds", self.api_wait)
            await self._stop_api_server()
            await self._stop_docker(project_root)
            return False

        self._started = True
        logger.info("✅ EverMemOS HTTP service ready at %s", self.api_url)
        return True

    async def cleanup(self) -> bool:
        """
        Stop EverMemOS HTTP service.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("EverMemOS builder not started, nothing to cleanup")
            return True

        project_root = self._get_project_root()

        # Stop API server
        await self._stop_api_server()

        # Stop docker services
        if project_root:
            await self._stop_docker(project_root)

        self._started = False
        logger.info("EverMemOS HTTP service stopped")
        return True

    async def _start_docker(self, project_root: Path):
        """Start docker-compose services.

        Returns:
            True if containers were already running (skip wait),
            False if containers were just started (need wait),
            None if failed.
        """
        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.error("Docker compose file not found: %s", compose_path)
            return None

        try:
            # Check if already running
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "-q"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                logger.info("Docker services already running")
                return True

            # Start services
            logger.info("Starting docker services: %s", self.docker_compose)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "up", "-d"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Docker start failed: %s", result.stderr)
                return None

            logger.info("Docker services started")
            return False

        except Exception as e:
            logger.error("Failed to start docker services: %s", e)
            return None

    async def _start_api_server(self, project_root: Path) -> bool:
        """Start FastAPI server in background process."""
        # Find EverMemOS_bz src directory
        evermemos_src = project_root / "systems" / "EverMemOS_bz" / "src"
        if not evermemos_src.exists():
            logger.error("EverMemOS_bz src not found: %s", evermemos_src)
            return False

        run_py = evermemos_src / "run.py"
        if not run_py.exists():
            logger.error("run.py not found: %s", run_py)
            return False

        # Use the venv Python from EverMemOS_bz
        venv_python = project_root / "systems" / "EverMemOS_bz" / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            logger.error("EverMemOS venv Python not found: %s", venv_python)
            return False

        # Check if API server is already running
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.api_url}/docs") as resp:
                    if resp.status == 200:
                        logger.info("API server already running at %s", self.api_url)
                        return True
        except Exception:
            pass

        try:
            logger.info("Starting FastAPI server: %s", run_py)

            # env_file is relative to project_root (LifeBench_eval)
            env_file_path = project_root / self.env_file

            # Start server process using venv Python
            # Set PYTHONIOENCODING=utf-8 to prevent GBK encoding crashes on Windows
            # (server log messages contain emoji like ✅, ❌ that crash with GBK encoding)
            server_env = os.environ.copy()
            server_env["PYTHONIOENCODING"] = "utf-8"
            server_env["PYTHONUTF8"] = "1"

            # Persist server stdout/stderr to a log file so internal processing
            # (DB sync, LLM call gaps, etc.) can be inspected when diagnosing slowness.
            # Written into the run's output dir (config["output_dir"]), falling
            # back to results/ for backward compatibility.
            server_log_dir = self.config.get("output_dir")
            server_log = (
                Path(server_log_dir) / "evermemos_server.log"
                if server_log_dir
                else project_root / "results" / "evermemos_server.log"
            )
            server_log.parent.mkdir(parents=True, exist_ok=True)
            server_log_handle = open(server_log, "a", encoding="utf-8", errors="replace")
            self._server_log_handle = server_log_handle

            self._api_process = subprocess.Popen(
                [str(venv_python), str(run_py), "--port", "8001", "--env-file", str(env_file_path)],
                cwd=str(evermemos_src),
                stdout=server_log_handle,
                stderr=subprocess.STDOUT,
                env=server_env,
            )
            logger.info("API server logs: %s", server_log)

            logger.info("API server process started (pid=%d)", self._api_process.pid)
            return True

        except Exception as e:
            logger.error("Failed to start API server: %s", e)
            return False

    async def _wait_for_api(self) -> bool:
        """Wait for API server to be ready."""
        import aiohttp

        logger.info("Waiting for API server to be ready (timeout=%ds)...", self.api_wait)
        start_time = time.time()

        while time.time() - start_time < self.api_wait:
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    async with session.get(f"{self.api_url}/docs") as resp:
                        if resp.status == 200:
                            # Server is up, but wait a bit more for lifespan to complete
                            await asyncio.sleep(3)
                            # Double-check it's still up
                            async with session.get(f"{self.api_url}/docs") as resp2:
                                if resp2.status == 200:
                                    elapsed = time.time() - start_time
                                    logger.info("API server ready after %.1fs", elapsed)
                                    return True
            except Exception:
                pass

            await asyncio.sleep(2)

        return False

    async def _stop_api_server(self) -> bool:
        """Stop FastAPI server process."""
        if self._api_process is None:
            return True

        try:
            logger.info("Stopping API server (pid=%d)", self._api_process.pid)
            self._api_process.terminate()
            try:
                self._api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._api_process.kill()
                self._api_process.wait()
            logger.info("API server stopped")
            self._api_process = None
            return True
        except Exception as e:
            logger.error("Failed to stop API server: %s", e)
            return False

    async def _stop_docker(self, project_root: Path) -> bool:
        """Stop docker-compose services."""
        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
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
                logger.error("Docker stop failed: %s", result.stderr)
                return False
            logger.info("Docker services stopped")
            return True
        except Exception as e:
            logger.error("Failed to stop docker services: %s", e)
            return False

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "api_url": self.api_url,
            "docker_compose": self.docker_compose,
        }
