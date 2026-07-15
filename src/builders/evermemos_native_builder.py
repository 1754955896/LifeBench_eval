"""
EverMemOS Native Builder - manages Docker services for native Python import.

Responsibilities:
1. Check if EverMemOS_bz .venv is activated
2. Start docker-compose services (MongoDB, Elasticsearch, Milvus, Redis)
3. Wait for service readiness
4. Cleanup: stop services

Note: This builder does NOT start HTTP API server - the native adapter
directly imports EverMemOS Python modules.
"""

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder

logger = logging.getLogger(__name__)


@register_builder("evermemos_native")
class EverMemOSNativeBuilder(BaseBuilder):
    """
    EverMemOS Native builder - manages Docker services only.

    Responsibilities:
    - Detect and activate EverMemOS_bz .venv if not already activated
    - Start docker-compose services (MongoDB, Elasticsearch, Milvus, Redis)
    - Wait for readiness
    - Cleanup on shutdown

    No HTTP API server needed - native adapter uses direct Python imports.
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self._started = False
        self._venv_auto_activated = False

        # Config
        self.docker_compose = config.get(
            "docker_compose", "systems/EverMemOS_bz/docker-compose.yaml"
        )
        self.docker_wait = config.get("docker_wait", 30)

    def _get_project_root(self) -> Optional[Path]:
        """Get project root from project_root config or cli.py location."""
        if self.project_root:
            return Path(self.project_root)
        cli_path = Path(__file__).parent.parent.parent / "cli.py"
        if cli_path.exists():
            return cli_path.parent.resolve()
        return None

    def _is_venv_activated(self) -> bool:
        """Check if .venv is activated (not equal to base Python)."""
        return sys.prefix != sys.base_prefix

    def _get_venv_python(self) -> Optional[Path]:
        """Get path to EverMemOS_bz .venv Python interpreter."""
        project_root = self._get_project_root()
        if not project_root:
            return None

        evermemos_root = project_root / "systems" / "EverMemOS_bz"
        if sys.platform == "win32":
            venv_python = evermemos_root / ".venv" / "Scripts" / "python.exe"
        else:
            venv_python = evermemos_root / ".venv" / "bin" / "python"

        if venv_python.exists():
            return venv_python
        return None

    def _print_venv_activation_instructions(self):
        """Print instructions for manually activating .venv."""
        print("\n" + "=" * 60)
        print("❌ EverMemOS_bz .venv is NOT activated!")
        print("=" * 60)
        print("\nThe evermemos_native adapter requires EverMemOS_bz .venv")
        print("because it imports modules from EverMemOS_bz (memory_layer, etc.)")
        print("\nTo activate .venv, run:")
        if sys.platform == "win32":
            print("  cd systems\\EverMemOS_bz")
            print("  .venv\\Scripts\\activate")
        else:
            print("  cd systems/EverMemOS_bz")
            print("  source .venv/bin/activate")
        print("\nOr run with the correct Python:")
        print("  systems\\EverMemOS_bz\\.venv\\Scripts\\python.exe cli.py ...")
        print("=" * 60 + "\n")

    async def build(self) -> bool:
        """
        Start Docker services for EverMemOS native.

        Returns:
            True if successful, False otherwise
        """
        if self._started:
            logger.info("EverMemOS Native builder already started")
            return True

        project_root = self._get_project_root()
        if not project_root:
            logger.error("Cannot determine project root")
            return False

        # Check if .venv is activated
        if not self._is_venv_activated():
            venv_python = self._get_venv_python()
            if venv_python:
                # Found .venv Python, use it to re-launch the evaluation
                print("\n🔄 Auto-activating EverMemOS_bz .venv...")
                return await self._restart_with_venv(venv_python)
            else:
                self._print_venv_activation_instructions()
                return False

        # Start docker-compose services
        if not await self._start_docker(project_root):
            logger.error("Failed to start docker services")
            return False

        # Wait for docker services to be ready
        if not await self._wait_for_docker():
            logger.error("Docker services not ready after %ds", self.docker_wait)
            await self._stop_docker(project_root)
            return False

        self._started = True
        logger.info("✅ EverMemOS Docker services ready")
        return True

    async def _restart_with_venv(self, venv_python: Path) -> bool:
        """
        Restart the evaluation with the correct .venv Python interpreter.

        This spawns a subprocess with the .venv Python, which activates
        the .venv automatically and then runs the evaluation.
        """
        import os
        import sys as sys_module

        # Get the current command line arguments
        cmd_args = sys_module.argv

        print(f"\n🔄 Restarting with EverMemOS_bz .venv Python: {venv_python}")
        print(f"   Command: {' '.join(cmd_args)}\n")

        try:
            # Use shell=True on Windows to properly handle the activation
            if sys.platform == "win32":
                # On Windows, use cmd.exe with activate and run
                activate_script = venv_python.parent / "activate.bat"
                cmd = f'"{activate_script}" && python {" ".join(cmd_args[1:])}" if len(cmd_args) > 1 else "python"'
                # Actually, simpler: just run with the venv python directly
                result = subprocess.run(
                    [str(venv_python)] + cmd_args[1:],
                    cwd=str(Path.cwd()),
                )
            else:
                # On Unix, use bash with source activate
                activate_script = venv_python.parent / "activate"
                cmd = f'source "{activate_script}" && python {" ".join(cmd_args[1:])}'
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(Path.cwd()),
                )

            # Exit with the same exit code
            sys_module.exit(result.returncode)

        except Exception as e:
            logger.error("Failed to restart with .venv: %s", e)
            print(f"\n❌ Failed to auto-activate .venv: {e}")
            print("\nPlease activate .venv manually and try again:")
            self._print_venv_activation_instructions()
            return False

    async def cleanup(self) -> bool:
        """
        Stop Docker services.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("EverMemOS Native builder not started, nothing to cleanup")
            return True

        project_root = self._get_project_root()

        # Stop docker services
        if project_root:
            await self._stop_docker(project_root)

        self._started = False
        logger.info("EverMemOS Docker services stopped")
        return True

    async def _start_docker(self, project_root: Path) -> bool:
        """Start docker-compose services."""
        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.error("Docker compose file not found: %s", compose_path)
            return False

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
                return False

            logger.info("Docker services started")
            return True

        except Exception as e:
            logger.error("Failed to start docker services: %s", e)
            return False

    async def _wait_for_docker(self) -> bool:
        """Wait for docker services to be ready."""
        logger.info("Waiting for docker services to be ready (timeout=%ds)...", self.docker_wait)
        start_time = time.time()

        # Give services a moment to start
        await asyncio.sleep(5)

        while time.time() - start_time < self.docker_wait:
            try:
                # Check docker is running
                result = subprocess.run(
                    ["docker", "ps", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                running_containers = result.stdout.strip().split("\n")

                required_services = [
                    "memsys-mongodb",
                    "memsys-elasticsearch",
                    "memsys-milvus-standalone",
                    "memsys-redis",
                ]
                missing = [s for s in required_services if s not in running_containers]

                if not missing:
                    elapsed = time.time() - start_time
                    logger.info("All Docker services ready after %.1fs", elapsed)
                    return True

                logger.info("Waiting for services: %s", missing)

            except Exception as e:
                logger.warning("Docker health check failed: %s", e)

            await asyncio.sleep(2)

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
            "docker_compose": self.docker_compose,
        }
