"""
EverMemOS Native Builder - ensures .venv is activated for native Python import.

This builder does NOT start Docker - the native adapter uses:
- InMemory storage for clustering (no MongoDB needed)
- Local .pkl files for BM25/Embedding indexes (no ES/Milvus needed)
- External HTTP APIs for embedding/rerank (no local services)

Responsibilities:
1. Check if EverMemOS_bz .venv is activated
2. Auto-restart with correct .venv if not

Note: This builder does NOT start Docker or HTTP API server - the native adapter
directly imports EverMemOS Python modules.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder

logger = logging.getLogger(__name__)


@register_builder("evermemos_native")
class EverMemOSNativeBuilder(BaseBuilder):
    """
    EverMemOS Native builder - ensures correct .venv activation.

    Responsibilities:
    - Detect and activate EverMemOS_bz .venv if not already activated

    No Docker or HTTP API server needed - native adapter uses direct Python imports,
    InMemory storage, and local file indexes.
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self._started = False

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
        Verify .venv is activated, auto-restart if needed.

        Does NOT start Docker - native adapter stores data locally.

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

        self._started = True
        logger.info("✅ EverMemOS Python environment ready (no Docker needed)")
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
            if sys.platform == "win32":
                result = subprocess.run(
                    [str(venv_python), cmd_args[0]] + cmd_args[1:],
                    cwd=str(Path.cwd()),
                )
            else:
                # On Unix, use bash with source activate
                activate_script = venv_python.parent / "activate"
                cmd = f'source "{activate_script}" && python {" ".join(cmd_args)}'
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
        Cleanup (no-op, no Docker to stop).

        Returns:
            True if successful
        """
        if not self._started:
            return True

        self._started = False
        logger.info("EverMemOS Native builder cleaned up")
        return True

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
        }
