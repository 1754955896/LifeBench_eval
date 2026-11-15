"""
Hindsight-specific resource tracker.

Extends DefaultTracker with Hindsight-specific metrics:
- pg0 embedded database storage
- Memory engine statistics
"""
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from src.trackers.system_trackers.base import SystemSnapshot, register_tracker
from src.trackers.system_trackers.default import DefaultTracker


@register_tracker("hindsight")
class HindsightTracker(DefaultTracker):
    """
    Hindsight-specific resource tracker.

    Extends DefaultTracker with:
    - pg0 embedded database storage measurement
    - Hindsight memory engine stats (if available via config)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        pid: Optional[int] = None,
        llm_proxy_url: Optional[str] = None,
    ):
        super().__init__(config, pid, llm_proxy_url)
        self._config = config or {}

    def _find_pg0_path(self) -> Optional[Path]:
        """Find pg0 embedded data directory."""
        # pg0 stores data in ~/.pg0/ on all platforms
        home_dir = Path.home()
        pg0_home = home_dir / ".pg0"

        # Check ~/.pg0/ directly
        if pg0_home.exists() and pg0_home.is_dir():
            # Look for instances subdirectory (hindsight-embed-* profiles)
            instances_dir = pg0_home / "instances"
            if instances_dir.exists():
                for path in instances_dir.iterdir():
                    if path.is_dir() and "hindsight" in path.name.lower():
                        return path
            # Fallback: check for any pg0 data
            if any(pg0_home.iterdir()):
                return pg0_home

        # Also check temp directory (for older pg0 versions or other patterns)
        import tempfile
        temp_dir = Path(tempfile.gettempdir())

        # Look for pg0 data directories (pattern: .pg0-* or pg0-* )
        patterns = [".pg0-*", "pg0-*", "pgrst-*"]
        for pattern in patterns:
            for path in temp_dir.glob(pattern):
                if path.is_dir():
                    return path

        # Also check current working directory
        cwd = Path.cwd()
        for pattern in patterns:
            for path in cwd.glob(pattern):
                if path.is_dir():
                    return path

        return None

    def _get_pg0_size(self) -> float:
        """Get size of pg0 embedded database in MB."""
        # Dynamically find pg0 path each time (in case it wasn't started during init)
        pg0_path = self._find_pg0_path()
        if not pg0_path or not pg0_path.exists():
            return 0.0

        total_size = 0.0
        try:
            for dirpath, dirnames, filenames in os.walk(pg0_path):
                for f in filenames:
                    fp = Path(dirpath) / f
                    try:
                        total_size += fp.stat().st_size
                    except (OSError, FileNotFoundError):
                        pass
        except Exception:
            pass

        return total_size / (1024 * 1024)  # Convert to MB

    def _get_memory_engine_stats(self) -> Dict[str, Any]:
        """Get stats from Hindsight MemoryEngine if available in config."""
        stats: Dict[str, Any] = {}

        # Try to get MemoryEngine from config (set by builder)
        memory = self._config.get("_hindsight_memory")
        if memory and hasattr(memory, "backend_specific_stats"):
            try:
                stats = memory.backend_specific_stats()
            except Exception:
                pass

        return stats

    def snapshot(self) -> SystemSnapshot:
        """Take a snapshot with Hindsight-specific metrics."""
        # Get base snapshot from parent (DefaultTracker)
        snapshot = super().snapshot()

        # Add pg0 storage measurement
        pg0_size_mb = self._get_pg0_size()
        if pg0_size_mb > 0:
            snapshot.extra["pg0_storage_mb"] = round(pg0_size_mb, 3)
            snapshot.storage_mb = round(pg0_size_mb, 3)

        # Add MemoryEngine stats if available
        memory_stats = self._get_memory_engine_stats()
        if memory_stats:
            snapshot.extra["memory_engine_stats"] = memory_stats

        return snapshot

    @property
    def system_name(self) -> str:
        return "hindsight"
