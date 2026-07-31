"""
Registry for system trackers.
"""
from typing import Dict, List, Optional, Callable

from src.trackers.system_trackers.base import _TRACKERS


def get_tracker(name: str, config: Optional[dict] = None) -> Optional[Callable]:
    """
    Get a tracker instance by system name.

    Args:
        name: System name (e.g., 'mem0', 'graphiti').
        config: Optional configuration dict to pass to tracker constructor.

    Returns:
        SystemTracker instance, or None if not found.
    """
    if name not in _TRACKERS:
        return None

    tracker_class = _TRACKERS[name]
    if config is not None:
        return tracker_class(config)
    return tracker_class()


def list_trackers() -> List[str]:
    """List all registered tracker names."""
    return list(_TRACKERS.keys())
