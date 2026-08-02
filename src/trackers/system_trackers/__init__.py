"""
System trackers - per-memory-system resource trackers.

Each memory system (mem0, graphiti, hindsight, etc.) has its own storage backend
and requires a custom tracker implementation. This module provides the base class
and registry for system trackers.

Usage:
    from src.trackers.system_trackers import get_tracker

    tracker = get_tracker("default")  # Returns DefaultTracker instance
"""
from src.trackers.system_trackers.base import SystemTracker, register_tracker
from src.trackers.system_trackers.registry import get_tracker, list_trackers

# Import all trackers to register them
from src.trackers.system_trackers.default import DefaultTracker
from src.trackers.system_trackers.hindsight import HindsightTracker  # noqa: F401
from src.trackers.system_trackers.cognee import CogneeTracker  # noqa: F401

__all__ = [
    "SystemTracker",
    "DefaultTracker",
    "HindsightTracker",
    "CogneeTracker",
    "register_tracker",
    "get_tracker",
    "list_trackers",
]
