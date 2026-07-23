"""
Base classes for system-specific resource trackers and common data structures.

This module contains:
- SystemSnapshot: snapshot dataclass for system tracker results
- SystemTracker: abstract base class for memory system trackers
- ResourceSnapshot: generic resource snapshot for PerOpTracker
- OpRecord: record of a single operation's resource usage
- _system_snapshot_to_resource: converter function
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Callable


# Registry of system trackers - populated by @register_tracker decorator
_TRACKERS: Dict[str, Callable] = {}


def register_tracker(name: str) -> Callable:
    """
    Decorator to register a system tracker.

    Usage:
        @register_tracker("mem0")
        class Mem0Tracker(SystemTracker):
            ...
    """
    def decorator(cls: Callable) -> Callable:
        _TRACKERS[name] = cls
        return cls
    return decorator


@dataclass
class SystemSnapshot:
    """
    Snapshot of system-specific resource usage.

    Each system tracker returns its own set of metrics in the `extra` field.
    """
    storage_mb: float = 0.0
    memory_rss_mb: float = 0.0
    cpu_percent: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_mb": round(self.storage_mb, 3),
            "memory_rss_mb": round(self.memory_rss_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "extra": self.extra,
        }


class SystemTracker(ABC):
    """
    Abstract base class for memory system resource trackers.

    Each memory system (mem0, graphiti, hindsight, etc.) implements this interface
    to provide system-specific resource metrics.
    """

    @property
    @abstractmethod
    def system_name(self) -> str:
        """Return the memory system name (e.g., 'mem0', 'graphiti')."""
        pass

    @abstractmethod
    def snapshot(self) -> SystemSnapshot:
        """Take a snapshot of current resource usage."""
        pass

    def backend_specific_stats(self) -> Dict[str, Any]:
        """Get backend-specific statistics."""
        return {}


@dataclass
class ResourceSnapshot:
    """A snapshot of resource usage at a point in time."""
    storage_mb: float = 0.0
    storage_delta_mb: float = 0.0
    memory_rss_mb: float = 0.0
    memory_vms_mb: float = 0.0
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_mb": round(self.storage_mb, 3),
            "memory_rss_mb": round(self.memory_rss_mb, 2),
            "memory_vms_mb": round(self.memory_vms_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "extra": self.extra,
        }

    def __sub__(self, other: "ResourceSnapshot") -> "ResourceSnapshot":
        """Subtract two snapshots to get delta."""
        return ResourceSnapshot(
            storage_delta_mb=self.storage_mb - other.storage_mb,
            memory_delta_mb=self.memory_rss_mb - other.memory_rss_mb,
            memory_rss_mb=self.memory_rss_mb,
            memory_vms_mb=self.memory_vms_mb,
            cpu_percent=max(self.cpu_percent, other.cpu_percent),
            extra=self.extra,
        )


@dataclass
class OpRecord:
    """Record of a single operation's resource usage."""
    backend: str
    operation: str
    elapsed_seconds: float
    before: ResourceSnapshot
    after: ResourceSnapshot
    delta: ResourceSnapshot = field(init=False)
    result_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=0.0)

    def __post_init__(self):
        self.delta = self.after - self.before
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "operation": self.operation,
            "timestamp": round(self.timestamp, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "storage_mb": round(self.after.storage_mb, 3),
            "storage_delta_mb": round(self.delta.storage_delta_mb, 3),
            "memory_rss_mb": round(self.after.memory_rss_mb, 2),
            "memory_delta_mb": round(self.delta.memory_delta_mb, 2),
            "cpu_percent": round(self.delta.cpu_percent, 2),
            "extra": self.delta.extra,
            "result_data": self.result_data,
        }


def _system_snapshot_to_resource(snap: SystemSnapshot) -> ResourceSnapshot:
    """Convert SystemSnapshot to ResourceSnapshot."""
    return ResourceSnapshot(
        storage_mb=snap.storage_mb,
        memory_rss_mb=snap.memory_rss_mb,
        cpu_percent=snap.cpu_percent,
        extra=snap.extra,
    )
