"""
Base classes for system-specific resource trackers and common data structures.

This module contains:
- SystemSnapshot: snapshot dataclass for system tracker results
- SystemTracker: abstract base class for memory system trackers
- OpRecord: record of a single operation (timing + metadata)
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

    @staticmethod
    def transform_llm_request(data: dict) -> dict:
        """Transform incoming LLM request before forwarding.

        Override this in system-specific trackers to handle quirks like
        liteLLM expanding ``extra_body`` keys as top-level JSON fields.
        The proxy calls this before forwarding to the real LLM SDK.

        Returns a (possibly modified) shallow copy of *data*.
        """
        return dict(data)

    @staticmethod
    def wrap_llm_response(response_dict: dict,
                          tool_info: dict | None) -> dict:
        """Wrap LLM response after receiving it from the upstream API.

        The companion to ``transform_llm_request``.  Override when the
        request transform changes the response format (e.g. TOOLS→JSON
        mode conversion) and the response needs to be converted back.

        *tool_info* is the metadata stashed by ``transform_llm_request``
        (``data["_proxy_tool_info"]``), or ``None`` if no conversion was
        applied.

        Returns *response_dict* unchanged by default.
        """
        return response_dict


@dataclass
class OpRecord:
    """Record of a single operation's timing and metadata.

    Resource data (CPU/memory/storage) is captured by GlobalMonitor's
    timeline; slice it between op_start and op_end markers to get
    per-operation internal behaviour.
    """
    backend: str
    operation: str
    elapsed_seconds: float
    result_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "operation": self.operation,
            "timestamp": round(self.timestamp, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "result_data": self.result_data,
        }
