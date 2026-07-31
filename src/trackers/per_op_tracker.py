"""
Per-operation resource tracker.

Tracks resource usage for single operations (add/search) by capturing
before/after snapshots using a SystemTracker.
"""
import json
import time
from pathlib import Path
from typing import Any, List, Optional

from src.trackers.system_trackers.base import (
    SystemTracker,
    SystemSnapshot,
    ResourceSnapshot,
    OpRecord,
)


def _system_snapshot_to_resource(snap: SystemSnapshot) -> ResourceSnapshot:
    """Convert SystemSnapshot to ResourceSnapshot."""
    return ResourceSnapshot(
        storage_mb=snap.storage_mb,
        memory_rss_mb=snap.memory_rss_mb,
        cpu_percent=snap.cpu_percent,
        extra=snap.extra,
    )


class PerOpTracker:
    """
    Per-operation resource tracker.

    Wraps before/after snapshot logic for tracking single operations.
    Used by Pipeline to record resource usage per add/search operation.

    Usage:
        tracker = PerOpTracker(get_tracker("mem0"), output_dir=Path("results"))
        with tracker.track("add") as ctx:
            result = await adapter.add_chunks(chunks)
        record = ctx.record(result_data={"added": 10, "failed": 0})
        tracker.save()
    """

    def __init__(self, tracker: SystemTracker, output_dir: Optional[Path] = None):
        self.tracker = tracker
        self.output_dir = Path(output_dir) if output_dir else None
        self._records: List[OpRecord] = []

    def track(self, operation: str) -> "_OpContext":
        return _OpContext(self.tracker, operation, self)

    def add_record(self, record: OpRecord) -> None:
        """Add a record to the tracker."""
        self._records.append(record)
        self.save()

    def save(self) -> None:
        """Save records to JSON file."""
        if not self.output_dir or not self._records:
            return

        tracker_dir = self.output_dir / "tracker"
        tracker_dir.mkdir(parents=True, exist_ok=True)
        filepath = tracker_dir / "tracker_records.json"

        data = [r.to_dict() for r in self._records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_records(self) -> List[OpRecord]:
        """Get all records."""
        return list(self._records)


class _OpContext:
    """Internal context manager for per-operation tracking."""

    def __init__(self, tracker: SystemTracker, operation: str, parent: PerOpTracker):
        self.tracker = tracker
        self.operation = operation
        self.parent = parent
        self._before: Optional[ResourceSnapshot] = None
        self._start: float = 0.0
        self._after: Optional[ResourceSnapshot] = None
        self._elapsed: float = 0.0

    def __enter__(self) -> "_OpContext":
        snap = self.tracker.snapshot()
        self._before = _system_snapshot_to_resource(snap)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._elapsed = time.perf_counter() - self._start
        snap = self.tracker.snapshot()
        self._after = _system_snapshot_to_resource(snap)
        return False

    def record(self, result_data: Any = None) -> OpRecord:
        """Create an OpRecord from the tracked operation."""
        import time
        if self._before is None or self._after is None:
            raise RuntimeError("Cannot record without entering context")
        op_record = OpRecord(
            backend=self.tracker.system_name,
            operation=self.operation,
            elapsed_seconds=self._elapsed,
            before=self._before,
            after=self._after,
            result_data=result_data or {},
            timestamp=time.time(),
        )
        self.parent.add_record(op_record)
        return op_record
