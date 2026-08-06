"""
Per-operation tracker (timing + metadata only).

Resource data (CPU/memory/storage) is captured by GlobalMonitor's
periodic timeline. PerOpTracker records operation boundaries and
elapsed time, and emits signals so GlobalMonitor can insert
op_start / op_end markers for downstream slicing.
"""
import json
import time
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

from src.trackers.system_trackers.base import OpRecord

if TYPE_CHECKING:
    from src.trackers.global_monitor import GlobalMonitor


class PerOpTracker:
    """
    Per-operation tracker — records timing and metadata for each add/search.

    Resource data (CPU, memory, storage) is measured by GlobalMonitor's
    background sampling and correlated via op_start/op_end markers.

    Usage:
        tracker = PerOpTracker(
            backend="hindsight",
            output_dir=Path("results"),
            global_monitor=global_monitor,
        )
        with tracker.track("add") as ctx:
            result = await adapter.add_chunks(chunks)
        record = ctx.record(result_data={"added": 10, "failed": 0})
    """

    def __init__(
        self,
        backend: str,
        output_dir: Optional[Path] = None,
        global_monitor: Optional["GlobalMonitor"] = None,
    ):
        self.backend = backend
        self.output_dir = Path(output_dir) if output_dir else None
        self.global_monitor = global_monitor
        self._records: List[OpRecord] = self._load_existing()

    def _load_existing(self) -> List[OpRecord]:
        if not self.output_dir:
            return []
        filepath = self.output_dir / "tracker" / "tracker_records.json"
        if not filepath.exists():
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = []
            for d in data:
                r = OpRecord(
                    backend=d.get("backend", self.backend),
                    operation=d.get("operation", ""),
                    elapsed_seconds=d.get("elapsed_seconds", 0),
                    result_data=d.get("result_data", {}),
                    timestamp=d.get("timestamp", 0),
                )
                records.append(r)
            return records
        except Exception:
            return []

    def track(self, operation: str) -> "_OpContext":
        return _OpContext(self.backend, operation, self)

    def add_record(self, record: OpRecord) -> None:
        self._records.append(record)
        self.save()

    def save(self) -> None:
        if not self.output_dir or not self._records:
            return

        tracker_dir = self.output_dir / "tracker"
        tracker_dir.mkdir(parents=True, exist_ok=True)
        filepath = tracker_dir / "tracker_records.json"

        data = [r.to_dict() for r in self._records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_records(self) -> List[OpRecord]:
        return list(self._records)


class _OpContext:
    """Context manager that emits op_start/op_end signals and records wall time."""

    def __init__(self, backend: str, operation: str, parent: PerOpTracker):
        self.backend = backend
        self.operation = operation
        self.parent = parent
        self._start: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self) -> "_OpContext":
        if self.parent.global_monitor:
            self.parent.global_monitor.signal(self.operation, "op_start")
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._elapsed = time.perf_counter() - self._start
        if self.parent.global_monitor:
            self.parent.global_monitor.signal(self.operation, "op_end")
        return False

    def record(self, result_data: Any = None) -> OpRecord:
        op_record = OpRecord(
            backend=self.backend,
            operation=self.operation,
            elapsed_seconds=self._elapsed,
            result_data=result_data or {},
            timestamp=time.time(),
        )
        self.parent.add_record(op_record)
        return op_record