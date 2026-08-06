"""
Global resource monitor - background thread for periodic sampling.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json

from src.trackers.system_trackers.base import SystemTracker

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    """A single point in the resource timeline.

    entry_type: "sample" (periodic snapshot), "op_start", or "op_end".
    Non-sample entries carry no resource data — they serve as time anchors
    so per-operation internal behaviour can be sliced from the timeline.
    """
    timestamp: float  # Unix timestamp
    elapsed_seconds: float  # Seconds since monitoring started
    entry_type: str = "sample"
    operation: str = ""  # "add" / "search", meaningful only for op_start/op_end
    storage_mb: float = 0.0
    memory_rss_mb: float = 0.0
    cpu_percent: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "t": round(self.timestamp, 2),
            "elapsed": round(self.elapsed_seconds, 2),
            "storage_mb": round(self.storage_mb, 3),
            "memory_rss_mb": round(self.memory_rss_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "extra": self.extra,
        }
        if self.entry_type != "sample":
            d["entry_type"] = self.entry_type
            d["operation"] = self.operation
        return d


class GlobalMonitor:
    """
    Background resource monitor that periodically samples system resources.

    Runs in a separate daemon thread, collecting resource snapshots at regular
    intervals. Useful for observing trends, detecting memory leaks, and
    capturing peak resource usage.

    Usage:
        monitor = GlobalMonitor(tracker=ProcessTracker(), interval=5.0, output_dir=Path("results"))
        monitor.start()
        # ... run pipeline ...
        monitor.stop()
        report = monitor.get_report()
    """

    def __init__(
        self,
        tracker: SystemTracker,
        interval: float = 5.0,
        output_dir: Optional[Path] = None,
        save_interval: float = 1.0,
    ):
        """
        Args:
            tracker: The tracker to use for snapshots.
            interval: Sampling interval in seconds (default: 5.0).
            output_dir: Directory to save timeline JSON (optional).
            save_interval: Minimum seconds between file writes (default: 1.0).
        """
        self.tracker = tracker
        self.interval = interval
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_interval = save_interval

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Load existing timeline for resume, and compute the offset so new
        # samples continue with monotonically increasing elapsed_seconds.
        existing_timeline, previous_duration = self._load_existing_timeline()
        self._timeline: List[TimelineEntry] = existing_timeline
        self._previous_duration: float = previous_duration
        self._start_time: float = 0.0
        self._stop_event = threading.Event()
        self._last_save_time: float = 0.0

    def _load_existing_timeline(self):
        """Load existing timeline from disk for resume.

        Returns (entries, duration_seconds) so new samples continue with
        monotonically-increasing elapsed_seconds.
        """
        if not self.output_dir:
            return [], 0.0
        filepath = self.output_dir / "tracker" / "global_resource_timeline.json"
        if not filepath.exists():
            return [], 0.0
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", [])
            entries = []
            for s in samples:
                entries.append(TimelineEntry(
                    timestamp=s.get("t", 0),
                    elapsed_seconds=s.get("elapsed", 0),
                    entry_type=s.get("entry_type", "sample"),
                    operation=s.get("operation", ""),
                    storage_mb=s.get("storage_mb", 0.0),
                    memory_rss_mb=s.get("memory_rss_mb", 0.0),
                    cpu_percent=s.get("cpu_percent", 0.0),
                    extra=s.get("extra", {}),
                ))
            duration = data.get("summary", {}).get("duration_seconds", 0.0)
            if entries and not duration:
                duration = entries[-1].elapsed_seconds
            return entries, duration
        except Exception:
            return [], 0.0

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None

        # Final save on stop
        self._save_timeline()

    def _sample_loop(self) -> None:
        """Main sampling loop running in background thread."""
        while self._running and not self._stop_event.is_set():
            try:
                snap = self.tracker.snapshot()
                elapsed = time.time() - self._start_time + self._previous_duration

                entry = TimelineEntry(
                    timestamp=time.time(),
                    elapsed_seconds=elapsed,
                    storage_mb=snap.storage_mb,
                    memory_rss_mb=snap.memory_rss_mb,
                    cpu_percent=snap.cpu_percent,
                    extra=snap.extra,
                )

                with self._lock:
                    self._timeline.append(entry)
                    # Real-time write: save after each sample if enough time has passed
                    now = time.time()
                    if self.output_dir and (now - self._last_save_time) >= self.save_interval:
                        self._last_save_time = now
                        # Release lock before saving to avoid blocking writes
                        timeline_copy = list(self._timeline)
                    else:
                        timeline_copy = None

                if timeline_copy is not None:
                    self._write_timeline(timeline_copy)

            except Exception:
                logger.warning("GlobalMonitor sample failed", exc_info=True)

            # Wait for next interval or stop event
            self._stop_event.wait(timeout=self.interval)

    def _write_timeline(self, entries: List[TimelineEntry]) -> None:
        """Write timeline entries to JSON file (called from sampling thread)."""
        if not self.output_dir or not entries:
            return

        tracker_dir = self.output_dir / "tracker"
        tracker_dir.mkdir(parents=True, exist_ok=True)
        filepath = tracker_dir / "global_resource_timeline.json"

        # Compute summary from sample entries only (skip op_start/op_end markers)
        sample_entries = [e for e in entries if e.entry_type == "sample"]
        memory_values = [e.memory_rss_mb for e in sample_entries]
        storage_values = [e.storage_mb for e in sample_entries]
        cpu_values = [e.cpu_percent for e in sample_entries]

        summary = {
            "total_samples": len(entries),
            "duration_seconds": round(entries[-1].elapsed_seconds, 2) if entries else 0,
            "peak_memory_mb": round(max(memory_values), 2) if memory_values else 0,
            "avg_memory_mb": round(sum(memory_values) / len(memory_values), 2) if memory_values else 0,
            "peak_cpu_percent": round(max(cpu_values), 2) if cpu_values else 0,
            "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else 0,
            "final_storage_mb": round(storage_values[-1], 3) if storage_values else 0,
            "storage_delta_mb": round(storage_values[-1] - storage_values[0], 3) if len(storage_values) > 1 else 0,
        }

        data = {
            "system": self.tracker.system_name,
            "interval_seconds": self.interval,
            "samples": [entry.to_dict() for entry in entries],
            "summary": summary,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_timeline(self) -> None:
        """Save current timeline to JSON file (called from stop())."""
        with self._lock:
            if not self.output_dir or not self._timeline:
                return
            entries = list(self._timeline)
        self._write_timeline(entries)

    def signal(self, operation: str, marker: str) -> None:
        """Insert an operation marker (op_start / op_end) into the timeline.

        Called by PerOpTracker when an operation begins or ends.
        The marker carries no resource data — it serves as a time anchor
        so downstream consumers can slice per-operation intervals from
        the surrounding periodic samples.
        """
        elapsed = (time.time() - self._start_time + self._previous_duration) if self._start_time else self._previous_duration
        entry = TimelineEntry(
            timestamp=time.time(),
            elapsed_seconds=elapsed,
            entry_type=marker,
            operation=operation,
        )
        with self._lock:
            self._timeline.append(entry)

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics from timeline."""
        if not self._timeline:
            return {}

        sample_entries = [e for e in self._timeline if e.entry_type == "sample"]
        memory_values = [e.memory_rss_mb for e in sample_entries]
        storage_values = [e.storage_mb for e in sample_entries]
        cpu_values = [e.cpu_percent for e in sample_entries]

        return {
            "total_samples": len(self._timeline),
            "duration_seconds": self._timeline[-1].elapsed_seconds if self._timeline else 0,
            "peak_memory_mb": round(max(memory_values), 2),
            "avg_memory_mb": round(sum(memory_values) / len(memory_values), 2),
            "peak_cpu_percent": round(max(cpu_values), 2),
            "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 2),
            "final_storage_mb": round(storage_values[-1], 3) if storage_values else 0,
            "storage_delta_mb": round(storage_values[-1] - storage_values[0], 3) if len(storage_values) > 1 else 0,
        }

    def get_timeline(self) -> List[TimelineEntry]:
        """Get a copy of the current timeline."""
        with self._lock:
            return list(self._timeline)

    def get_report(self) -> Dict[str, Any]:
        """Get a full report with timeline and summary."""
        timeline = self.get_timeline()
        return {
            "system": self.tracker.system_name,
            "interval_seconds": self.interval,
            "timeline": [e.to_dict() for e in timeline],
            "summary": self._compute_summary(),
        }

    @property
    def is_running(self) -> bool:
        """Check if monitoring is active."""
        return self._running
