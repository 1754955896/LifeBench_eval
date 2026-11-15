"""
Process-level system tracker.

Tracks process resource usage (CPU, memory) using psutil.
Also queries llm_proxy for LLM token usage.
"""
import os
from typing import Any, Dict, Optional

from src.trackers.system_trackers.base import SystemSnapshot, SystemTracker, register_tracker
from src.trackers.system_trackers.utils import query_llm_proxy


@register_tracker("default")
class DefaultTracker(SystemTracker):
    """
    Default resource tracker using psutil and llm_proxy.

    Tracks:
    - Process CPU/memory usage via psutil
    - LLM token usage via llm_proxy (if available)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        pid: Optional[int] = None,
        llm_proxy_url: Optional[str] = None,
    ):
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            raise ImportError(
                "psutil is required for DefaultTracker. "
                "Install it with: pip install psutil"
            )

        self._pid = pid
        self._process = None
        self._peak_memory_mb = 0.0
        self._llm_proxy_url = llm_proxy_url or (config.get("llm_proxy_url") if config else None) or "http://localhost:18443"
        self._last_llm_tokens: Optional[Dict[str, int]] = None

    @property
    def system_name(self) -> str:
        return "default"

    def _get_process(self):
        if self._process is None:
            pid = self._pid or os.getpid()
            self._process = self._psutil.Process(pid)
        return self._process

    def snapshot(self) -> SystemSnapshot:
        """Take a snapshot of current resource usage (process + LLM tokens)."""
        extra: Dict[str, Any] = {
            "pid": self._pid or os.getpid(),
        }

        # Process-level metrics
        try:
            proc = self._get_process()
            mem_info = proc.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            if rss_mb > self._peak_memory_mb:
                self._peak_memory_mb = rss_mb

            extra["memory_rss_mb"] = round(rss_mb, 2)
            extra["peak_memory_mb"] = round(self._peak_memory_mb, 2)
            extra["vms_mb"] = round(mem_info.vms / (1024 * 1024), 2)

            # CPU percent (interval=None returns since last call, may be 0 on first call)
            cpu_percent = proc.cpu_percent(interval=None)
            extra["cpu_percent"] = round(cpu_percent, 2)
        except Exception as e:
            extra["process_error"] = str(e)

        # LLM token metrics
        if self._llm_proxy_url:
            llm_data = query_llm_proxy(self._llm_proxy_url)
            if llm_data:
                # Compute delta from last snapshot
                if self._last_llm_tokens is not None:
                    extra["llm_prompt_tokens_delta"] = llm_data.get("prompt_tokens", 0) - self._last_llm_tokens.get("prompt_tokens", 0)
                    extra["llm_completion_tokens_delta"] = llm_data.get("completion_tokens", 0) - self._last_llm_tokens.get("completion_tokens", 0)
                    extra["llm_total_tokens_delta"] = llm_data.get("total_tokens", 0) - self._last_llm_tokens.get("total_tokens", 0)

                self._last_llm_tokens = {
                    "prompt_tokens": llm_data.get("prompt_tokens", 0),
                    "completion_tokens": llm_data.get("completion_tokens", 0),
                    "total_tokens": llm_data.get("total_tokens", 0),
                }

                extra["llm_prompt_tokens"] = llm_data.get("prompt_tokens", 0)
                extra["llm_completion_tokens"] = llm_data.get("completion_tokens", 0)
                extra["llm_total_tokens"] = llm_data.get("total_tokens", 0)
                extra["llm_request_count"] = llm_data.get("request_count", 0)

        return SystemSnapshot(
            storage_mb=0.0,
            memory_rss_mb=extra.get("memory_rss_mb", 0.0),
            cpu_percent=extra.get("cpu_percent", 0.0),
            extra=extra,
        )

    def backend_specific_stats(self) -> Dict[str, Any]:
        """Return process and LLM stats."""
        stats = {
            "pid": self._pid,
            "peak_memory_mb": round(self._peak_memory_mb, 2),
        }
        if self._llm_proxy_url:
            llm_data = query_llm_proxy(self._llm_proxy_url)
            if llm_data:
                stats.update({
                    "llm_prompt_tokens": llm_data.get("prompt_tokens", 0),
                    "llm_completion_tokens": llm_data.get("completion_tokens", 0),
                    "llm_total_tokens": llm_data.get("total_tokens", 0),
                    "llm_request_count": llm_data.get("request_count", 0),
                })
        return stats

    def reset_peak(self):
        """Reset peak memory tracking."""
        self._peak_memory_mb = 0.0

    def reset_llm_delta(self):
        """Reset LLM token delta tracking."""
        self._last_llm_tokens = None
