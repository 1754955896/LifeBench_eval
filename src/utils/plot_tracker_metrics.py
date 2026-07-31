"""Plot tracker metrics from a LifeBench_eval results tracker directory.

Reads `<data_dir>/global_resource_timeline.json` and draws the five requested
metrics, plus an optional tokens-per-sample rate panel:

  - pg0_storage_mb   (extra)         -- graph storage on disk
  - memory_rss_mb    (top level)     -- process resident set size
  - vms_mb           (extra)         -- process virtual memory size
  - cpu_percent      (top level)     -- process CPU usage
  - llm_total_tokens (extra)         -- cumulative LLM tokens consumed
  - llm_total_tokens_delta (extra)   -- per-sample tokens (rate proxy)

Outputs land next to the input:
  - <data_dir>/metrics_overview.png  -- stacked subplots, shared x-axis (hours)
  - <data_dir>/metrics_summary.txt   -- min/max/mean/p95 per metric

Usage:
    python -m src.utils.plot_tracker_metrics --data-dir results/lifebench-hindsight/tracker
    python -m src.utils.plot_tracker_metrics --data-dir <path> --no-delta
    python -m src.utils.plot_tracker_metrics --data-dir <path> --smoothing 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Series:
    name: str
    label: str
    unit: str
    values: np.ndarray
    color: str
    elapsed: np.ndarray = None  # type: ignore[assignment]


def _load_timeline(data_dir: Path) -> dict:
    path = data_dir / "global_resource_timeline.json"
    if not path.is_file():
        raise SystemExit(f"missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract(samples: list[dict], getter) -> np.ndarray:
    return np.array([getter(s) for s in samples], dtype=float)


def _fmt(v: float, unit: str) -> str:
    if v != v:  # NaN
        return "n/a"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M {unit}"
    if abs(v) >= 1e3:
        return f"{v/1e3:.2f}k {unit}"
    return f"{v:.2f} {unit}"


def _format_hours(x: float, _pos: int) -> str:
    return f"{x/3600:.1f}h"


def _build_series(samples: list[dict], with_delta: bool) -> list[Series]:
    elapsed = _extract(samples, lambda s: s["elapsed"])
    series: list[Series] = [
        Series("pg0_storage_mb", "pg0 storage (graph on disk)", "MB",
               _extract(samples, lambda s: s["extra"]["pg0_storage_mb"]),
               "#1f77b4"),
        Series("memory_rss_mb", "process RSS", "MB",
               _extract(samples, lambda s: s["memory_rss_mb"]),
               "#2ca02c"),
        Series("vms_mb", "process VMS", "MB",
               _extract(samples, lambda s: s["extra"]["vms_mb"]),
               "#9467bd"),
        Series("cpu_percent", "process CPU", "%",
               _extract(samples, lambda s: s["cpu_percent"]),
               "#d62728"),
        Series("llm_total_tokens", "LLM tokens (cumulative)", "tokens",
               _extract(samples, lambda s: s["extra"]["llm_total_tokens"]),
               "#ff7f0e"),
    ]
    if with_delta:
        series.append(Series(
            "llm_total_tokens_delta", "LLM tokens per sample (rate proxy)",
            "tokens/sample",
            _extract(samples, lambda s: s["extra"].get("llm_total_tokens_delta", 0.0)),
            "#8c564b",
        ))
    for s in series:
        s.elapsed = elapsed
    return series


def _plot(series_list: Sequence[Series], out_png: Path, title: str) -> None:
    n = len(series_list)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(11, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, series_list):
        ax.plot(s.elapsed, s.values, color=s.color, linewidth=1.0,
                marker="o", markersize=2.0, alpha=0.85, label=s.label)
        ax.set_ylabel(s.unit)
        ax.set_title(s.label, loc="left", fontsize=10, color=s.color, pad=4)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.axhline(np.mean(s.values), color=s.color, linestyle=":",
                   linewidth=0.8, alpha=0.5,
                   label=f"mean = {_fmt(np.mean(s.values), s.unit)}")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        finite = s.values[np.isfinite(s.values)]
        if finite.size:
            pad = (finite.max() - finite.min()) * 0.08 or 1.0
            ax.set_ylim(finite.min() - pad, finite.max() + pad)

    axes[-1].set_xlabel("elapsed time (hours)")
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(_format_hours))

    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def _write_summary(series_list: Sequence[Series], out_txt: Path,
                   duration_s: float, n_samples: int) -> None:
    lines = [
        f"samples: {n_samples}",
        f"duration: {duration_s:.1f}s ({duration_s/3600:.2f}h)",
        "",
    ]
    for s in series_list:
        finite = s.values[np.isfinite(s.values)]
        if finite.size == 0:
            lines.append(f"{s.name:28s}  (no data)")
            continue
        st = {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "p95": float(np.percentile(finite, 95)),
        }
        lines.append(
            f"{s.name:28s}  min={_fmt(st['min'], s.unit):>18s}  "
            f"max={_fmt(st['max'], s.unit):>18s}  "
            f"mean={_fmt(st['mean'], s.unit):>18s}  "
            f"p95={_fmt(st['p95'], s.unit):>18s}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_txt}")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="directory containing global_resource_timeline.json")
    p.add_argument("--out-png", type=Path, default=None,
                   help="output PNG path (default: <data-dir>/metrics_overview.png)")
    p.add_argument("--out-txt", type=Path, default=None,
                   help="output summary path (default: <data-dir>/metrics_summary.txt)")
    p.add_argument("--no-delta", action="store_true",
                   help="skip the LLM tokens-per-sample rate panel")
    p.add_argument("--smoothing", type=int, default=1,
                   help="rolling window for cpu/llm_delta; 1 = no smoothing")
    args = p.parse_args()

    data = _load_timeline(args.data_dir)
    samples = data.get("samples") or []
    if not samples:
        raise SystemExit("no samples in input file")

    series_list = _build_series(samples, with_delta=not args.no_delta)

    if args.smoothing > 1:
        kernel = np.ones(args.smoothing) / args.smoothing
        pad = args.smoothing // 2
        for s in series_list:
            if s.name in {"cpu_percent", "llm_total_tokens_delta"}:
                smoothed = np.convolve(s.values, kernel, mode="same")
                smoothed[:pad] = s.values[:pad]
                smoothed[-pad:] = s.values[-pad:]
                s.values = smoothed

    out_png = args.out_png or (args.data_dir / "pic" / "metrics_overview.png")
    out_txt = args.out_txt or (args.data_dir / "pic" / "metrics_summary.txt")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    duration = samples[-1]["elapsed"] - samples[0]["elapsed"]
    system = data.get("system", "unknown")

    _plot(series_list, out_png, title=f"Resource timeline — {system}")
    _write_summary(series_list, out_txt, duration_s=duration, n_samples=len(samples))


if __name__ == "__main__":
    main()
