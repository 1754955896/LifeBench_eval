"""Plot per-operation deltas from tracker_records.json.

Reads `<data-dir>/tracker_records.json` (one record per backend operation,
currently all `add` in this dataset) and draws 4 panels sharing a single
x-axis (wall-clock hours from the first record's timestamp):

  1. elapsed_seconds          -- latency of each operation (s)
  2. llm_total_tokens_delta   -- LLM tokens consumed by the operation
  3. memory_delta_mb          -- RSS change caused by the operation
  4. storage_delta_mb         -- on-disk storage change caused by the operation

Each panel shows per-op scatter + rolling-mean line + dataset mean reference.

Outputs (in <data-dir>/pic/):
  - operations_overview.png
  - operations_summary.txt

Usage:
    python -m src.utils.plot_operations --data-dir results/lifebench-hindsight/tracker
    python -m src.utils.plot_operations --data-dir <path> --rolling 50
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Panel:
    label: str
    unit: str
    values: np.ndarray
    color: str
    log: bool = False
    drop_zeros: bool = False  # treat 0 as "no event" — exclude from stats/plot


def _load_records(data_dir: Path) -> list[dict]:
    path = data_dir / "tracker_records.json"
    if not path.is_file():
        raise SystemExit(f"missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"expected list in {path}, got {type(data).__name__}")
    return data


def _fmt(v: float, unit: str) -> str:
    if v is None or v != v:
        return "n/a"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M {unit}"
    if abs(v) >= 1e3:
        return f"{v/1e3:.2f}k {unit}"
    return f"{v:.2f} {unit}"


def _format_hours(x: float, _pos: int) -> str:
    return f"{x/3600:.1f}h"


def _rolling_mean_nan(vals: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean ignoring NaNs (so zero-token events don't drag it down)."""
    if w <= 1:
        return vals.astype(float)
    out = np.full_like(vals, np.nan, dtype=float)
    arr = vals.astype(float)
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        lo = max(0, i - w // 2)
        hi = min(len(arr), i + w - w // 2)
        win = arr[lo:hi]
        win = win[~np.isnan(win)]
        if win.size:
            out[i] = win.mean()
    return out


def _stats(name: str, vals: np.ndarray, unit: str) -> str:
    arr = vals[np.isfinite(vals)]
    if arr.size == 0:
        return f"{name:24s}  (no data)"
    return (
        f"{name:24s}  n={arr.size:>6d}  "
        f"min={_fmt(arr.min(), unit):>14s}  "
        f"mean={_fmt(arr.mean(), unit):>14s}  "
        f"p50={_fmt(np.percentile(arr, 50), unit):>14s}  "
        f"p95={_fmt(np.percentile(arr, 95), unit):>14s}  "
        f"max={_fmt(arr.max(), unit):>14s}  "
        f"sum={_fmt(arr.sum(), unit):>14s}"
    )


def _build_panels(records: list[dict]) -> tuple[np.ndarray, list[Panel], dict]:
    ts = np.array([r["timestamp"] for r in records], dtype=float)
    x = ts - ts[0]  # seconds since first record

    def _get(field: str) -> np.ndarray:
        if "." in field:
            top, sub = field.split(".", 1)
            return np.array([r[top].get(sub) for r in records], dtype=float)
        return np.array([r.get(field) for r in records], dtype=float)

    panels = [
        Panel("elapsed_seconds", "latency per op (s)",
              _get("elapsed_seconds"), "#1f77b4"),
        Panel("llm_total_tokens_delta", "LLM tokens per op (>=1 only)",
              _get("extra.llm_total_tokens_delta"), "#d62728",
              log=True, drop_zeros=True),
        Panel("memory_delta_mb", "RSS delta per op (MB)",
              _get("memory_delta_mb"), "#2ca02c"),
        Panel("storage_delta_mb", "storage delta per op (MB)",
              _get("storage_delta_mb"), "#9467bd"),
    ]
    meta = {
        "operation_counts": dict(Counter(r["operation"] for r in records)),
        "n_records": len(records),
        "wall_clock_hours": float((ts[-1] - ts[0]) / 3600),
    }
    return x, panels, meta


def _plot(x: np.ndarray, panels: list[Panel], out_png: Path,
          title: str, rolling: int) -> None:
    n = len(panels)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(11, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, panels):
        finite_mask = np.isfinite(p.values)
        if p.drop_zeros:
            plot_mask = finite_mask & (p.values > 0)
        else:
            plot_mask = finite_mask
        ax.scatter(x[plot_mask], p.values[plot_mask],
                   s=5, alpha=0.25, color=p.color, rasterized=True,
                   label=f"{p.label} (n={plot_mask.sum()})")
        if rolling > 1 and plot_mask.sum() >= rolling:
            plot_vals = p.values.copy()
            # zeros would drag the rolling mean down; only average over events
            plot_vals[~plot_mask] = np.nan
            smooth = _rolling_mean_nan(plot_vals, rolling)
            ax.plot(x, smooth, color=p.color, linewidth=1.4,
                    label=f"rolling mean (w={rolling})")
        finite = p.values[plot_mask]
        if finite.size:
            ax.axhline(finite.mean(), color=p.color, linestyle=":",
                       linewidth=0.8, alpha=0.5,
                       label=f"mean = {_fmt(finite.mean(), p.label.split('(')[-1].rstrip(')') if '(' in p.label else '')}")
            if p.log:
                ax.set_yscale("log")
                positive = finite[finite > 0]
                if positive.size:
                    ax.set_ylim(positive.min() * 0.5, finite.max() * 2)
            else:
                pad = (finite.max() - finite.min()) * 0.08 or 1.0
                ax.set_ylim(finite.min() - pad, finite.max() + pad)
        ax.set_title(p.label, loc="left", fontsize=10, color=p.color, pad=4)
        ax.grid(True, which="both" if p.log else "major",
                linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=3)

    axes[-1].set_xlabel("wall-clock time (hours since first record)")
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(_format_hours))
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def _write_summary(panels: list[Panel], out_txt: Path, meta: dict) -> None:
    unit_lookup = {
        "elapsed_seconds": "s",
        "llm_total_tokens_delta": "tokens",
        "memory_delta_mb": "MB",
        "storage_delta_mb": "MB",
    }
    lines = [
        f"records: {meta['n_records']}",
        f"operation counts: {meta['operation_counts']}",
        f"wall-clock span: {meta['wall_clock_hours']:.2f}h",
        "",
        "=== per-operation deltas ===",
    ]
    for p in panels:
        if p.drop_zeros:
            mask = np.isfinite(p.values) & (p.values > 0)
            total_zero = int((np.isfinite(p.values) & (p.values == 0)).sum())
            suffix = f"  (excluded {total_zero} zero-token ops from stats)"
        else:
            mask = np.isfinite(p.values)
            suffix = ""
        lines.append(_stats(p.label, p.values[mask], unit_lookup.get(p.label, "")) + suffix)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_txt}")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="directory containing tracker_records.json")
    p.add_argument("--out-png", type=Path, default=None)
    p.add_argument("--out-txt", type=Path, default=None)
    p.add_argument("--rolling", type=int, default=20,
                   help="rolling-mean window for the smooth line (default 20)")
    args = p.parse_args()

    data_dir: Path = args.data_dir
    print(f"loading from {data_dir}")
    records = _load_records(data_dir)
    print(f"  records: {len(records)}")

    x, panels, meta = _build_panels(records)

    out_png = args.out_png or (data_dir / "pic" / "operations_overview.png")
    out_txt = args.out_txt or (data_dir / "pic" / "operations_summary.txt")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    _plot(x, panels, out_png, title=f"per-operation deltas — {meta['operation_counts']}",
          rolling=args.rolling)
    _write_summary(panels, out_txt, meta)


if __name__ == "__main__":
    main()
