#!/usr/bin/env python3
"""Plot two-system tracker comparison (EverMemOS vs MemosCloud) on one figure.

Generates 2 figures (no per-op-sampled figure 3):
  1. Global timed metrics overlaid: storage, memory RSS, CPU, llm tokens
  2. Per-operation metrics overlaid: add/search elapsed (log), peak RSS, avg CPU

Usage:
    python -m src.utils.plot_tracker_compare \
        results/locomo-evermemos-tracker results/locomo-memos_cloud-tracker \
        --labels EverMemOS MemosCloud

    # Custom output dir:
    python -m src.utils.plot_tracker_compare <dir1> <dir2> --output results/plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

COLORS = ["#4CAF50", "#2196F3"]  # evermemos green, memos blue


# ── data loading (adapted from plot_tracker.py) ──────────────────────────────


def load_data(result_dir: Path) -> Dict:
    """Load timeline + records, compute per-op metrics (same logic as plot_tracker)."""
    tracker_dir = result_dir / "tracker"
    with open(tracker_dir / "global_resource_timeline.json", encoding="utf-8") as f:
        timeline = json.load(f)
    with open(tracker_dir / "tracker_records.json", encoding="utf-8") as f:
        records = json.load(f)

    all_samples = timeline["samples"]
    samples = [s for s in all_samples if not s.get("entry_type")]

    # Forward-fill anomalous zero drops (same as plot_tracker)
    _last_sto = _last_rss = _last_llm = 0
    for _s in samples:
        if _s["storage_mb"] > 0:
            _last_sto = _s["storage_mb"]
        elif _last_sto > 0:
            _s["storage_mb"] = _last_sto
        if _s["memory_rss_mb"] > 0:
            _last_rss = _s["memory_rss_mb"]
        elif _last_rss > 0:
            _s["memory_rss_mb"] = _last_rss
        _tok = _s.get("extra", {}).get("llm_total_tokens", 0)
        if _tok > 0:
            _last_llm = _tok
        elif _last_llm > 0:
            _s.setdefault("extra", {})["llm_total_tokens"] = _last_llm

    # Time series (relative minutes)
    ts = [s["t"] for s in samples]
    t0 = ts[0]
    elapsed_min = [(t - t0) / 60.0 for t in ts]
    storage = [s["storage_mb"] for s in samples]
    rss = [s["memory_rss_mb"] for s in samples]
    cpu = [s["cpu_percent"] for s in samples]
    llm_total = [s.get("extra", {}).get("llm_total_tokens", 0) for s in samples]

    # op windows from sentinel markers (same as plot_tracker)
    op_windows: List = []
    current_op = -1
    current_t_start = None
    for s in all_samples:
        et = s.get("entry_type", "")
        if et == "op_start":
            current_op += 1
            current_t_start = s["t"]
        if et == "op_end" and current_t_start is not None:
            while len(op_windows) <= current_op:
                op_windows.append(None)
            op_windows[current_op] = (current_t_start, s["t"])
    total_ops = current_op + 1
    while len(op_windows) < total_ops:
        op_windows.append(None)

    # Per-op metrics
    op_peak_rss = {}
    op_avg_cpu = {}
    for op_idx in range(len(records)):
        if op_idx >= len(op_windows) or op_windows[op_idx] is None:
            op_peak_rss[op_idx] = 0
            op_avg_cpu[op_idx] = 0
            continue
        t_start, t_end = op_windows[op_idx]
        window_samples = [s for s in all_samples
                          if not s.get("entry_type") and t_start <= s["t"] <= t_end]
        if len(window_samples) >= 2:
            op_peak_rss[op_idx] = max(s["memory_rss_mb"] for s in window_samples)
            op_avg_cpu[op_idx] = sum(s["cpu_percent"] for s in window_samples) / len(window_samples)
        else:
            op_peak_rss[op_idx] = 0
            op_avg_cpu[op_idx] = 0

    return {
        "name": result_dir.name,
        "elapsed_min": elapsed_min,
        "storage": storage,
        "rss": rss,
        "cpu": cpu,
        "llm_total": llm_total,
        "records": records,
        "op_peak_rss": op_peak_rss,
        "op_avg_cpu": op_avg_cpu,
    }


def _op_series(data: Dict, operation: str, key: str) -> List:
    """Extract per-op values: 'elapsed' from records, others from op dicts."""
    out = []
    for i, r in enumerate(data["records"]):
        if r["operation"] != operation:
            continue
        if key == "elapsed":
            out.append(r["elapsed_seconds"])
        elif key == "peak_rss":
            out.append(data["op_peak_rss"].get(i, 0))
        elif key == "avg_cpu":
            out.append(data["op_avg_cpu"].get(i, 0))
    return out


# ── figure 1: global timed metrics overlaid ─────────────────────────────────

def fig1_global(datasets: List[Dict], out: Path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    metrics = [
        ("storage", "Storage (MB)", "#2196F3"),
        ("rss", "Memory RSS (MB)", "#4CAF50"),
        ("cpu", "CPU (%)", "#FF9800"),
        ("llm_total", "llm_total_tokens", "#9C27B0"),
    ]
    for ax, (key, ylabel, color) in zip(axes, metrics):
        for d, c in zip(datasets, COLORS):
            vals = d[key]
            if key == "storage":
                vals = [v - vals[0] for v in vals]  # baseline to 0
            ax.plot(d["elapsed_min"], vals, linewidth=0.9, color=c, alpha=0.85,
                    label=d["name"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Elapsed (min)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        if key == "llm_total":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
    fig.suptitle("Global Timed Metrics — %s vs %s" % (datasets[0]["name"], datasets[1]["name"]),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "01_global_compare.png", dpi=150)
    plt.close(fig)
    print("  -> saved 01_global_compare.png")


# ── figure 2: per-operation metrics overlaid ────────────────────────────────

def fig2_perop(datasets: List[Dict], out: Path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # add elapsed (log)
    ax = axes[0]
    for d, c in zip(datasets, COLORS):
        vals = _op_series(d, "add", "elapsed")
        ax.plot(range(1, len(vals) + 1), vals, linewidth=1.0, color=c, alpha=0.85,
                label="%s (n=%d)" % (d["name"], len(vals)))
    ax.set_yscale("log")
    ax.set_ylabel("add elapsed (s, log)")
    ax.set_title("Add Elapsed Time per Operation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # search elapsed (log)
    ax = axes[1]
    for d, c in zip(datasets, COLORS):
        vals = _op_series(d, "search", "elapsed")
        ax.plot(range(1, len(vals) + 1), vals, linewidth=1.0, color=c, alpha=0.85,
                label="%s (n=%d)" % (d["name"], len(vals)))
    ax.set_yscale("log")
    ax.set_ylabel("search elapsed (s, log)")
    ax.set_title("Search Elapsed Time per Operation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # add peak RSS
    ax = axes[2]
    for d, c in zip(datasets, COLORS):
        vals = _op_series(d, "add", "peak_rss")
        ax.plot(range(1, len(vals) + 1), vals, linewidth=1.0, color=c, alpha=0.85,
                label=d["name"])
    ax.set_ylabel("add peak RSS (MB)")
    ax.set_title("Add Operations: Peak Memory (RSS)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # add avg CPU
    ax = axes[3]
    for d, c in zip(datasets, COLORS):
        vals = _op_series(d, "add", "avg_cpu")
        ax.plot(range(1, len(vals) + 1), vals, linewidth=1.0, color=c, alpha=0.85,
                label=d["name"])
    ax.set_ylabel("add avg CPU (%)")
    ax.set_xlabel("Operation #")
    ax.set_title("Add Operations: Average CPU Usage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Per-Operation Metrics — %s vs %s" % (datasets[0]["name"], datasets[1]["name"]),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "02_perop_compare.png", dpi=150)
    plt.close(fig)
    print("  -> saved 02_perop_compare.png")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Two-system tracker comparison plots")
    parser.add_argument("dirs", nargs=2, help="Two result directories")
    parser.add_argument("--labels", nargs=2, default=None,
                        help="System labels (default: directory names)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <dir1>/../plots_compare)")
    args = parser.parse_args()

    labels = args.labels or [Path(d).name for d in args.dirs]
    out = Path(args.output) if args.output else (
        Path(args.dirs[0]).resolve().parent / "plots_compare"
    )
    out.mkdir(parents=True, exist_ok=True)

    datasets = []
    for d, label in zip(args.dirs, labels):
        data = load_data(Path(d))
        data["name"] = label
        datasets.append(data)
        n_add = sum(1 for r in data["records"] if r["operation"] == "add")
        n_search = sum(1 for r in data["records"] if r["operation"] == "search")
        print("  %s: %d samples, %d adds, %d searches" % (label, len(data["elapsed_min"]), n_add, n_search))

    print(f"\nGenerating comparison plots -> {out}")
    fig1_global(datasets, out)
    fig2_perop(datasets, out)
    print(f"\nDone. Plots saved to: {out}")


if __name__ == "__main__":
    main()
