#!/usr/bin/env python3
"""Plot three-system Locomo serial-run comparison.

Generates 3 figures from evaluation result directories:
  1. Latency comparison (log-scale grouped bars): total time + per-op add/search/answer
  2. Local load comparison (grouped bars): memory peak/avg, CPU peak/avg
  3. Accuracy vs total time scatter (log x-axis) — the cost/quality trade-off

Usage:
    python -m src.utils.plot_system_compare \
        --dirs results/locomo-memos_cloud-tracker results/locomo-memu_cloud-tracker results/locomo-evermemos-tracker \
        --labels MemosCloud MemuCloud EverMemOS

    # Custom output dir:
    python -m src.utils.plot_system_compare --dirs ... --labels ... --output results/plots
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── data loading (all best-effort; missing files -> None) ───────────────────


def load_report(result_dir: Path) -> Dict[str, Optional[float]]:
    """Parse report.txt for accuracy and stage timings."""
    data: Dict[str, Optional[float]] = {
        "accuracy": None, "total_time": None,
        "add_search": None, "answer": None, "evaluate": None,
    }
    path = result_dir / "report.txt"
    if not path.exists():
        return data
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return data
    m = re.search(r"Total Time:\s*([\d,]+\.?\d*)", text)
    if m:
        data["total_time"] = float(m.group(1).replace(",", ""))
    m = re.search(r"Accuracy:\s*([\d.]+)%", text)
    if m:
        data["accuracy"] = float(m.group(1))
    m = re.search(r"Add Search:\s*([\d,]+\.?\d*)", text)
    if m:
        data["add_search"] = float(m.group(1).replace(",", ""))
    m = re.search(r"Answer:\s*([\d,]+\.?\d*)", text)
    if m:
        data["answer"] = float(m.group(1).replace(",", ""))
    m = re.search(r"Evaluate:\s*([\d,]+\.?\d*)", text)
    if m:
        data["evaluate"] = float(m.group(1).replace(",", ""))
    return data


def load_latency_avg(result_dir: Path, filename: str) -> Optional[float]:
    """Average latency_seconds from add_latency.json / search_latency.json."""
    path = result_dir / filename
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            return None
        return sum(d.get("latency_seconds", 0) for d in data) / len(data)
    except Exception:
        return None


def load_timeline_summary(result_dir: Path) -> Dict[str, Optional[float]]:
    """Load tracker/global_resource_timeline.json summary."""
    out: Dict[str, Optional[float]] = {
        "mem_peak": None, "mem_avg": None, "cpu_peak": None, "cpu_avg": None,
    }
    path = result_dir / "tracker" / "global_resource_timeline.json"
    if not path.exists():
        return out
    try:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f).get("summary", {})
        out["mem_peak"] = summary.get("peak_memory_mb")
        out["mem_avg"] = summary.get("avg_memory_mb")
        out["cpu_peak"] = summary.get("peak_cpu_percent")
        out["cpu_avg"] = summary.get("avg_cpu_percent")
    except Exception:
        pass
    return out


def collect(systems: List[tuple]) -> List[Dict]:
    """Load all metrics for each (label, dir) system."""
    rows = []
    for label, d in systems:
        rd = Path(d)
        row = {"label": label, "dir": d}
        row.update(load_report(rd))
        row["add_avg"] = load_latency_avg(rd, "add_latency.json")
        row["search_avg"] = load_latency_avg(rd, "search_latency.json")
        row.update(load_timeline_summary(rd))
        rows.append(row)
    return rows


# ── plotting helpers ─────────────────────────────────────────────────────────

BAR_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]


def _grouped_bar(ax, groups: List[Dict], metric_key: str, title: str,
                 ylabel: str, log: bool = False, fmt: str = "%.1f"):
    """One metric across systems as grouped bars. None values are skipped."""
    n = len(groups)
    x = np.arange(n)
    heights = []
    for g in groups:
        v = g.get(metric_key)
        heights.append(v if v is not None else 0.0)
    bars = ax.bar(x, heights, width=0.55, color=BAR_COLORS[:n], alpha=0.85,
                  edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.01)
    # label values above bars (skip None)
    for bar, g in zip(bars, groups):
        v = g.get(metric_key)
        if v is None:
            continue
        if log:
            label = fmt % v
        else:
            label = (fmt + ("s" if ylabel.startswith(("add", "search", "answer")) else "")) % v
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=8)


# ── figures ─────────────────────────────────────────────────────────────────

def fig1_latency(systems_rows: List[Dict], out: Path):
    """Log-scale grouped bars: total time + per-op add/search/answer."""
    metric_groups = [
        ("total_time", "Total Time (s)", "%.0f"),
        ("add_avg", "add per-op (s)", "%.1f"),
        ("search_avg", "search per-op (s)", "%.2f"),
        ("answer_avg", "answer per-op (s)", "%.2f"),
    ]
    # answer per-op from report stage timing / 233 questions
    for r in systems_rows:
        r["answer_avg"] = r["answer"] / 233 if r.get("answer") else None

    n = len(metric_groups)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for ax, (key, title, fmt) in zip(axes, metric_groups):
        _grouped_bar(ax, systems_rows, key, title, title, log=True, fmt=fmt)
    fig.suptitle("Latency Comparison (log scale)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "01_latency_compare.png", dpi=150)
    plt.close(fig)
    print("  -> saved 01_latency_compare.png")


def fig2_load(systems_rows: List[Dict], out: Path):
    """Grouped bars: memory peak/avg + CPU peak/avg."""
    n = len(systems_rows)
    x = np.arange(n)
    width = 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # memory
    mem_peak = [r["mem_peak"] or 0 for r in systems_rows]
    mem_avg = [r["mem_avg"] or 0 for r in systems_rows]
    ax1.bar(x - width / 2, mem_peak, width, color="#2196F3", alpha=0.85, label="peak")
    ax1.bar(x + width / 2, mem_avg, width, color="#90CAF9", alpha=0.85, label="avg")
    for i, r in enumerate(systems_rows):
        if r["mem_peak"]:
            ax1.text(x[i] - width / 2, r["mem_peak"], "%.1f" % r["mem_peak"],
                     ha="center", va="bottom", fontsize=8)
        if r["mem_avg"]:
            ax1.text(x[i] + width / 2, r["mem_avg"], "%.1f" % r["mem_avg"],
                     ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Memory RSS (MB)")
    ax1.set_title("Local Memory Load (test process)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # CPU
    cpu_peak = [r["cpu_peak"] or 0 for r in systems_rows]
    cpu_avg = [r["cpu_avg"] or 0 for r in systems_rows]
    ax2.bar(x - width / 2, cpu_peak, width, color="#FF9800", alpha=0.85, label="peak")
    ax2.bar(x + width / 2, cpu_avg, width, color="#FFCC80", alpha=0.85, label="avg")
    for i, r in enumerate(systems_rows):
        if r["cpu_peak"]:
            ax2.text(x[i] - width / 2, r["cpu_peak"], "%.1f" % r["cpu_peak"],
                     ha="center", va="bottom", fontsize=8)
        if r["cpu_avg"]:
            ax2.text(x[i] + width / 2, r["cpu_avg"], "%.1f" % r["cpu_avg"],
                     ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("CPU (%)")
    ax2.set_title("Local CPU Load (test process)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    ax2.set_xticks(x)
    ax2.set_xticklabels([r["label"] for r in systems_rows], fontsize=10)

    fig.suptitle("Local Load Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "02_local_load_compare.png", dpi=150)
    plt.close(fig)
    print("  -> saved 02_local_load_compare.png")


def fig3_tradeoff(systems_rows: List[Dict], out: Path):
    """Accuracy vs total time scatter (log x) — the core conclusion figure."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = BAR_COLORS[:len(systems_rows)]
    for r, c in zip(systems_rows, colors):
        tt = r.get("total_time")
        acc = r.get("accuracy")
        if tt is None or acc is None:
            continue
        minutes = tt / 60
        ax.scatter(minutes, acc, s=180, color=c, edgecolor="white", zorder=3)
        ax.annotate(r["label"], (minutes, acc),
                    textcoords="offset points", xytext=(8, 6), fontsize=10,
                    color=c, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Total Time (minutes, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Total Time — Cost/Quality Trade-off")
    ax.grid(True, alpha=0.3, which="both")

    # annotate the gap between best time and best accuracy
    xs = [r["total_time"] / 60 for r in systems_rows if r.get("total_time")]
    if len(xs) > 1:
        ax.annotate("", xy=(max(xs), 79.83), xytext=(min(xs), 79.83),
                    arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
        ax.text(np.sqrt(max(xs) * min(xs)), 79.3,
                "14x time gap for +5.6pp accuracy",
                ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(out / "03_accuracy_vs_time.png", dpi=150)
    plt.close(fig)
    print("  -> saved 03_accuracy_vs_time.png")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Three-system Locomo comparison plots")
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="Result directories (2-3 systems)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="System labels (default: directory names)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <first_dir>/../system_compare_plots)")
    args = parser.parse_args()

    labels = args.labels or [Path(d).name for d in args.dirs]
    if len(labels) != len(args.dirs):
        print("ERROR: --labels must match --dirs count")
        sys.exit(1)

    out = Path(args.output) if args.output else (
        Path(args.dirs[0]).resolve().parent / "system_compare_plots"
    )
    out.mkdir(parents=True, exist_ok=True)

    systems = list(zip(labels, args.dirs))
    rows = collect(systems)
    for r in rows:
        print("  %s: acc=%s total=%ss add=%ss search=%ss mem_peak=%sMB cpu_peak=%s%%" % (
            r["label"], r["accuracy"], r["total_time"], r["add_avg"],
            r["search_avg"], r["mem_peak"], r["cpu_peak"]))

    print(f"\nGenerating plots -> {out}")
    fig1_latency(rows, out)
    fig2_load(rows, out)
    fig3_tradeoff(rows, out)
    print(f"\nDone. Plots saved to: {out}")


if __name__ == "__main__":
    main()
