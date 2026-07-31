"""Plot per-operation latency from LifeBench_eval results, grouped by date.

Inputs (all under <data-dir>):
  - add_latency.json     list[{date, session_id, latency_seconds, ...}]
  - search_latency.json  list[{question_id, conversation_id, latency_seconds}]
  - search_results.json  list[{question_id, query, ...}]   (only to derive date
                          for search records, since search_latency.json has no
                          `date` field; we parse "（提问时间：YYYY-MM-DD）"
                          from `query`)

Dedup:
  - add    : keep first per (date, session_id)
  - search : keep first per question_id, then join date from search_results

Outputs (in <data-dir>):
  - latency_overview.png   3 stacked subplots, x-axis = date
    1) per-op latency scatter (add + search) with daily mean lines
    2) cumulative latency over time (add + search, separate)
    3) cumulative average latency over time (add + search, separate)
  - latency_summary.txt    per-operation min/mean/p95 plus per-day summary

Usage:
    python -m src.utils.plot_latency --data-dir results/lifebench-hindsight
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


QUERY_DATE_RE = re.compile(r"（提问时间[：:](\d{4}-\d{2}-\d{2})）")
ADD_COLOR = "#1f77b4"
SEARCH_COLOR = "#d62728"
ADD_COLOR_SOFT = "#aec7e8"
SEARCH_COLOR_SOFT = "#ff9896"


def _load_json(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"expected list in {path}, got {type(data).__name__}")
    return data


def _dedup_add(records: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r in records:
        key = (r.get("date"), r.get("session_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _dedup_search(records: list[dict], date_by_qid: dict[str, str]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    missing = 0
    for r in records:
        qid = r.get("question_id")
        if qid in seen:
            continue
        seen.add(qid)
        if qid not in date_by_qid:
            missing += 1
            continue
        out.append({**r, "date": date_by_qid[qid]})
    if missing:
        print(f"  search records without resolvable date: {missing} (skipped)")
    return out


def _build_date_index(search_results: list[dict]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for r in search_results:
        m = QUERY_DATE_RE.search(r.get("query", ""))
        if m and r.get("question_id"):
            idx[r["question_id"]] = m.group(1)
    return idx


def _daily_stats(records: list[dict]) -> dict[str, list[float]]:
    by_date: dict[str, list[float]] = defaultdict(list)
    for r in records:
        lat = r.get("latency_seconds")
        d = r.get("date")
        if lat is None or d is None:
            continue
        by_date[d].append(float(lat))
    return by_date


def _sort_by_date(dates, values):
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    return [dates[i] for i in order], [values[i] for i in order]


def _cumavg(dates, values: list[float]) -> tuple[list, list[float]]:
    s_dates, s_vals = _sort_by_date(dates, values)
    cs = np.cumsum(s_vals)
    cn = np.arange(1, len(s_vals) + 1)
    return s_dates, (cs / cn).tolist()


def _cumsum(dates, values: list[float]) -> tuple[list, list[float]]:
    s_dates, s_vals = _sort_by_date(dates, values)
    return s_dates, np.cumsum(s_vals).tolist()


def _fmt_seconds(v: float) -> str:
    if v is None or v != v:
        return "n/a"
    if v >= 3600:
        return f"{v/3600:.2f}h"
    if v >= 60:
        return f"{v/60:.1f}min"
    if v >= 1:
        return f"{v:.2f}s"
    return f"{v*1000:.0f}ms"


def _stats_line(name: str, vals: list[float]) -> str:
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{name:20s}  (no data)"
    return (
        f"{name:20s}  n={arr.size:>7d}  "
        f"min={_fmt_seconds(arr.min()):>10s}  "
        f"mean={_fmt_seconds(arr.mean()):>10s}  "
        f"p50={_fmt_seconds(np.percentile(arr, 50)):>10s}  "
        f"p95={_fmt_seconds(np.percentile(arr, 95)):>10s}  "
        f"max={_fmt_seconds(arr.max()):>10s}  "
        f"total={_fmt_seconds(arr.sum()):>10s}"
    )


def _plot(add_recs: list[dict], search_recs: list[dict], out_png: Path) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True)

    add_daily = _daily_stats(add_recs)
    search_daily = _daily_stats(search_recs)
    all_dates = sorted(set(add_daily) | set(search_daily))

    # ---- subplot 1: scatter of per-op latency + daily mean lines ----
    ax = axes[0]
    add_dates = [_parse_date(r["date"]) for r in add_recs
                 if r.get("date") and r.get("latency_seconds") is not None]
    add_vals = [float(r["latency_seconds"]) for r in add_recs
                if r.get("date") and r.get("latency_seconds") is not None]
    search_dates = [_parse_date(r["date"]) for r in search_recs
                    if r.get("date") and r.get("latency_seconds") is not None]
    search_vals = [float(r["latency_seconds"]) for r in search_recs
                   if r.get("date") and r.get("latency_seconds") is not None]

    if add_vals:
        ax.scatter(add_dates, add_vals, s=4, alpha=0.25, color=ADD_COLOR_SOFT,
                   label=f"add (n={len(add_vals)})", rasterized=True)
    if search_vals:
        ax.scatter(search_dates, search_vals, s=4, alpha=0.25, color=SEARCH_COLOR_SOFT,
                   label=f"search (n={len(search_vals)})", rasterized=True)

    if add_daily:
        xs = [_parse_date(d) for d in sorted(add_daily)]
        ys = [np.mean(add_daily[d]) for d in sorted(add_daily)]
        ax.plot(xs, ys, color=ADD_COLOR, linewidth=1.4, marker="o", markersize=3,
                label="add daily mean")
    if search_daily:
        xs = [_parse_date(d) for d in sorted(search_daily)]
        ys = [np.mean(search_daily[d]) for d in sorted(search_daily)]
        ax.plot(xs, ys, color=SEARCH_COLOR, linewidth=1.4, marker="o", markersize=3,
                label="search daily mean")

    ax.set_yscale("log")
    ax.set_ylabel("latency per op (s, log)")
    ax.set_title("per-operation latency by date (with daily means)",
                 loc="left", fontsize=10, pad=4)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=2)

    # ---- subplot 2: cumulative latency ----
    ax = axes[1]
    if add_vals:
        d_cum, v_cum = _cumsum(add_dates, add_vals)
        ax.plot(d_cum, v_cum, color=ADD_COLOR, linewidth=1.6,
                label=f"add cumulative ({_fmt_seconds(v_cum[-1])})")
    if search_vals:
        d_cum, v_cum = _cumsum(search_dates, search_vals)
        ax.plot(d_cum, v_cum, color=SEARCH_COLOR, linewidth=1.6,
                label=f"search cumulative ({_fmt_seconds(v_cum[-1])})")
    if add_vals and search_vals:
        d_all = add_dates + search_dates
        v_all = add_vals + search_vals
        d_cum, v_cum = _cumsum(d_all, v_all)
        ax.plot(d_cum, v_cum, color="black", linewidth=1.0, linestyle="--",
                alpha=0.6, label=f"total cumulative ({_fmt_seconds(v_cum[-1])})")
    ax.set_ylabel("cumulative latency (s)")
    ax.set_title("cumulative latency over time",
                 loc="left", fontsize=10, pad=4)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ---- subplot 3: cumulative average latency ----
    ax = axes[2]
    if add_vals:
        d_cum, v_cum = _cumavg(add_dates, add_vals)
        ax.plot(d_cum, v_cum, color=ADD_COLOR, linewidth=1.6,
                label=f"add running mean (final {_fmt_seconds(v_cum[-1])})")
    if search_vals:
        d_cum, v_cum = _cumavg(search_dates, search_vals)
        ax.plot(d_cum, v_cum, color=SEARCH_COLOR, linewidth=1.6,
                label=f"search running mean (final {_fmt_seconds(v_cum[-1])})")
    ax.set_ylabel("running mean latency (s)")
    ax.set_xlabel("date")
    ax.set_title("cumulative average latency over time (cumulative_sum / cumulative_n)",
                 loc="left", fontsize=10, pad=4)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")

    fig.suptitle("add / search latency by date (deduplicated)", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def _write_summary(add_recs: list[dict], search_recs: list[dict],
                   out_txt: Path) -> None:
    add_vals = [float(r["latency_seconds"]) for r in add_recs if r.get("latency_seconds") is not None]
    search_vals = [float(r["latency_seconds"]) for r in search_recs if r.get("latency_seconds") is not None]
    add_daily = _daily_stats(add_recs)
    search_daily = _daily_stats(search_recs)
    all_days = set(add_daily) | set(search_daily)

    lines = [
        "=== per-operation latency (after dedup) ===",
        _stats_line("add", add_vals),
        _stats_line("search", search_vals),
        "",
        "=== daily aggregates ===",
        f"{'date':12s}  {'add_n':>7s}  {'add_mean':>10s}  {'search_n':>9s}  {'search_mean':>12s}",
    ]
    for d in sorted(all_days):
        a = add_daily.get(d, [])
        s = search_daily.get(d, [])
        lines.append(
            f"{d:12s}  {len(a):>7d}  {(_fmt_seconds(np.mean(a)) if a else '-'):>10s}  "
            f"{len(s):>9d}  {(_fmt_seconds(np.mean(s)) if s else '-'):>12s}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_txt}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="directory containing add_latency.json, search_latency.json, search_results.json")
    p.add_argument("--out-png", type=Path, default=None)
    p.add_argument("--out-txt", type=Path, default=None)
    args = p.parse_args()

    data_dir: Path = args.data_dir
    print(f"loading from {data_dir}")

    add_raw = _load_json(data_dir / "add_latency.json")
    search_raw = _load_json(data_dir / "search_latency.json")
    search_results = _load_json(data_dir / "search_results.json")
    print(f"  raw: add={len(add_raw)}  search={len(search_raw)}  search_results={len(search_results)}")

    date_by_qid = _build_date_index(search_results)
    print(f"  dates parsed from search_results.query: {len(date_by_qid)}")

    add_recs = _dedup_add(add_raw)
    search_recs = _dedup_search(search_raw, date_by_qid)
    print(f"  dedup: add={len(add_recs)}  search={len(search_recs)}")

    out_png = args.out_png or (data_dir / "pic" / "latency_overview.png")
    out_txt = args.out_txt or (data_dir / "pic" / "latency_summary.txt")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _plot(add_recs, search_recs, out_png)
    _write_summary(add_recs, search_recs, out_txt)


if __name__ == "__main__":
    main()
