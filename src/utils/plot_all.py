"""One-shot plotter that runs the three tracker scripts and writes all outputs
to a single <results-dir>/pic/ folder.

Runs in order:
  1. src.utils.plot_tracker_metrics  (resource timeline from
     tracker/global_resource_timeline.json)
  2. src.utils.plot_latency          (per-op latency by date, from
     add_latency.json / search_latency.json / search_results.json)
  3. src.utils.plot_operations       (per-op deltas from
     tracker/tracker_records.json)

Each step is skipped if its required inputs are missing. Outputs all land in:
    <results-dir>/pic/
        metrics_overview.png   +   metrics_summary.txt
        latency_overview.png   +   latency_summary.txt
        operations_overview.png +  operations_summary.txt
        run_stats.txt                      <-- aggregated run-level stats

The run_stats.txt aggregates across all sources:
  - system wall-clock runtime (from global_resource_timeline)
  - total time spent in add / search phases (from add/search latency logs)
  - mean latency per add / per search
  - total cumulative LLM tokens
  - mean tokens per add (over all adds, and over LLM-calling adds only)

Usage:
    python -m src.utils.plot_all --results-dir results/lifebench-hindsight
    python -m src.utils.plot_all                                 # default path
    python -m src.utils.plot_all --only latency,operations      # subset
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent.parent.parent  # LifeBench_eval/

ALL_STEPS = ("metrics", "latency", "operations")
MODULE_NAME = {
    "metrics": "plot_tracker_metrics",
    "latency": "plot_latency",
    "operations": "plot_operations",
}

_QUERY_DATE_RE = re.compile(r"（提问时间[：:](\d{4}-\d{2}-\d{2})）")


def _fmt_time(s) -> str:
    if s is None:
        return "n/a"
    if s >= 3600:
        return f"{s/3600:.2f}h"
    if s >= 60:
        return f"{s/60:.1f}min"
    return f"{s:.2f}s"


def _fmt_count(n) -> str:
    if n is None:
        return "n/a"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.2f}k"
    return f"{n}"


def _build_steps(results_dir: Path, out_dir: Path) -> list[tuple[str, list[str]]]:
    tracker_dir = results_dir / "tracker"
    steps: list[tuple[str, list[str]]] = []

    # 1. resource timeline
    timeline = tracker_dir / "global_resource_timeline.json"
    if timeline.is_file():
        steps.append((
            "metrics",
            [
                "--data-dir", str(tracker_dir),
                "--out-png", str(out_dir / "metrics_overview.png"),
                "--out-txt", str(out_dir / "metrics_summary.txt"),
            ],
        ))
    else:
        print(f"[skip] metrics: missing {timeline}")

    # 2. latency
    needed = ["add_latency.json", "search_latency.json", "search_results.json"]
    if all((results_dir / n).is_file() for n in needed):
        steps.append((
            "latency",
            [
                "--data-dir", str(results_dir),
                "--out-png", str(out_dir / "latency_overview.png"),
                "--out-txt", str(out_dir / "latency_summary.txt"),
            ],
        ))
    else:
        missing = [n for n in needed if not (results_dir / n).is_file()]
        print(f"[skip] latency: missing {missing}")

    # 3. operations
    records = tracker_dir / "tracker_records.json"
    if records.is_file():
        steps.append((
            "operations",
            [
                "--data-dir", str(tracker_dir),
                "--out-png", str(out_dir / "operations_overview.png"),
                "--out-txt", str(out_dir / "operations_summary.txt"),
            ],
        ))
    else:
        print(f"[skip] operations: missing {records}")

    return steps


def _run(name: str, extra_args: list[str], py: str) -> None:
    module = MODULE_NAME[name]
    cmd = [py, "-m", f"src.utils.{module}", *extra_args]
    print(f"\n=== {name} ({module}) ===")
    print("  $ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(SCRIPT_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"{name} failed (returncode={result.returncode})")


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _compute_run_stats(results_dir: Path, out_txt: Path) -> None:
    """Aggregate run-level stats from all available sources and write a report.

    Pulls from:
      - tracker/global_resource_timeline.json   system runtime + cumulative tokens
      - add_latency.json                        add total time + per-add latency
      - search_latency.json + search_results.json  search total time + per-search latency
      - tracker/tracker_records.json            per-add token usage
    """
    tracker_dir = results_dir / "tracker"

    # 1) system runtime + cumulative LLM tokens from the periodic timeline
    timeline = _load_json(tracker_dir / "global_resource_timeline.json")
    runtime_s: float | None = None
    total_tokens_cum: float | None = None
    if timeline:
        runtime_s = timeline.get("summary", {}).get("duration_seconds")
        samples = timeline.get("samples") or []
        if samples:
            total_tokens_cum = samples[-1].get("extra", {}).get("llm_total_tokens")

    # 2) add total time + count, deduped by (date, session_id)
    add_raw = _load_json(results_dir / "add_latency.json")
    add_count = 0
    add_total_s = 0.0
    if add_raw:
        seen: set[tuple] = set()
        for r in add_raw:
            key = (r.get("date"), r.get("session_id"))
            lat = r.get("latency_seconds")
            if key in seen or lat is None:
                continue
            seen.add(key)
            add_total_s += float(lat)
            add_count += 1

    # 3) search total time + count, deduped by question_id (drop records
    #    we can't attach a date to, matching plot_latency.py)
    search_raw = _load_json(results_dir / "search_latency.json")
    search_results = _load_json(results_dir / "search_results.json")
    date_by_qid: dict[str, str] = {}
    if search_results:
        for r in search_results:
            m = _QUERY_DATE_RE.search(r.get("query", "") or "")
            if m and r.get("question_id"):
                date_by_qid[r["question_id"]] = m.group(1)
    search_count = 0
    search_total_s = 0.0
    if search_raw:
        seen_q: set[str] = set()
        for r in search_raw:
            qid = r.get("question_id")
            lat = r.get("latency_seconds")
            if qid in seen_q or lat is None or qid not in date_by_qid:
                continue
            seen_q.add(qid)
            search_total_s += float(lat)
            search_count += 1

    # 4) per-add token usage from tracker_records
    records = _load_json(tracker_dir / "tracker_records.json")
    add_token_total = 0.0
    add_token_count = 0
    add_token_count_nonzero = 0
    add_wallclock_s: float | None = None
    if records:
        timestamps = [r["timestamp"] for r in records
                      if isinstance(r.get("timestamp"), (int, float))]
        if timestamps:
            add_wallclock_s = float(max(timestamps) - min(timestamps))
        for r in records:
            delta = r.get("extra", {}).get("llm_total_tokens_delta")
            if delta is None:
                continue
            add_token_total += float(delta)
            add_token_count += 1
            if float(delta) > 0:
                add_token_count_nonzero += 1

    # ---- write the report ----
    def _row(label: str, value: str) -> str:
        return f"  {label:<28s} : {value}"

    lines = ["=== run statistics ===", ""]
    lines.append("[system runtime]")
    if runtime_s is not None:
        lines.append(_row("total wall-clock runtime", f"{_fmt_time(runtime_s)} ({runtime_s:.1f}s)"))
    else:
        lines.append(_row("total wall-clock runtime", "n/a (no global_resource_timeline.json)"))
    lines.append("")

    lines.append("[operation time]")
    lines.append(_row("add total time", f"{_fmt_time(add_total_s)} over {add_count} ops"))
    lines.append(_row("search total time", f"{_fmt_time(search_total_s)} over {search_count} ops"))
    lines.append(_row("add real wall-clock span",
                      f"{_fmt_time(add_wallclock_s)}" if add_wallclock_s is not None
                      else "n/a (no tracker_records.json)"))
    lines.append(_row("add mean latency",
                      _fmt_time(add_total_s / add_count) if add_count else "n/a"))
    lines.append(_row("search mean latency",
                      _fmt_time(search_total_s / search_count) if search_count else "n/a"))
    lines.append("")

    lines.append("[tokens]")
    lines.append(_row("total LLM tokens (cumulative)",
                      f"{_fmt_count(total_tokens_cum)} tokens" if total_tokens_cum is not None else "n/a"))
    lines.append(_row("add ops tracked", str(add_token_count)))
    lines.append(_row("add ops with LLM call", str(add_token_count_nonzero)))
    if add_token_count:
        lines.append(_row("add mean tokens / op (all adds)",
                          f"{_fmt_count(add_token_total / add_token_count)} tokens"))
    if add_token_count_nonzero:
        lines.append(_row("add mean tokens / LLM-call op",
                          f"{_fmt_count(add_token_total / add_token_count_nonzero)} tokens"))

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_txt}")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path,
                   default=Path("results/lifebench-hindsight"),
                   help="LifeBench_eval results root (default: %(default)s)")
    p.add_argument("--only", default=None,
                   help=f"comma-separated subset of {ALL_STEPS} (default: run all)")
    args = p.parse_args()

    results_dir: Path = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"results dir not found: {results_dir}")

    out_dir = results_dir / "pic"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"results dir: {results_dir}")
    print(f"output dir : {out_dir}")

    selected = set(args.only.split(",")) if args.only else set(ALL_STEPS)
    unknown = selected - set(ALL_STEPS)
    if unknown:
        raise SystemExit(f"unknown --only values: {sorted(unknown)}; valid: {ALL_STEPS}")

    steps = _build_steps(results_dir, out_dir)
    for name, extra in steps:
        if name not in selected:
            print(f"\n[skip] {name} (not in --only)")
            continue
        _run(name, extra, sys.executable)

    stats_out = out_dir / "run_stats.txt"
    print("\n=== run stats ===")
    _compute_run_stats(results_dir, stats_out)

    print("\n=== done ===")
    print(f"outputs in: {out_dir}")
    for f in sorted(out_dir.glob("*")):
        if f.is_file():
            print(f"  {f.relative_to(results_dir)}")


if __name__ == "__main__":
    main()
