#!/usr/bin/env python3
"""Compare accuracy on ask_time-changed questions between online and offline runs.

For every question whose ``ask_time`` was rewritten (see
``src.utils.extract_asktime_changes``), read the per-question verdicts from two
``eval_results.json`` files and report how the changed questions' accuracy moves
between the online run (original ask_time) and the offline run (ask_time=2025-12-31).

Usage:
    python -m src.utils.compare_asktime_accuracy \
        results/lifebench-hindsight \
        results/lifebench_offline-hindsight

    # Optional: override the changed-QA record and/or write a JSON summary
    python -m src.utils.compare_asktime_accuracy \
        results/lifebench-hindsight results/lifebench_offline-hindsight \
        --changed-qa datasets/lifebench_offline/asktime_changed_qa.json \
        --output results/asktime_accuracy_comparison.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_eval_results(result_dir: Path) -> Tuple[List[str], Dict[str, Dict]]:
    """Load eval_results.json and return (ordered question_ids, qid -> verdict)."""
    path = result_dir / "eval_results.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    detailed = data["detailed_results"]
    order = [d["question_id"] for d in detailed]
    verdicts = {
        d["question_id"]: {
            "is_correct": bool(d.get("is_correct")),
            "weighted_score": float(d.get("weighted_score", 0.0)),
        }
        for d in detailed
    }
    return order, verdicts


def load_changed_ids(record_path: Path) -> Tuple[List[str], List[Dict]]:
    """Load the changed-QA record and return (ordered question_ids, entries)."""
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    entries = record["changed_qa"]
    return [e["question_id"] for e in entries], entries


def acc(correct: int, total: int) -> Optional[float]:
    return (correct / total * 100.0) if total else None


def fmt_pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f}%"


def fmt_delta(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}pp"


def disp_width(s: str) -> int:
    """Display width treating CJK chars as 2 columns (for aligned tables)."""
    return sum(2 if ord(c) > 127 else 1 for c in s)


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - disp_width(s))


def summarize(
    qids: List[str], online: Dict[str, Dict], offline: Dict[str, Dict]
) -> Dict:
    """Compute accuracy/weighted-score summary over a subset of question ids."""
    on_correct = sum(1 for q in qids if q in online and online[q]["is_correct"])
    off_correct = sum(1 for q in qids if q in offline and offline[q]["is_correct"])
    on_score = sum(online[q]["weighted_score"] for q in qids if q in online)
    off_score = sum(offline[q]["weighted_score"] for q in qids if q in offline)
    on_n = sum(1 for q in qids if q in online)
    off_n = sum(1 for q in qids if q in offline)

    on_acc = acc(on_correct, on_n)
    off_acc = acc(off_correct, off_n)
    delta_acc = (off_acc - on_acc) if on_acc is not None and off_acc is not None else None
    on_ws = (on_score / on_n * 100.0) if on_n else None
    off_ws = (off_score / off_n * 100.0) if off_n else None
    delta_ws = (off_ws - on_ws) if on_ws is not None and off_ws is not None else None

    return {
        "n": on_n,
        "online_correct": on_correct,
        "offline_correct": off_correct,
        "online_accuracy": on_acc,
        "offline_accuracy": off_acc,
        "delta_accuracy": delta_acc,
        "online_weighted_score": on_ws,
        "offline_weighted_score": off_ws,
        "delta_weighted_score": delta_ws,
    }


def category_stats(
    changed_entries: List[Dict], online: Dict[str, Dict], offline: Dict[str, Dict]
) -> List[Dict]:
    """Per-question_type flip/accuracy stats over changed QAs.

    Multi-label: a QA is counted under every ``question_type`` it carries, so
    row totals can exceed the number of changed QAs.
    """
    members: Dict[str, List[str]] = defaultdict(list)
    for e in changed_entries:
        qid = e["question_id"]
        for t in (e.get("question_type") or []):
            members[t].append(qid)

    rows: List[Dict] = []
    for cat, qids in members.items():
        s = summarize(qids, online, offline)
        improved: List[str] = []
        regressed: List[str] = []
        for q in qids:
            if q not in online or q not in offline:
                continue
            on_ok = online[q]["is_correct"]
            off_ok = offline[q]["is_correct"]
            if off_ok and not on_ok:
                improved.append(q)
            elif on_ok and not off_ok:
                regressed.append(q)
        rows.append(
            {
                "category": cat,
                **s,
                "improved_count": len(improved),
                "regressed_count": len(regressed),
                "net_change": len(improved) - len(regressed),
                "improved": improved,
                "regressed": regressed,
            }
        )

    rows.sort(key=lambda r: (-r["n"], r["category"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "online_dir",
        type=Path,
        help="Online run result directory (original ask_time), e.g. results/lifebench-hindsight.",
    )
    parser.add_argument(
        "offline_dir",
        type=Path,
        help="Offline run result directory (ask_time rewritten), e.g. results/lifebench_offline-hindsight.",
    )
    parser.add_argument(
        "--changed-qa",
        type=Path,
        default=Path("datasets/lifebench_offline/asktime_changed_qa.json"),
        help="Changed-QA record produced by extract_asktime_changes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write the full comparison summary.",
    )
    args = parser.parse_args()

    on_order, online = load_eval_results(args.online_dir)
    off_order, offline = load_eval_results(args.offline_dir)

    # Detect which side is offline (for labeling only); positional args are the truth.
    def is_offline(p: Path) -> bool:
        return "offline" in p.name.lower()

    if is_offline(args.online_dir) and not is_offline(args.offline_dir):
        print("⚠  first dir name contains 'offline' — double-check argument order.\n")

    changed_ids, changed_entries = load_changed_ids(args.changed_qa)
    changed_set = set(changed_ids)

    all_ids = sorted(set(on_order) | set(off_order))
    unchanged_ids = [q for q in all_ids if q not in changed_set]

    changed_summary = summarize(changed_ids, online, offline)
    unchanged_summary = summarize(unchanged_ids, online, offline)
    overall_summary = summarize(all_ids, online, offline)

    # Flip analysis for changed questions.
    improved = []
    regressed = []
    for q in changed_ids:
        if q not in online or q not in offline:
            continue
        on_ok = online[q]["is_correct"]
        off_ok = offline[q]["is_correct"]
        if off_ok and not on_ok:
            improved.append(q)
        elif on_ok and not off_ok:
            regressed.append(q)

    cat_rows = category_stats(changed_entries, online, offline)

    print("=" * 72)
    print("ask_time 变更题正确率对比 (online vs offline)")
    print("=" * 72)
    print(f"Online  dir: {args.online_dir}")
    print(f"Offline dir: {args.offline_dir}")
    print(f"Changed-QA record: {args.changed_qa}")
    print()
    print(f"总题数: {overall_summary['n']}")
    print(f"变更题 (ask_time 被改写): {changed_summary['n']}")
    print(f"未变更题: {unchanged_summary['n']}")
    print()

    def print_block(title: str, s: Dict) -> None:
        print(f"── {title} ({s['n']}) ──")
        print(f"  Online  正确率: {s['online_correct']}/{s['n']} = {fmt_pct(s['online_accuracy'])}")
        print(f"  Offline 正确率: {s['offline_correct']}/{s['n']} = {fmt_pct(s['offline_accuracy'])}")
        print(f"  正确率变化: {fmt_delta(s['delta_accuracy'])}")
        print(f"  Online  加权分: {fmt_pct(s['online_weighted_score'])}")
        print(f"  Offline 加权分: {fmt_pct(s['offline_weighted_score'])}")
        print(f"  加权分变化: {fmt_delta(s['delta_weighted_score'])}")
        print()

    print_block("变更题", changed_summary)
    print_block("未变更题 (对照组)", unchanged_summary)
    print_block("全部题", overall_summary)

    print(f"── 变更题翻转明细 ──")
    print(f"  变好 (offline 答对 / online 答错): {len(improved)}")
    print(f"  变差 (offline 答错 / online 答对): {len(regressed)}")
    if improved:
        print(f"  变好题目: {', '.join(improved[:50])}")
        if len(improved) > 50:
            print(f"    ... 共 {len(improved)} 题")
    if regressed:
        print(f"  变差题目: {', '.join(regressed[:50])}")
        if len(regressed) > 50:
            print(f"    ... 共 {len(regressed)} 题")

    print()
    print("── 变更题分类别统计 (question_type，多标签，一题可属多类) ──")
    header = (
        pad("类别", 34)
        + pad("总数", 6)
        + pad("变好", 6)
        + pad("变差", 6)
        + pad("净变化", 8)
        + pad("Online", 9)
        + pad("Offline", 9)
        + "Δ正确率"
    )
    print(header)
    for r in cat_rows:
        net = r["net_change"]
        net_s = f"{net:+d}" if net != 0 else "0"
        print(
            pad(r["category"], 34)
            + pad(str(r["n"]), 6)
            + pad(str(r["improved_count"]), 6)
            + pad(str(r["regressed_count"]), 6)
            + pad(net_s, 8)
            + pad(fmt_pct(r["online_accuracy"]), 9)
            + pad(fmt_pct(r["offline_accuracy"]), 9)
            + fmt_delta(r["delta_accuracy"])
        )

    if args.output:
        summary = {
            "online_dir": str(args.online_dir),
            "offline_dir": str(args.offline_dir),
            "changed_qa_record": str(args.changed_qa),
            "changed": changed_summary,
            "unchanged": unchanged_summary,
            "overall": overall_summary,
            "flips": {
                "improved": improved,
                "regressed": regressed,
                "improved_count": len(improved),
                "regressed_count": len(regressed),
            },
            "by_category": cat_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSummary written to: {args.output}")


if __name__ == "__main__":
    main()