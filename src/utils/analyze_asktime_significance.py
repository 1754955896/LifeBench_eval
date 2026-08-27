#!/usr/bin/env python3
"""Test whether online-vs-offline accuracy changes on ask_time-changed QAs are real.

For paired binary verdicts (same questions judged in both modes) the appropriate
test is McNemar's exact test on the discordant pairs (questions that flip):

    b = online correct -> offline wrong   (regressed)
    c = online wrong   -> offline correct (improved)

Under "no real difference" the flips are symmetric around 0.5, so b ~ Binomial(b+c, 0.5).

The unchanged questions (identical ask_time in both modes) serve as a control
group: their change isolates run-mechanism + sampling noise, so the net effect of
rewriting ask_time is (changed delta) - (control delta), estimated with bootstrap.

Usage:
    python -m src.utils.analyze_asktime_significance \
        results/lifebench-hindsight \
        results/lifebench_offline-hindsight
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest

from src.utils.compare_asktime_accuracy import (
    load_changed_ids,
    load_eval_results,
)


def discordant(
    ids: List[str], online: Dict[str, Dict], offline: Dict[str, Dict]
) -> Tuple[int, int, int, int]:
    """Return (both_correct, regressed, improved, both_wrong) for a question subset."""
    both_correct = regressed = improved = both_wrong = 0
    for q in ids:
        if q not in online or q not in offline:
            continue
        on = online[q]["is_correct"]
        off = offline[q]["is_correct"]
        if on and off:
            both_correct += 1
        elif on and not off:
            regressed += 1
        elif not on and off:
            improved += 1
        else:
            both_wrong += 1
    return both_correct, regressed, improved, both_wrong


def mcnemar_pvalue(regressed: int, improved: int) -> float:
    """Two-sided exact McNemar p-value from discordant pairs."""
    total = regressed + improved
    if total == 0:
        return 1.0
    return float(binomtest(regressed, total, 0.5).pvalue)


def accuracy_delta(
    ids: List[str], online: Dict[str, Dict], offline: Dict[str, Dict]
) -> float:
    """(offline accuracy - online accuracy) over a subset, as a proportion delta."""
    valid = [q for q in ids if q in online and q in offline]
    if not valid:
        return 0.0
    on = sum(online[q]["is_correct"] for q in valid)
    off = sum(offline[q]["is_correct"] for q in valid)
    return (off - on) / len(valid)


def bootstrap_net_effect(
    changed_ids: List[str],
    unchanged_ids: List[str],
    online: Dict[str, Dict],
    offline: Dict[str, Dict],
    n_boot: int = 20000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Net effect of ask_time rewrite = changed delta - control delta, with 95% CI."""
    rng = np.random.default_rng(seed)

    def verdicts(ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        valid = [q for q in ids if q in online and q in offline]
        on = np.array([online[q]["is_correct"] for q in valid], dtype=float)
        off = np.array([offline[q]["is_correct"] for q in valid], dtype=float)
        return on, off

    on_c, off_c = verdicts(changed_ids)
    on_u, off_u = verdicts(unchanged_ids)

    net = (off_c.mean() - on_c.mean()) - (off_u.mean() - on_u.mean())

    nc, nu = len(on_c), len(on_u)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ic = rng.integers(0, nc, nc)
        iu = rng.integers(0, nu, nu)
        dc = off_c[ic].mean() - on_c[ic].mean()
        du = off_u[iu].mean() - on_u[iu].mean()
        boots[i] = dc - du

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return net, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("online_dir", type=Path)
    parser.add_argument("offline_dir", type=Path)
    parser.add_argument(
        "--changed-qa",
        type=Path,
        default=Path("datasets/lifebench_offline/asktime_changed_qa.json"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    on_order, online = load_eval_results(args.online_dir)
    off_order, offline = load_eval_results(args.offline_dir)
    changed_ids, changed_entries = load_changed_ids(args.changed_qa)
    changed_set = set(changed_ids)

    all_ids = sorted(set(on_order) | set(off_order))
    unchanged_ids = [q for q in all_ids if q not in changed_set]

    print("=" * 78)
    print("ask_time 变更正确率变化的显著性分析 (McNemar 精确检验)")
    print("=" * 78)
    print(f"Online:  {args.online_dir}")
    print(f"Offline: {args.offline_dir}")
    print(f"alpha = {args.alpha}")
    print()

    # ── overall ──
    def report(label: str, ids: List[str]) -> None:
        bc, reg, imp, bw = discordant(ids, online, offline)
        n = bc + reg + imp + bw
        p = mcnemar_pvalue(reg, imp)
        d = accuracy_delta(ids, online, offline)
        sig = "显著" if p < args.alpha else "不显著"
        print(f"── {label} (n={n}) ──")
        print(f"  变好(online错→offline对): {imp}")
        print(f"  变差(online对→offline错): {reg}")
        print(f"  McNemar p = {p:.4f}  →  {sig}")
        print(f"  Δ正确率 = {d*100:+.2f}pp")
        print()

    report("变更题 (ask_time 被改写)", changed_ids)
    report("未变更题 (对照组，ask_time 相同)", unchanged_ids)

    # ── per category ──
    members: Dict[str, List[str]] = defaultdict(list)
    for e in changed_entries:
        for t in (e.get("question_type") or []):
            members[t].append(e["question_id"])

    print("── 变更题分类别 McNemar ──")
    print(f"{'类别':34s}{'n':>5s}{'变好':>6s}{'变差':>6s}{'p值':>9s}   判定")
    rows = []
    for cat, qids in sorted(members.items(), key=lambda kv: -len(kv[1])):
        bc, reg, imp, bw = discordant(qids, online, offline)
        p = mcnemar_pvalue(reg, imp)
        rows.append((cat, len(qids), imp, reg, p))
        sig = "显著" if p < args.alpha else ""
        print(f"{cat:34s}{len(qids):>5d}{imp:>6d}{reg:>6d}{p:>9.4f}   {sig}")

    print()
    print("注：类别为多标签，一题可属多类，故 n 之和 > 1499；小样本类别检验力弱。")
    print()

    # ── net effect ──
    net, lo, hi = bootstrap_net_effect(
        changed_ids, unchanged_ids, online, offline
    )
    print("── ask_time 改写的净效应 (变更Δ - 对照组Δ, bootstrap 95% CI) ──")
    print(f"  net = {net*100:+.2f}pp  95% CI = [{lo*100:+.2f}pp, {hi*100:+.2f}pp]")
    if lo > 0:
        verdict = "显著为正（改写 ask_time 显著提升正确率）"
    elif hi < 0:
        verdict = "显著为负（改写 ask_time 显著降低正确率）"
    else:
        verdict = "包含 0，未检测到显著净效应（变化可归因于噪声/运行机制差异）"
    print(f"  判定: {verdict}")


if __name__ == "__main__":
    main()