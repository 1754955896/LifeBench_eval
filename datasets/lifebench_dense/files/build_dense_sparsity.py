#!/usr/bin/env python3
"""
构建 sparsity（稀疏）数据集。

目标：在 evidence（lifebench_evidence.json）的「纯净 evidence」基础上，从各个月份
补入额外的非 evidence 数据（distractor），使得：
  1. token 总量与 dense（lifebench_dense.json）完全一致；
  2. 每个月的数据量分布尽量均匀 —— 数据不集中在 4~9 目标月份。

做法（water-fill / 注水）：
  - 保留 evidence 全部（固定，不可删减）；
  - 求一个「水位」L，使 sum(max(月 evidence 量, L)) == 目标总量；
  - 对每个 evidence 不足 L 的月份，从该月非 evidence 数据中抽样补足到 L；
  - evidence 本身已超 L 的月份（05 月）保持原样，不额外补、也不删。

非 evidence 抽样按 dia_id 分组（保持多轮 agent_chat 完整），固定随机种子可复现。
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
SEED = 42

# 数据文件位于上一级 lifebench_dense/，本脚本在 files/ 下
BASE = Path(__file__).resolve().parent.parent
RAW_PATH = BASE / "lifebench_raw.json"
EVIDENCE_PATH = BASE / "lifebench_evidence.json"
DENSE_PATH = BASE / "lifebench_dense.json"
OUTPUT_PATH = BASE / "lifebench_sparse.json"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)[0]


def tokens(text: str) -> int:
    return len(ENC.encode(text or ""))


def session_items(conv):
    """返回 [(session_key, date_time, [items])]，保持原始顺序。"""
    out = []
    for k, v in conv.items():
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(v, list):
            out.append((k, conv.get(f"{k}_date_time", ""), v))
    return out


def month_of(dt: str) -> str:
    return dt[5:7] if dt else "??"


def water_fill_level(E, target):
    """求水位 L，使 sum(max(e, L)) == target。E 为各月 evidence token 量。"""
    n = len(E)
    E_sorted = sorted(E)
    for k in range(n + 1):  # k = 严格高于水位的月份数
        top = sum(E_sorted[n - k:])
        rem = n - k
        if rem == 0:
            continue
        L = (target - top) / rem
        lo_ok = (n - k - 1 < 0) or (L >= E_sorted[n - k - 1])
        hi_ok = (n - k >= n) or (L <= E_sorted[n - k])
        if lo_ok and hi_ok:
            return L
    return float("nan")


def main() -> None:
    raw = load(RAW_PATH)
    evidence = load(EVIDENCE_PATH)
    dense = load(DENSE_PATH)

    # 1) evidence 的 dia_id（来自 evidence 对话数据）
    evidence_dia_ids = set()
    for _k, _dt, items in session_items(evidence["conversation"]):
        for it in items:
            evidence_dia_ids.add(it["dia_id"])

    # 2) 目标 token 总量 = dense
    target = sum(
        tokens(it["text"])
        for _k, _dt, items in session_items(dense["conversation"])
        for it in items
    )

    # 3) 原始数据按月分桶：evidence 与 非 evidence（按 dia_id 分组，保持多轮完整）
    months = [f"{m:02d}" for m in range(1, 13)]
    ev_by_month = {m: [] for m in months}            # month -> [evidence items]
    non_ev_groups = {m: defaultdict(list) for m in months}  # month -> dia_id -> [items]

    for _k, dt, items in session_items(raw["conversation"]):
        m = month_of(dt)
        if m not in ev_by_month:
            continue
        for it in items:
            if it["dia_id"] in evidence_dia_ids:
                ev_by_month[m].append(it)
            else:
                non_ev_groups[m][it["dia_id"]].append(it)

    E = {m: sum(tokens(it["text"]) for it in ev_by_month[m]) for m in months}
    evidence_total = sum(E.values())

    # 4) 水位 L 与逐月目标
    L = water_fill_level([E[m] for m in months], target)
    raw_level = {m: max(E[m], L) for m in months}
    floors = {m: math.floor(raw_level[m]) for m in months}
    residual = target - sum(floors.values())
    level_months = [m for m in months if E[m] <= L]
    for j in range(residual):
        floors[level_months[j % len(level_months)]] += 1

    # 5) 抽样非 evidence 数据补足每月缺口
    rng = random.Random(SEED)
    sampled_non_ev_dia_ids = set()
    for m in months:
        extra_needed = floors[m] - E[m]
        if extra_needed <= 0:
            continue
        groups = list(non_ev_groups[m].items())  # [(dia_id, [items])]
        rng.shuffle(groups)
        cum = 0
        for dia_id, items in groups:
            t = sum(tokens(it["text"]) for it in items)
            if cum + t >= extra_needed:
                if (cum + t - extra_needed) <= (extra_needed - cum):
                    sampled_non_ev_dia_ids.add(dia_id)
                break
            sampled_non_ev_dia_ids.add(dia_id)
            cum += t

    # 6) 重建 conversation：保留 evidence 与抽中的非 evidence
    keep_dia_ids = evidence_dia_ids | sampled_non_ev_dia_ids
    new_conv = {
        "speaker_a": raw["conversation"].get("speaker_a"),
        "speaker_b": raw["conversation"].get("speaker_b"),
    }
    for k, dt, items in session_items(raw["conversation"]):
        kept = [it for it in items if it["dia_id"] in keep_dia_ids]
        if kept:
            new_conv[k] = kept
            new_conv[f"{k}_date_time"] = dt

    new_person = {
        "sample_id": evidence["sample_id"],
        "conversation": new_conv,
        "qa": evidence["qa"],
    }

    # 7) 统计与校验
    final_tokens_by_month = defaultdict(int)
    final_items_by_month = defaultdict(int)
    for k, dt, items in session_items(new_conv):
        m = month_of(dt)
        final_tokens_by_month[m] += sum(tokens(it["text"]) for it in items)
        final_items_by_month[m] += len(items)

    final_total = sum(final_tokens_by_month.values())
    present_dia_ids = {it["dia_id"] for _k, _dt, items in session_items(new_conv) for it in items}
    missing_ev = evidence_dia_ids - present_dia_ids

    print(f"目标 token 总量（dense）: {target}")
    print(f"evidence token 总量: {evidence_total}")
    print(f"额外补入 token 预算: {target - evidence_total}")
    print(f"水位 L: {L:.2f}")
    print(f"\n{'月':<4}{'ev token':>10}{'目标':>8}{'最终token':>10}{'最终条数':>8}")
    for m in months:
        print(f"{m:<4}{E[m]:>10}{floors[m]:>8}{final_tokens_by_month[m]:>10}{final_items_by_month[m]:>8}")
    print(f"\n最终 token 总量: {final_total}  (与目标差 {final_total - target:+d})")
    print(f"evidence 缺失 dia_id: {len(missing_ev)}")
    print(f"qa: {len(new_person['qa'])} 条")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump([new_person], f, ensure_ascii=False, indent=2)
    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()