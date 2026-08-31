#!/usr/bin/env python3
"""
通用构建脚本：从 lifebench_raw.json 一次性生成 evidence / dense / sparse 三个数据集。

输入：
  - raw 数据集（{sample_id, conversation, qa} 列表，可含 1 人或多人）
  - question_id -> evidence 的映射（question_id_to_evidence_mapping.json）

输出（按人生成三份）：
  1. {prefix}_evidence.json —— 纯净 evidence（只保留被选中 QA 引用的证据数据）
  2. {prefix}_dense.json     —— 目标窗口月份的完整上下文
  3. {prefix}_sparse.json    —— evidence + 全年均匀采样的 distractor，token 总量对齐 dense

核心参数（均可通过命令行覆盖）：
  --window    目标月份（默认 4 5 6 7 8 9）
  --threshold 可答题 evidence 落在窗口内的比例阈值（默认 90，即 >=90%）
  --seed      稀疏采样随机种子（默认 42）
  --prefix    输出文件名前缀（默认 lifebench）

说明：
  - token 口径为 tiktoken 的 cl100k_base（与 recall_evaluator 一致，缺 tiktoken 时退回 CJK 启发式）。
  - evidence 的唯一键为 dia_id = {session_date}_{source}{phone_id}。
  - Unanswerable 无 evidence，按题面事件日期（X月X日）判断；题面无日期退回 ask_time。
  - dense / sparse 的 QA 与 evidence 完全一致；sparse 用「注水法」补齐 distractor 到 dense 的 token 总量。

用法示例：
  python build_lifebench_datasets.py \
      --raw ../lifebench_raw.json \
      --mapping ../../lifebench_raw/question_id_to_evidence_mapping.json \
      --out-dir .. --prefix lifebench
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - 退回 CJK 启发式
    _ENC = None

SCRIPT_DIR = Path(__file__).resolve().parent          # files/
DENSE_DIR = SCRIPT_DIR.parent                          # lifebench_dense/
DATASETS_DIR = DENSE_DIR.parent                        # datasets/

# Unanswerable 题面事件日期：匹配「X月X日」，提问时间前缀先剥离避免误匹配
EVENT_DATE_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
ASK_TIME_PREFIX_RE = re.compile(r"（提问时间：[^）]*）")


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_tokens(text: str) -> int:
    """一条文本的 token 数（cl100k_base；缺 tiktoken 时 CJK 启发式）。"""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    cjk = len(re.findall(r"[一-鿿]", text))
    rest = re.sub(r"[一-鿿]", " ", text)
    return cjk + len(rest.split())


def session_items(conv):
    """返回 [(session_key, date_time, [items])]，保持原始顺序。"""
    out = []
    for k, v in conv.items():
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(v, list):
            out.append((k, conv.get(f"{k}_date_time", ""), v))
    return out


def month_of(dt: str) -> str:
    return dt[5:7] if dt else "??"


def evidence_dia_ids(evs):
    return {f"{e['session_date']}_{e['source']}{e['phone_id']}" for e in evs}


def body_event_months(question: str):
    text = ASK_TIME_PREFIX_RE.sub("", question)
    return {int(m.group(1)) for m in EVENT_DATE_RE.finditer(text)}


def water_fill_level(E, target):
    """求水位 L，使 sum(max(e, L)) == target。E 为各月 evidence token 量。"""
    n = len(E)
    Es = sorted(E)
    for k in range(n + 1):
        top = sum(Es[n - k:])
        rem = n - k
        if rem == 0:
            continue
        L = (target - top) / rem
        if (n - k - 1 < 0 or L >= Es[n - k - 1]) and (n - k >= n or L <= Es[n - k]):
            return L
    return float("nan")


def _meta_entries(conv):
    """返回非 session 的元信息键（speaker_a / speaker_b 等）。"""
    return {k: v for k, v in conv.items() if not k.startswith("session_")}


def filter_conv_keep_empty(conv, dia_ids):
    """按 dia_id 过滤，保留所有 session 键（证据不足的 session 留空列表）。"""
    filtered = {}
    for k, v in conv.items():
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(v, list):
            filtered[k] = [it for it in v if it.get("dia_id") in dia_ids]
        else:
            filtered[k] = v
    return filtered


def filter_conv_drop_empty(conv, dia_ids):
    """按 dia_id 过滤，丢弃为空的 session。"""
    filtered = _meta_entries(conv)
    for k, dt, items in session_items(conv):
        kept = [it for it in items if it.get("dia_id") in dia_ids]
        if kept:
            filtered[k] = kept
            filtered[f"{k}_date_time"] = dt
    return filtered


def select_months_full(conv, months: set):
    """只保留指定月份 session 的完整数据（丢弃其他月份）。"""
    filtered = _meta_entries(conv)
    for k, dt, items in session_items(conv):
        if month_of(dt) in months:
            filtered[k] = items
            filtered[f"{k}_date_time"] = dt
    return filtered


# ---------------------------------------------------------------------------
# QA 选择 + evidence 提取
# ---------------------------------------------------------------------------

def select_qa_and_evidence(raw_person, mapping, window, threshold):
    """选出 evidence 的 qa，返回 (selected_qa, evidence_dia_id 集合)。"""
    win = set(window)
    selected_qa = []
    dia_ids = set()
    for qa in raw_person["qa"]:
        qid = qa["question_id"]
        is_unanswerable = "Unanswerable" in qa["question_type"]
        if is_unanswerable:
            months = body_event_months(qa["question"])
            if months:
                keep = bool(months & win)
            else:
                keep = int(qa["ask_time"][5:7]) in win
        else:
            evs = mapping.get(qid, [])
            if not evs:
                keep = False
            else:
                inside = sum(1 for e in evs if int(e["session_date"][5:7]) in win)
                keep = inside * 100 >= len(evs) * threshold
        if keep:
            selected_qa.append(qa)
            if not is_unanswerable:
                dia_ids |= evidence_dia_ids(mapping.get(qid, []))
    return selected_qa, dia_ids


# ---------------------------------------------------------------------------
# 三个构建阶段
# ---------------------------------------------------------------------------

def build_evidence(raw_person, mapping, window, threshold):
    qa, dia_ids = select_qa_and_evidence(raw_person, mapping, window, threshold)
    conv = filter_conv_keep_empty(raw_person["conversation"], dia_ids)
    return {"sample_id": raw_person["sample_id"], "conversation": conv, "qa": qa}


def build_dense(raw_person, mapping, window, threshold):
    qa, _ = select_qa_and_evidence(raw_person, mapping, window, threshold)
    months = {f"{m:02d}" for m in window}
    conv = select_months_full(raw_person["conversation"], months)
    return {"sample_id": raw_person["sample_id"], "conversation": conv, "qa": qa}


def build_sparse(raw_person, mapping, window, threshold, seed):
    evidence = build_evidence(raw_person, mapping, window, threshold)
    dense = build_dense(raw_person, mapping, window, threshold)
    target = sum(
        count_tokens(it["text"])
        for _k, _dt, items in session_items(dense["conversation"])
        for it in items
    )

    ev_dia_ids = {
        it["dia_id"] for _k, _dt, items in session_items(evidence["conversation"]) for it in items
    }
    months = [f"{m:02d}" for m in range(1, 13)]
    ev_by_month = {m: [] for m in months}
    non_ev_groups = {m: defaultdict(list) for m in months}
    for _k, dt, items in session_items(raw_person["conversation"]):
        m = month_of(dt)
        if m not in ev_by_month:
            continue
        for it in items:
            if it["dia_id"] in ev_dia_ids:
                ev_by_month[m].append(it)
            else:
                non_ev_groups[m][it["dia_id"]].append(it)

    E = {m: sum(count_tokens(it["text"]) for it in ev_by_month[m]) for m in months}
    L = water_fill_level([E[m] for m in months], target)
    floors = {m: math.floor(max(E[m], L)) for m in months}
    residual = target - sum(floors.values())
    level_months = [m for m in months if E[m] <= L]
    for j in range(residual):
        floors[level_months[j % len(level_months)]] += 1

    rng = random.Random(seed)
    sampled = set()
    for m in months:
        need = floors[m] - E[m]
        if need <= 0:
            continue
        groups = list(non_ev_groups[m].items())
        rng.shuffle(groups)
        cum = 0
        for dia_id, items in groups:
            t = sum(count_tokens(it["text"]) for it in items)
            if cum + t >= need:
                if (cum + t - need) <= (need - cum):
                    sampled.add(dia_id)
                break
            sampled.add(dia_id)
            cum += t

    conv = filter_conv_drop_empty(raw_person["conversation"], ev_dia_ids | sampled)
    return {"sample_id": raw_person["sample_id"], "conversation": conv, "qa": evidence["qa"]}


# ---------------------------------------------------------------------------
# 汇总与主流程
# ---------------------------------------------------------------------------

def summarize(name, person):
    items = 0
    tk = 0
    sessions = 0
    for k, dt, lst in session_items(person["conversation"]):
        if lst:
            sessions += 1
        for it in lst:
            items += 1
            tk += count_tokens(it.get("text", ""))
    print(f"  {name:<10} qa={len(person['qa']):>4}  items={items:>5}  token={tk:>7}  session={sessions:>4}")
    return tk


def main():
    parser = argparse.ArgumentParser(description="从 raw 生成 evidence / dense / sparse")
    parser.add_argument("--raw", default=str(DENSE_DIR / "lifebench_raw.json"))
    parser.add_argument("--mapping",
                        default=str(DATASETS_DIR / "lifebench_raw" / "question_id_to_evidence_mapping.json"))
    parser.add_argument("--out-dir", default=str(DENSE_DIR))
    parser.add_argument("--prefix", default="lifebench")
    parser.add_argument("--window", nargs="+", type=int, default=[4, 5, 6, 7, 8, 9])
    parser.add_argument("--threshold", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_json(args.raw)
    mapping = load_json(args.mapping)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    multi = len(raw) > 1
    for person in raw:
        base = f"{args.prefix}_{person['sample_id']}" if multi else args.prefix
        print(f"sample_id: {person['sample_id']}")

        evidence = build_evidence(person, mapping, args.window, args.threshold)
        dense = build_dense(person, mapping, args.window, args.threshold)
        sparse = build_sparse(person, mapping, args.window, args.threshold, args.seed)

        ev_tk = summarize("evidence", evidence)
        dn_tk = summarize("dense", dense)
        sp_tk = summarize("sparse", sparse)

        outputs = [
            (f"{base}_evidence.json", evidence),
            (f"{base}_dense.json", dense),
            (f"{base}_sparse.json", sparse),
        ]
        for fname, data in outputs:
            with open(out_dir / fname, "w", encoding="utf-8") as f:
                json.dump([data], f, ensure_ascii=False, indent=2)

        # 校验：evidence 应完全被 dense / sparse 包含
        ev_keys = {it["dia_id"] for _k, _dt, items in session_items(evidence["conversation"]) for it in items}
        dn_keys = {it["dia_id"] for _k, _dt, items in session_items(dense["conversation"]) for it in items}
        sp_keys = {it["dia_id"] for _k, _dt, items in session_items(sparse["conversation"]) for it in items}
        print(f"  evidence 缺失 dia_id -> dense: {len(ev_keys - dn_keys)} 个, sparse: {len(ev_keys - sp_keys)} 个")
        print(f"  token: dense {dn_tk}  vs  sparse {sp_tk}  (差 {sp_tk - dn_tk:+d})")
        print(f"  -> {base}_evidence.json / {base}_dense.json / {base}_sparse.json")
        print()

    print("Done.")


if __name__ == "__main__":
    main()