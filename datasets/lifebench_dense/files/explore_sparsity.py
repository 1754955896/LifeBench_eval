#!/usr/bin/env python3
"""探索 original / v2 / full_sessions 三个数据集的每月条数与 token 分布。"""
import json
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
BASE = Path(__file__).parent


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)[0]


def month_of(key, conv):
    dt = conv.get(f"{key}_date_time", "")
    return dt[5:7] if dt else "??"


def per_month_items(conv):
    out = {}
    for k, v in conv.items():
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(v, list):
            m = month_of(k, conv)
            out[m] = out.get(m, 0) + len(v)
    return out


def per_month_tokens(conv):
    out = {}
    for k, v in conv.items():
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(v, list):
            m = month_of(k, conv)
            t = sum(len(ENC.encode(item.get("text", ""))) for item in v)
            out[m] = out.get(m, 0) + t
    return out


def show(name, conv):
    items = per_month_items(conv)
    tokens = per_month_tokens(conv)
    total_items = sum(items.values())
    total_tokens = sum(tokens.values())
    print(f"\n=== {name} ===  items={total_items}  tokens={total_tokens}")
    print(f"{'月':<4}{'条数':>8}{'token':>10}")
    for m in sorted(items):
        print(f"{m:<4}{items[m]:>8}{tokens[m]:>10}")
    return total_items, total_tokens


if __name__ == "__main__":
    orig = load(BASE / "lifebench_locomo_1people_backup.json")
    v2 = load(BASE / "lifebench_locomo_1people_dense_apr_sep_v2.json")
    full = load(BASE / "lifebench_locomo_1people_dense_full_sessions.json")

    show("original backup", orig["conversation"])
    show("v2 (evidence-only)", v2["conversation"])
    show("full_sessions", full["conversation"])