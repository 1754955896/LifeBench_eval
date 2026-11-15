#!/usr/bin/env python3
"""
从 3380QA 数据集抽取指定人物，生成多人的 lifebench_raw.json（后续构建的 raw 输入）。

默认抽取三人：孙雨薇 / 于晓薇 / 冯浩然。
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # files/
DENSE_DIR = SCRIPT_DIR.parent                          # lifebench_dense/
DATASETS_DIR = DENSE_DIR.parent                        # datasets/

SRC = DATASETS_DIR / "lifebench_locomo_format" / "lifebench_locomo_conversation_format_v2.0_3380QA.json"
OUT = DENSE_DIR / "lifebench_raw.json"
PEOPLE = ["孙雨薇", "于晓薇", "冯浩然"]


def main() -> None:
    data = json.load(open(SRC, encoding="utf-8"))
    by_id = {p["sample_id"]: p for p in data}

    missing = [name for name in PEOPLE if name not in by_id]
    if missing:
        raise SystemExit(f"3380QA 中找不到: {missing}")

    selected = [by_id[name] for name in PEOPLE]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    for p in selected:
        conv = p["conversation"]
        items = sum(
            len(v)
            for k, v in conv.items()
            if k.startswith("session_") and not k.endswith("_date_time") and isinstance(v, list)
        )
        print(f"  {p['sample_id']:6s} qa={len(p['qa']):>4} items={items:>5}")

    print(f"Done. 共 {len(selected)} 人 -> {OUT}")


if __name__ == "__main__":
    main()