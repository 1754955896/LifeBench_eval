#!/usr/bin/env python3
"""
构建新的 dense 数据集：联合原始数据集与 v2 数据集。

规则：
- qa：完整保留 v2 数据集的全部 qa（155 条，含可答题与 Unanswerable）；
- conversation：对 v2 中每个「非空 session」，从原始数据集取该 session 的
  完整数据（而非 v2 里的 evidence-only 版本）；v2 中的空 session 丢弃。

结果是一个「保留选中 qa + 保留相关整天完整上下文」的 dense 数据集。
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
ORIGINAL_PATH = BASE_DIR / "lifebench_locomo_1people_backup.json"
V2_PATH = BASE_DIR / "lifebench_locomo_1people_dense_apr_sep_v2.json"
OUTPUT_PATH = BASE_DIR / "lifebench_locomo_1people_dense_full_sessions.json"


def main() -> None:
    with open(ORIGINAL_PATH, "r", encoding="utf-8") as f:
        original = json.load(f)[0]
    with open(V2_PATH, "r", encoding="utf-8") as f:
        v2 = json.load(f)[0]

    orig_conv = original["conversation"]

    # v2 中的非空 session 键（保持原始顺序）
    non_empty_keys = [
        k for k in v2["conversation"]
        if k.startswith("session_") and not k.endswith("_date_time")
        and v2["conversation"][k]
    ]

    # 从原始数据集取这些 session 的完整数据
    new_conv = {
        "speaker_a": orig_conv.get("speaker_a"),
        "speaker_b": orig_conv.get("speaker_b"),
    }
    for k in non_empty_keys:
        new_conv[k] = orig_conv[k]
        dt_key = f"{k}_date_time"
        if dt_key in orig_conv:
            new_conv[dt_key] = orig_conv[dt_key]

    new_person = {
        "sample_id": v2["sample_id"],
        "conversation": new_conv,
        "qa": v2["qa"],
    }

    total_items = sum(len(new_conv[k]) for k in non_empty_keys)

    print(f"sample_id: {new_person['sample_id']}")
    print(f"qa: {len(new_person['qa'])} 条")
    print(f"session: {len(non_empty_keys)} 个（非空）")
    print(f"session 数据总条数（完整）: {total_items}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump([new_person], f, ensure_ascii=False, indent=2)

    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()