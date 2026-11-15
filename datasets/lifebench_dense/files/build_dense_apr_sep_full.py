#!/usr/bin/env python3
"""
构建「完整 4~9 月」dense 数据集。

在现有 dense（full_sessions）基础上，补入原始数据 4~9 月中所有尚未加入的
session（即把 4~9 月补满），qa 保持 v2 的全部 qa 不变。

结果 = 原始数据 4~9 月的全部 session（完整数据）+ v2 的 155 条 qa。
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
ORIGINAL_PATH = BASE_DIR / "lifebench_locomo_1people_backup.json"
V2_PATH = BASE_DIR / "lifebench_locomo_1people_dense_apr_sep_v2.json"
OUTPUT_PATH = BASE_DIR / "lifebench_locomo_1people_dense_apr_sep_full.json"

TARGET_MONTHS = {"04", "05", "06", "07", "08", "09"}


def main() -> None:
    with open(ORIGINAL_PATH, "r", encoding="utf-8") as f:
        original = json.load(f)[0]
    with open(V2_PATH, "r", encoding="utf-8") as f:
        v2 = json.load(f)[0]

    orig_conv = original["conversation"]

    new_conv = {
        "speaker_a": orig_conv.get("speaker_a"),
        "speaker_b": orig_conv.get("speaker_b"),
    }
    session_count = 0
    item_count = 0
    for key in orig_conv:
        if key.startswith("session_") and not key.endswith("_date_time"):
            dt = orig_conv.get(f"{key}_date_time", "")
            if dt[5:7] in TARGET_MONTHS:
                new_conv[key] = orig_conv[key]
                new_conv[f"{key}_date_time"] = dt
                session_count += 1
                item_count += len(orig_conv[key])

    new_person = {
        "sample_id": v2["sample_id"],
        "conversation": new_conv,
        "qa": v2["qa"],
    }

    print(f"sample_id: {new_person['sample_id']}")
    print(f"qa: {len(new_person['qa'])} 条")
    print(f"session: {session_count} 个（4~9 月全部）")
    print(f"session 数据总条数: {item_count}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump([new_person], f, ensure_ascii=False, indent=2)

    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()