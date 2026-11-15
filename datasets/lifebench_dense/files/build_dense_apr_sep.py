#!/usr/bin/env python3
"""
筛选 evidence 集中在 4~9 月（2025-04 ~ 2025-09）的 qa，构成新的 dense 数据集。

从 lifebench_locomo_1people_backup.json 出发，借助
question_id_to_evidence_mapping.json 中每条 evidence 的 session_date，
统计每个 qa 的 evidence 落在目标 6 个月份里的占比：
    - 占比 >= 90%（含）的 qa 被保留；
    - 空 evidence 的 qa（Unanswerable 类型）无时间集中度，被剔除；
    - 其余 qa 被剔除。

新数据集沿用原格式 {sample_id, conversation, qa}，其中：
    - qa 为筛选后的子集；
    - conversation 只保留被选中 qa 所引用 evidence 对应的数据（按 dia_id 对齐），
      即 self-contained 的 dense 子集。
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR.parent / "lifebench_raw"

DENSE_PATH = BASE_DIR / "lifebench_locomo_1people_backup.json"
MAPPING_PATH = RAW_DIR / "question_id_to_evidence_mapping.json"
OUTPUT_PATH = BASE_DIR / "lifebench_locomo_1people_dense_apr_sep.json"

TARGET_MONTHS = {"04", "05", "06", "07", "08", "09"}  # 4~9 月
THRESHOLD_PERCENT = 90  # 90% 以上（含）


def evidence_dia_ids(evs: list[dict]) -> set[str]:
    """evidence 记录 -> dia_id 集合（{session_date}_{source}{phone_id}）。"""
    return {f"{e['session_date']}_{e['source']}{e['phone_id']}" for e in evs}


def in_target_months(session_date: str) -> bool:
    return session_date[5:7] in TARGET_MONTHS


def filter_conversation(conversation: dict, evidence_dia_ids: set[str]) -> dict:
    """只保留 session 中被选中 evidence 引用的数据，其余剔除。"""
    filtered = {}
    for key, value in conversation.items():
        is_session_content = key.startswith("session_") and not key.endswith("_date_time")
        if is_session_content and isinstance(value, list):
            filtered[key] = [item for item in value if item.get("dia_id") in evidence_dia_ids]
        else:
            filtered[key] = value
    return filtered


def main() -> None:
    with open(DENSE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    new_data = []
    stats = {"total": 0, "empty": 0, "selected": 0, "rejected": 0}

    for person in data:
        selected_qa = []
        selected_evidence_dia_ids: set[str] = set()

        for qa_item in person["qa"]:
            stats["total"] += 1
            evs = mapping.get(qa_item["question_id"], [])

            if not evs:
                # 空 evidence（Unanswerable）无时间集中度，剔除
                stats["empty"] += 1
                continue

            inside = sum(1 for e in evs if in_target_months(e["session_date"]))
            # 整数比较避免浮点误差：inside/len >= 0.9 等价于 inside*100 >= len*90
            if inside * 100 >= len(evs) * THRESHOLD_PERCENT:
                stats["selected"] += 1
                selected_qa.append(qa_item)
                selected_evidence_dia_ids |= evidence_dia_ids(evs)
            else:
                stats["rejected"] += 1

        filtered_conv = filter_conversation(person["conversation"], selected_evidence_dia_ids)

        new_person = {
            "sample_id": person["sample_id"],
            "conversation": filtered_conv,
            "qa": selected_qa,
        }
        new_data.append(new_person)

        # 统计对话保留情况
        total_kept = 0
        for key, value in filtered_conv.items():
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
                total_kept += len(value)
        print(f"sample_id: {person['sample_id']}")
        print(f"  原始 qa: {len(person['qa'])} -> 选中 qa: {len(selected_qa)}")
        print(f"  选中 qa 引用 evidence（去重 dia_id）: {len(selected_evidence_dia_ids)}")
        print(f"  conversation 保留数据: {total_kept} 条")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"\n汇总: total={stats['total']}, selected={stats['selected']}, "
          f"rejected={stats['rejected']}, empty(Unanswerable)={stats['empty']}")
    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()