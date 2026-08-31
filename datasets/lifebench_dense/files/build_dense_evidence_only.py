#!/usr/bin/env python3
"""
构建 dense（仅证据）版对话数据。

从 lifebench_locomo_1people_backup.json 中，借助
question_id_to_evidence_mapping.json 里每个 question_id 的 evidence 记录，
把每个 session 中「不是 evidence」的数据剔除，只保留「是 evidence」的数据。

匹配规则：
    evidence 记录含 source / phone_id / session_date 三个字段，
    session 里每条数据用 dia_id 标识，格式为 `{session_date}_{source}{phone_id}`，
    因此用 `{session_date}_{source}{phone_id}` 作为唯一键即可精确对齐。
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR.parent / "lifebench_raw"

DENSE_PATH = BASE_DIR / "lifebench_locomo_1people_backup.json"
MAPPING_PATH = RAW_DIR / "question_id_to_evidence_mapping.json"
OUTPUT_PATH = BASE_DIR / "lifebench_locomo_1people_evidence_only.json"


def build_evidence_dia_ids(qa_list: list[dict], mapping: dict) -> set[str]:
    """根据本文件 qa 的 question_id，从 mapping 中收集所有 evidence 对应的 dia_id。"""
    dia_ids: set[str] = set()
    for qa_item in qa_list:
        qid = qa_item["question_id"]
        for ev in mapping.get(qid, []):
            dia_id = f"{ev['session_date']}_{ev['source']}{ev['phone_id']}"
            dia_ids.add(dia_id)
    return dia_ids


def filter_conversation(conversation: dict, evidence_dia_ids: set[str]) -> dict:
    """只保留 session 中是 evidence 的数据，其余剔除；speaker 与 date_time 字段原样保留。"""
    filtered = {}
    for key, value in conversation.items():
        # session 内容键形如 session_N（date_time 键形如 session_N_date_time）
        is_session_content = key.startswith("session_") and not key.endswith("_date_time")
        if is_session_content and isinstance(value, list):
            kept = [item for item in value if item.get("dia_id") in evidence_dia_ids]
            filtered[key] = kept
        else:
            filtered[key] = value
    return filtered


def main() -> None:
    with open(DENSE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for person in data:
        qa_list = person["qa"]
        evidence_dia_ids = build_evidence_dia_ids(qa_list, mapping)

        # 统计过滤前/后的数据量
        total_before = 0
        total_after = 0
        empty_sessions = 0
        for key, value in person["conversation"].items():
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
                total_before += len(value)
                kept = sum(1 for item in value if item.get("dia_id") in evidence_dia_ids)
                total_after += kept
                if kept == 0:
                    empty_sessions += 1

        person["conversation"] = filter_conversation(person["conversation"], evidence_dia_ids)

        print(f"sample_id: {person['sample_id']}")
        print(f"  QA 数量: {len(qa_list)}")
        print(f"  去重后的 evidence 数据条数: {len(evidence_dia_ids)}")
        print(f"  session 数据: {total_before} -> {total_after}（剔除 {total_before - total_after} 条）")
        print(f"  过滤后为空的 session 数: {empty_sessions}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()