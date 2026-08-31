#!/usr/bin/env python3
"""
构建 4~9 月 dense 子集（含 Unanswerable）。

筛选规则：
- 可答题（非 Unanswerable）：evidence 落在 4~9 月的比例 >= 90% 保留；
- Unanswerable（无 evidence）：
    优先按题面「事件日期」（年月日格式）判断，任一日期落在 4~9 月即保留；
    题面无明确日期的，退回按 ask_time（提问时间）判断，落在 4~9 月即保留。

输出沿用 {sample_id, conversation, qa} 格式，conversation 只保留被选中
可答题所引用 evidence 对应的数据（Unanswerable 无 evidence，不贡献对话数据）。
"""

import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR.parent / "lifebench_raw"

DENSE_PATH = BASE_DIR / "lifebench_locomo_1people_backup.json"
MAPPING_PATH = RAW_DIR / "question_id_to_evidence_mapping.json"
OUTPUT_PATH = BASE_DIR / "lifebench_locomo_1people_dense_apr_sep_v2.json"

WINDOW_MONTHS = {4, 5, 6, 7, 8, 9}  # 4~9 月
THRESHOLD_PERCENT = 90

# 题面事件日期：匹配「X月X日」（中文格式，含「2025年X月X日」中的部分）
EVENT_DATE_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
# 提问时间前缀，提取事件日期前先去掉，避免误匹配
ASK_TIME_PREFIX_RE = re.compile(r"（提问时间：[^）]*）")


def body_event_months(question: str) -> set[int]:
    """题面中出现的所有事件日期月份（去掉提问时间前缀后）。"""
    text = ASK_TIME_PREFIX_RE.sub("", question)
    months: set[int] = set()
    for m in EVENT_DATE_RE.finditer(text):
        months.add(int(m.group(1)))
    return months


def keep_unanswerable(qa_item: dict) -> bool:
    """Unanswerable：优先按题面事件日期，无日期则退回 ask_time。"""
    event_months = body_event_months(qa_item["question"])
    if event_months:
        return bool(event_months & WINDOW_MONTHS)
    return int(qa_item["ask_time"][5:7]) in WINDOW_MONTHS


def keep_answerable(qa_item: dict, mapping: dict) -> bool:
    """可答题：evidence 落在 4~9 月的比例 >= 90%。"""
    evs = mapping.get(qa_item["question_id"], [])
    if not evs:
        return False
    inside = sum(1 for e in evs if int(e["session_date"][5:7]) in WINDOW_MONTHS)
    return inside * 100 >= len(evs) * THRESHOLD_PERCENT


def evidence_dia_ids(evs: list[dict]) -> set[str]:
    return {f"{e['session_date']}_{e['source']}{e['phone_id']}" for e in evs}


def filter_conversation(conversation: dict, evidence_dia_ids: set[str]) -> dict:
    filtered = {}
    for key, value in conversation.items():
        is_session_content = key.startswith("session_") and not key.endswith("_date_time")
        if is_session_content and isinstance(value, list):
            filtered[key] = [item for item in value if item.get("dia_id") in evidence_dia_ids]
        else:
            filtered[key] = value
    return filtered


def multi_label_distribution(qa_list: list[dict]) -> Counter:
    counter: Counter = Counter()
    for qa_item in qa_list:
        for t in qa_item["question_type"]:
            counter[t] += 1
    return counter


def main() -> None:
    with open(DENSE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    new_data = []
    for person in data:
        selected_qa = []
        selected_evidence_dia_ids: set[str] = set()
        stats = Counter()

        for qa_item in person["qa"]:
            if "Unanswerable" in qa_item["question_type"]:
                if keep_unanswerable(qa_item):
                    stats["ua_kept"] += 1
                    selected_qa.append(qa_item)
                else:
                    stats["ua_dropped"] += 1
            else:
                if keep_answerable(qa_item, mapping):
                    stats["ans_kept"] += 1
                    selected_qa.append(qa_item)
                    selected_evidence_dia_ids |= evidence_dia_ids(
                        mapping.get(qa_item["question_id"], [])
                    )
                else:
                    stats["ans_dropped"] += 1

        filtered_conv = filter_conversation(person["conversation"], selected_evidence_dia_ids)

        new_person = {
            "sample_id": person["sample_id"],
            "conversation": filtered_conv,
            "qa": selected_qa,
        }
        new_data.append(new_person)

        # 统计对话保留
        total_kept = 0
        for key, value in filtered_conv.items():
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
                total_kept += len(value)

        print(f"sample_id: {person['sample_id']}")
        print(f"  原始 qa: {len(person['qa'])} -> 选中 qa: {len(selected_qa)}")
        print(f"    可答题: 保留 {stats['ans_kept']} / 剔除 {stats['ans_dropped']}")
        print(f"    Unanswerable: 保留 {stats['ua_kept']} / 剔除 {stats['ua_dropped']}")
        print(f"  conversation 保留数据: {total_kept} 条")

        # 最终多标签类别分布
        dist = multi_label_distribution(selected_qa)
        print(f"  === 最终多标签类别分布（选中 {len(selected_qa)} 条）===")
        order = ["Single_hop", "Multi_hop", "Temporal", "Conflict", "Causal",
                 "Pattern_recognition(Non-declarative)", "Knowledge_update",
                 "Hidden_info", "Unanswerable"]
        for t in order:
            if dist.get(t):
                print(f"    {t:<32} {dist[t]}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()