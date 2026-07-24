#!/usr/bin/env python3
"""
构建 question_id → raw evidence 映射字典。

从 lifebench_locomo_conversation_format_v2.0_3380QA.json 中提取每个 question_id，
通过 (question文本, ask_time) 复合键精确匹配到 lifebench_raw/ 下对应人物数据中
qa 的 evidence 字段。

question_id 格式: lifebench_v5_{pinyin}_qa{N}
- pinyin: 对应 raw 文件名 lifebench_multi_source_format_{pinyin}.json
- N: conv 数据集内部编号（不直接对应 raw 索引）
"""

import json
import os
import re
from pathlib import Path


def build_mapping(conv_path: str, raw_dir: str, output_path: str):
    with open(conv_path, 'r', encoding='utf-8') as f:
        conv_data = json.load(f)

    raw_cache: dict[str, dict] = {}
    raw_index_cache: dict[str, dict[tuple[str, str], int]] = {}

    question_id_pattern = re.compile(r'^lifebench_v5_(.+)_qa(\d+)$')

    mapping: dict[str, list[dict]] = {}
    stats = {"total": 0, "unmatched": 0, "missing_raw_file": 0}

    for person in conv_data:
        qa_list = person["qa"]

        for qa_item in qa_list:
            qid = qa_item["question_id"]
            m = question_id_pattern.match(qid)
            if not m:
                print(f"[WARN] Unexpected question_id format: {qid}")
                continue

            pinyin = m.group(1)

            # 加载对应 raw 文件（带缓存）
            if pinyin not in raw_cache:
                raw_path = os.path.join(raw_dir, f"lifebench_multi_source_format_{pinyin}.json")
                if not os.path.exists(raw_path):
                    print(f"[WARN] Raw file not found: {raw_path}")
                    stats["missing_raw_file"] += 1
                    continue
                with open(raw_path, 'r', encoding='utf-8') as f:
                    raw_cache[pinyin] = json.load(f)

                # 为该人物构建 (question, ask_time) → raw_index 查找表
                raw_qa_list = raw_cache[pinyin].get("qa", [])
                lookup: dict[tuple[str, str], int] = {}
                for idx, rq in enumerate(raw_qa_list):
                    key = (rq["question"], rq["ask_time"])
                    lookup[key] = idx
                raw_index_cache[pinyin] = lookup

            raw_qa_list = raw_cache[pinyin].get("qa", [])
            lookup = raw_index_cache[pinyin]

            # 去掉 conv question 中的 （提问时间：XXXX-XX-XX） 前缀
            conv_q = qa_item["question"]
            if "）" in conv_q:
                conv_q = conv_q.split("）", 1)[-1]

            ask_time = qa_item["ask_time"]
            key = (conv_q, ask_time)

            raw_idx = lookup.get(key)
            if raw_idx is None:
                print(f"[WARN] Unmatched: {qid} (q='{conv_q[:60]}...', ask_time={ask_time})")
                stats["unmatched"] += 1
                continue

            evidence = raw_qa_list[raw_idx].get("evidence", [])
            mapping[qid] = evidence
            stats["total"] += 1

    # 输出
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Mapping built successfully!")
    print(f"  Total question_id → evidence entries: {stats['total']}")
    print(f"  Raw files loaded: {len(raw_cache)}")
    if stats["unmatched"]:
        print(f"  Unmatched: {stats['unmatched']}")
    if stats["missing_raw_file"]:
        print(f"  Missing raw files: {stats['missing_raw_file']}")
    print(f"  Output: {output_path}")

    return mapping


if __name__ == "__main__":
    base = Path(__file__).parent
    conv_path = base / "lifebench_locomo_format" / "lifebench_locomo_conversation_format_v2.0_3380QA.json"
    raw_dir = base / "lifebench_raw"
    output_path = base / "question_id_to_evidence_mapping.json"

    build_mapping(str(conv_path), str(raw_dir), str(output_path))
