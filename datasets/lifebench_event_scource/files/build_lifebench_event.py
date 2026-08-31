#!/usr/bin/env python3
"""由 daily_event.json + lifebench_raw.json 生成事件粒度的 lifebench_event.json。

用法（在任意目录下执行均可，默认路径基于脚本所在位置解析）：

    python build_lifebench_event.py
    python build_lifebench_event.py -e daily_event.json -r lifebench_raw.json -o ../lifebench_event.json
    python build_lifebench_event.py --speaker 孙雨薇 --sample-id 孙雨薇

转换规则：
    1. 保留 lifebench_raw.json 的样本骨架（sample_id / speaker_a / speaker_b / qa / 各 session 日期）。
    2. 每个 session_N 的内容由「按天日记」替换为「事件记录」：从 daily_event.json 取出该天全部事件，
       每条生成 { speaker, dia_id, text }。
    3. dia_id = "{日期}_event{编号}"，编号按天从 1 重新计数。
    4. text   = "{speaker}在{时间区间}的活动记录：{description}"，时间区间具体到秒；
       同一事件含多个时段的用「、」连接。
    5. 某天无事件（如 2025-10-06）时，对应 session 为空列表。
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_EVENTS = HERE / "daily_event.json"
DEFAULT_RAW = HERE / "lifebench_raw.json"
DEFAULT_OUT = HERE.parent / "lifebench_event.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_events_by_date(events):
    """按日期分组事件；同一事件跨多个时段（同天）只计一次，按 distinct 日期去重。"""
    by_date = defaultdict(list)
    for e in events:
        # date 形如 "2025-01-01 08:15:00至2025-01-01 09:25:00"，取日期部分
        for d in sorted({r.split(" ")[0] for r in e["date"]}):
            by_date[d].append(e)
    for d in by_date:
        by_date[d].sort(key=lambda e: int(e["event_id"]))
    return by_date


def session_numbers(conversation):
    """从 conversation 中提取 session 序号（按数字排序）。"""
    nums = set()
    for k in conversation:
        m = re.fullmatch(r"session_(\d+)_date_time", k)
        if m:
            nums.add(int(m.group(1)))
    return sorted(nums)


def build_event_session(date: str, day_events, speaker: str):
    """将某天的事件列表转为事件粒度的 session 记录。"""
    return [
        {
            "speaker": speaker,
            "dia_id": f"{date}_event{seq}",
            "text": f"{speaker}在{'、'.join(e['date'])}的活动记录：{e['description']}",
        }
        for seq, e in enumerate(day_events, start=1)
    ]


def convert_sample(sample, by_date, speaker=None):
    """转换单个样本。speaker 缺省时取 sample_id / speaker_a。"""
    speaker = speaker or sample.get("sample_id") or ""
    conv = sample["conversation"]
    new_conv = {
        "speaker_a": conv.get("speaker_a", speaker),
        "speaker_b": conv.get("speaker_b", ""),
    }
    for n in session_numbers(conv):
        dt_key = f"session_{n}_date_time"
        sess_key = f"session_{n}"
        dt = conv.get(dt_key, "")
        new_conv[dt_key] = dt
        new_conv[sess_key] = build_event_session(dt, by_date.get(dt, []), speaker)

    return {
        "sample_id": sample.get("sample_id"),
        "conversation": new_conv,
        "qa": sample.get("qa", []),
    }


def main():
    parser = argparse.ArgumentParser(
        description="由 daily_event.json + lifebench_raw.json 生成事件粒度样本。"
    )
    parser.add_argument("-e", "--events", type=Path, default=DEFAULT_EVENTS,
                        help="源事件数据路径（默认 files/daily_event.json）")
    parser.add_argument("-r", "--raw", type=Path, default=DEFAULT_RAW,
                        help="原始样本路径（默认 files/lifebench_raw.json）")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                        help="输出路径（默认 ../lifebench_event.json）")
    parser.add_argument("--speaker", type=str, default=None,
                        help="覆盖 speaker 名称（默认取样本的 sample_id）")
    parser.add_argument("--sample-id", type=str, default=None,
                        help="仅转换指定 sample_id 的样本（默认转换全部）")
    parser.add_argument("--indent", type=int, default=2, help="JSON 缩进空格数（默认 2）")
    args = parser.parse_args()

    events = load_json(args.events)
    raw = load_json(args.raw)

    by_date = group_events_by_date(events)

    samples = raw if isinstance(raw, list) else [raw]
    if args.sample_id:
        samples = [s for s in samples if s.get("sample_id") == args.sample_id]
        if not samples:
            raise SystemExit(f"未找到 sample_id={args.sample_id!r} 的样本")

    result = [convert_sample(s, by_date, args.speaker) for s in samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=args.indent)

    # 摘要
    total_events = sum(
        len(item["conversation"][k])
        for item in result
        for k in item["conversation"]
        if re.fullmatch(r"session_\d+", k)
    )
    empty_sessions = sum(
        1
        for item in result
        for k in item["conversation"]
        if re.fullmatch(r"session_\d+", k) and not item["conversation"][k]
    )
    print(f"样本数：{len(result)}")
    print(f"事件记录总数：{total_events}")
    print(f"空 session 数（当日无事件）：{empty_sessions}")
    print(f"已写入：{args.output}")


if __name__ == "__main__":
    main()