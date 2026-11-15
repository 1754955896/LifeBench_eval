#!/usr/bin/env python3
"""Build lifebench_locomo_3people_v2 from the v2.0 conversation-format source.

Extracts three persons — sunyuwei (孙雨薇), fenghaoran (冯浩然), yuxiaowei (于晓薇) —
from datasets/lifebench_locomo_format/lifebench_locomo_conversation_format_v2.0_3380QA.json
and writes them to lifebench_locomo_3people_v2.json.

sample_id follows the existing lifebench_locomo_3people convention (pinyin initials):
  孙雨薇 (sunyuwei)  -> syw
  冯浩然 (fenghaoran) -> fhr
  于晓薇 (yuxiaowei)  -> yxw
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "datasets" / "lifebench_locomo_format" / "lifebench_locomo_conversation_format_v2.0_3380QA.json"
OUTPUT = REPO / "datasets" / "lifebench_locomo_3people" / "lifebench_locomo_3people_v2.json"

# speaker_a -> target sample_id, in output order (user-specified order).
SELECTION = [
    ("孙雨薇", "syw"),   # sunyuwei
    ("冯浩然", "fhr"),   # fenghaoran
    ("于晓薇", "yxw"),   # yuxiaowei
]


def main() -> None:
    with open(SOURCE, encoding="utf-8") as f:
        source = json.load(f)

    by_speaker = {item["conversation"]["speaker_a"]: item for item in source}

    out = []
    for speaker, sample_id in SELECTION:
        item = by_speaker[speaker]
        out.append({
            "sample_id": sample_id,
            "conversation": item["conversation"],
            "qa": item["qa"],
        })
        print(f"  {speaker:6s} -> {sample_id:5s}  qa={len(item['qa'])}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {OUTPUT}")
    print(f"Total persons: {len(out)}, total QA: {sum(len(x['qa']) for x in out)}")


if __name__ == "__main__":
    main()
