#!/usr/bin/env python3
"""从 3380QA 抽取指定人物的原始样本，生成 files/lifebench_raw_{pinyin}.json（每人一份，列表含 1 个 sample）。

默认三人：孙雨薇 / 于晓薇 / 冯浩然。
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent          # files/
SRC = HERE.parent.parent / "lifebench_locomo_format" / "lifebench_locomo_conversation_format_v2.0_3380QA.json"

PERSONS = {
    "孙雨薇": "sunyuwei",
    "于晓薇": "yuxiaowei",
    "冯浩然": "fenghaoran",
}


def main() -> None:
    data = json.load(open(SRC, encoding="utf-8"))
    by_id = {p["sample_id"]: p for p in data}

    missing = [name for name in PERSONS if name not in by_id]
    if missing:
        raise SystemExit(f"3380QA 中找不到: {missing}")

    for name, pinyin in PERSONS.items():
        out = HERE / f"lifebench_raw_{pinyin}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump([by_id[name]], f, ensure_ascii=False, indent=2)
        print(f"  {name:6s} qa={len(by_id[name]['qa']):>4} -> {out.name}")

    print(f"Done. 共 {len(PERSONS)} 人 -> {HERE}")


if __name__ == "__main__":
    main()