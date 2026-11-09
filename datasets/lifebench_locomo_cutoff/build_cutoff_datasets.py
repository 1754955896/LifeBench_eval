#!/usr/bin/env python3
"""Build time-truncated variants of the v2.0 conversation-format dataset.

For a cutoff date D:
  - keep every person (sample_id / speakers unchanged)
  - keep conversation sessions whose date is <= D (inclusive)
  - keep only QA pairs whose ask_time == D

Writes one file per cutoff to this directory.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "datasets" / "lifebench_locomo_format" / "lifebench_locomo_conversation_format_v2.0_3380QA.json"
OUTPUT_DIR = Path(__file__).resolve().parent

CUTOFFS = ["2025-03-15", "2025-06-15", "2025-09-15"]
SESSION_DATE_RE = re.compile(r"session_(\d+)_date_time")


def truncate_sample(item: dict, cutoff: str) -> dict:
    conv = item["conversation"]

    # Which session numbers survive the cutoff.
    kept_nums = set()
    for key, value in conv.items():
        m = SESSION_DATE_RE.fullmatch(key)
        if m and isinstance(value, str) and value[:10] <= cutoff:
            kept_nums.add(m.group(1))

    new_conv = {}
    for key, value in conv.items():
        m = SESSION_DATE_RE.fullmatch(key)
        if m:
            if m.group(1) in kept_nums:
                new_conv[key] = value
            continue
        m = re.fullmatch(r"session_(\d+)", key)
        if m:
            if m.group(1) in kept_nums:
                new_conv[key] = value
            continue
        # speaker_a / speaker_b and any other metadata
        new_conv[key] = value

    new_qa = [qa for qa in item["qa"] if qa.get("ask_time") == cutoff]

    return {
        "sample_id": item["sample_id"],
        "conversation": new_conv,
        "qa": new_qa,
    }


def validate(samples: list, cutoff: str) -> None:
    for item in samples:
        conv = item["conversation"]
        dates = [v for k, v in conv.items() if SESSION_DATE_RE.fullmatch(k)]
        assert dates, f"{item['sample_id']}: no sessions kept"
        assert max(dates) <= cutoff, f"{item['sample_id']}: session after cutoff"
        assert all(qa.get("ask_time") == cutoff for qa in item["qa"]), "QA ask_time mismatch"

        kept_dia = {
            msg["dia_id"]
            for key, value in conv.items()
            if re.fullmatch(r"session_\d+", key)
            for msg in value
        }
        for qa in item["qa"]:
            for ev in qa.get("evidence", []):
                assert any(d == ev or d.endswith("_" + ev) for d in kept_dia), (
                    f"{item['sample_id']}/{qa['question_id']}: evidence {ev} "
                    "not found in kept sessions"
                )


def main() -> None:
    with open(SOURCE, encoding="utf-8") as f:
        source = json.load(f)

    for cutoff in CUTOFFS:
        out = [truncate_sample(item, cutoff) for item in source]
        validate(out, cutoff)

        path = OUTPUT_DIR / f"lifebench_locomo_cutoff_{cutoff}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        n_sessions = sum(
            len([k for k in x["conversation"] if re.fullmatch(r"session_\d+", k)])
            for x in out
        )
        n_qa = sum(len(x["qa"]) for x in out)
        print(f"{path.name}: persons={len(out)} sessions={n_sessions} qa={n_qa}")
        print("  per person:", {x["sample_id"]: len(x["qa"]) for x in out})


if __name__ == "__main__":
    main()
