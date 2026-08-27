#!/usr/bin/env python3
"""Find QA entries whose ``ask_time`` changed between the offline and locomo datasets.

The offline dataset rewrites ``ask_time`` of a subset of questions to ``2025-12-31``
(end-of-year "now"), so offline evaluation sees the full history. The locomo-format
dataset keeps the original ask time. This script diffs the two files by ``question_id``
and records every changed entry.

Usage:
    python -m src.utils.extract_asktime_changes \
        --offline datasets/lifebench_offline/lifebench_offline.json \
        --locomo datasets/lifebench_locomo_format/lifebench_locomo_conversation_format_v2.0_3380QA.json \
        --output datasets/lifebench_offline/asktime_changed_qa.json
"""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List


def load_qa_map(path: Path) -> Dict[str, Dict]:
    """Load a dataset file and map question_id -> (sample_id, qa_dict)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    qa_map: Dict[str, Dict] = {}
    for sample in data:
        sample_id = sample["sample_id"]
        for qa in sample["qa"]:
            qa_map[qa["question_id"]] = {
                "sample_id": sample_id,
                "qa": qa,
            }
    return qa_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        type=Path,
        default=Path("datasets/lifebench_offline/lifebench_offline.json"),
        help="Offline dataset JSON (ask_time rewritten).",
    )
    parser.add_argument(
        "--locomo",
        type=Path,
        default=Path(
            "datasets/lifebench_locomo_format/"
            "lifebench_locomo_conversation_format_v2.0_3380QA.json"
        ),
        help="Locomo-format dataset JSON (original ask_time).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/lifebench_offline/asktime_changed_qa.json"),
        help="Output JSON recording changed QA entries.",
    )
    args = parser.parse_args()

    offline_map = load_qa_map(args.offline)
    locomo_map = load_qa_map(args.locomo)

    changed: List[Dict] = []
    for qid, offline_entry in offline_map.items():
        if qid not in locomo_map:
            continue
        locomo_qa = locomo_map[qid]["qa"]
        offline_qa = offline_entry["qa"]
        locomo_time = locomo_qa.get("ask_time")
        offline_time = offline_qa.get("ask_time")
        if locomo_time == offline_time:
            continue

        changed.append(
            {
                "question_id": qid,
                "sample_id": offline_entry["sample_id"],
                "locomo_ask_time": locomo_time,
                "offline_ask_time": offline_time,
                "category": offline_qa.get("category"),
                "question_type": offline_qa.get("question_type"),
                "question": offline_qa.get("question"),
                "answer": offline_qa.get("answer"),
            }
        )

    # Sort by sample_id then question_id for stable output.
    changed.sort(key=lambda x: (x["sample_id"], x["question_id"]))

    record = {
        "metadata": {
            "description": (
                "QA entries whose ask_time differs between the offline dataset and "
                "the locomo-format dataset. Offline rewrites ask_time to 2025-12-31."
            ),
            "offline_dataset": str(args.offline),
            "locomo_dataset": str(args.locomo),
            "generated_at": date.today().isoformat(),
            "total_qa": len(offline_map),
            "changed_count": len(changed),
        },
        "changed_qa": changed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"total QA: {len(offline_map)}")
    print(f"changed ask_time: {len(changed)}")
    print(f"written to: {args.output}")


if __name__ == "__main__":
    main()