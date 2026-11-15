#!/usr/bin/env python3
"""Convert QAs from locomo10.json to smoke format."""

import json
import re
from pathlib import Path
from datetime import datetime

def determine_question_type(category):
    """Map category to question_type."""
    mapping = {
        1: "topic retrieval",
        2: "factual recall",
        3: "inference",
        4: "summary",
        5: "adversarial"
    }
    return mapping.get(category, "factual recall")

def parse_session_date(date_str):
    """Parse date string like '1:56 pm on 8 May, 2023' to '2023-05-08'."""
    # Format: "1:56 pm on 8 May, 2023" or "1:14 pm on 25 May, 2023"
    try:
        dt = datetime.strptime(date_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d")
    except:
        return ""

def dia_id_to_session(dia_id):
    """Convert dia_id like 'D1:3' to 'session_1'."""
    if not dia_id:
        return dia_id
    match = re.search(r'D(\d+):', dia_id)
    if match:
        session_num = match.group(1)
        return f"session_{session_num}"
    return dia_id

def build_session_date_map(conversation):
    """Build mapping from session name to date string like '2023-05-08'."""
    date_map = {}
    for key, value in conversation.items():
        if key.endswith("_date_time") and isinstance(value, str):
            session_name = key.replace("_date_time", "")
            date_map[session_name] = parse_session_date(value)
    return date_map

def prepend_date_to_dia_id(dia_id, session_date_map):
    """Prepend date to dia_id like 'D1:1' -> '2023-05-08_D1:1'."""
    if not dia_id:
        return dia_id
    # Extract session number from dia_id like "D1:1" -> 1
    match = re.search(r'D(\d+):', dia_id)
    if match:
        session_num = match.group(1)
        session_name = f"session_{session_num}"
        if session_name in session_date_map:
            date_str = session_date_map[session_name]
            if date_str:
                return f"{date_str}_{dia_id}"
    return dia_id

def convert_conversation(conversation):
    """Convert conversation to smoke format with full dia_ids."""
    session_date_map = build_session_date_map(conversation)

    result = {}
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            # Prepend date to dia_id in each utterance
            for utterance in value:
                if "dia_id" in utterance:
                    utterance["dia_id"] = prepend_date_to_dia_id(utterance["dia_id"], session_date_map)
            result[key] = value
        else:
            result[key] = value
    return result

def convert_qa(qa, question_id):
    """Convert a single QA to smoke format."""
    # Handle both regular and adversarial QAs
    if "adversarial_answer" in qa:
        answer = qa.get("adversarial_answer", qa.get("answer", ""))
    else:
        answer = qa.get("answer", "")

    # Convert evidence from dia_ids to session references
    evidence = []
    for e in qa.get("evidence", []):
        evidence.append(dia_id_to_session(e))

    return {
        "question": qa["question"],
        "answer": answer,
        "evidence": evidence,
        "category": [qa["category"]] if isinstance(qa["category"], int) else qa["category"],
        "question_type": [determine_question_type(qa["category"])],
        "question_id": question_id,
        "ask_time": "2025-05-08",
        "score_points": [
            {
                "description": f"Correctly identified {answer}",
                "score": 5.0
            }
        ]
    }

def main():
    # Read locomo10.json
    locomo_path = Path(__file__).parent / ".." / "locomo" / "locomo10.json"
    with open(locomo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert all samples
    converted_samples = []
    for sample_idx in range(len(data)):
        sample = data[sample_idx]
        sample_id = sample.get("sample_id", f"sample-{sample_idx}")

        # Convert conversation with full dia_ids
        conversation = sample.get("conversation", {})
        converted_conversation = convert_conversation(conversation)

        # Convert QAs
        qas = sample["qa"]
        converted_qas = []
        for qa_idx, qa in enumerate(qas):
            question_id = f"{sample_id}_qa{qa_idx+1}"
            converted_qa = convert_qa(qa, question_id)
            converted_qas.append(converted_qa)

        # Build converted sample
        converted_sample = {
            "sample_id": sample_id,
            "conversation": converted_conversation,
            "qa": converted_qas
        }
        converted_samples.append(converted_sample)

    # Backup original locomo10.json before overwriting
    locomo_path_new = Path(__file__).parent / ".." / "locomo" / "locomo10.json"
    locomo_raw_path = Path(__file__).parent / ".." / "locomo" / "locomo10_raw.json"

    # Copy original to _raw backup
    with open(locomo_path_new, "r", encoding="utf-8") as f:
        original_data = f.read()
    with open(locomo_raw_path, "w", encoding="utf-8") as f:
        f.write(original_data)
    print(f"Saved original backup to {locomo_raw_path}")

    # Overwrite locomo10.json with converted data
    with open(locomo_path_new, "w", encoding="utf-8") as f:
        json.dump(converted_samples, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted_samples)} samples with {sum(len(s['qa']) for s in converted_samples)} total QAs to {locomo_path_new}")

if __name__ == "__main__":
    main()
