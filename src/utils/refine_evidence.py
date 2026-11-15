"""
Evidence refinement script.

Uses an LLM to evaluate evidence quality and filter out irrelevant, redundant,
or meaningless evidence items from the LifeBench evidence mapping.

Usage:
    # Test with 20 QAs, 5 concurrent
    python -m src.utils.refine_evidence --limit 20 --concurrency 5

    # Dry run (preview without writing)
    python -m src.utils.refine_evidence --limit 50 --dry-run

    # Full processing
    python -m src.utils.refine_evidence --concurrency 20

    # Resume from checkpoint
    python -m src.utils.refine_evidence --resume
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- Paths ---
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

CONV_PATH = (
    project_root
    / "datasets"
    / "lifebench_locomo_format"
    / "lifebench_locomo_conversation_format_v2.0_3380QA.json"
)
EVI_MAPPING_PATH = (
    project_root / "datasets" / "lifebench_raw" / "question_id_to_evidence_mapping.json"
)
CHECKPOINT_PATH = project_root / "results" / "refine_checkpoint.json"
REPORT_PATH = project_root / "results" / "refine_evidence_report.json"
OUTPUT_PATH = EVI_MAPPING_PATH  # overwrite

# --- LLM Config ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = "deepseek-v4-pro"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 8192

# --- Constants ---
SMALL_QA_THRESHOLD = 3  # skip QAs with <= this many evidence items
LARGE_BATCH_THRESHOLD = 30  # split into batches if > this many evidence items
CHECKPOINT_INTERVAL = 100  # save progress every N QAs


# ============================================================
# Evidence formatting
# ============================================================


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to max_len chars, adding ellipsis if truncated."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _turn_key(key: str) -> tuple:
    """Sort key for turn labels ('turn 2' before 'turn 10')."""
    m = re.search(r"\d+", key or "")
    return (int(m.group()) if m else 0, str(key))


def _join_list(v) -> str:
    """Render a list/tuple as comma-joined text, else the raw string."""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v or "")


def _location_str(loc) -> str:
    """Render a location dict as a full address (province → poi), best-effort."""
    if isinstance(loc, dict):
        parts = []
        for k in ("province", "city", "district", "streetName", "streetNumber", "poi"):
            val = str(loc.get(k) or "").strip()
            if val:
                parts.append(val)
        return " ".join(parts)
    return str(loc) if loc else ""


def _iter_agent_turns(conv: dict) -> list:
    """Return ordered ``(label, user, assistant)`` turn tuples.

    Flattens the malformed case (seen in the raw data) where a turn dict
    nests another ``turn N`` inside itself, so no turn content is dropped.
    """
    out: List[Tuple[str, dict, dict]] = []

    def visit(d: dict) -> None:
        for tk in sorted(d.keys(), key=_turn_key):
            tv = d[tk]
            if not isinstance(tv, dict):
                continue
            out.append((tk, tv.get("user", {}) or {}, tv.get("assistant", {}) or {}))
            nested = {
                k: v for k, v in tv.items()
                if k not in ("user", "assistant") and isinstance(v, dict)
            }
            for nk in sorted(nested.keys(), key=_turn_key):
                visit({nk: nested[nk]})

    visit(conv)
    return out


def format_evidence(ev: Dict, index: int, include_assistant: bool = False) -> str:
    """Format a single evidence item into a compact multi-line representation.

    Every content-bearing field of ``raw_data`` is surfaced; only internal
    linkage ids (``daily_event_id`` / ``event_id`` / ``phone_id``) are omitted.
    """
    raw = ev.get("raw_data", {})
    rtype = raw.get("type", ev.get("source", "unknown"))
    src = ev.get("source", "unknown")
    date = ev.get("session_date", "")

    if rtype == "call":
        _dir_map = {0: "incoming", 1: "outgoing", 2: "missed"}
        direction = _dir_map.get(raw.get("direction"), raw.get("direction"))
        has_content = raw.get("call_result", "") not in ("", "接通", "未接通")
        tag = "" if has_content else " [METADATA_ONLY]"
        return (
            f"[E{index}] call{tag} | {date} | {raw.get('contactName', '?')} "
            f"({raw.get('phoneNumber', '')}) | {direction} | "
            f"{raw.get('datetime', '')} → {raw.get('datetime_end', '')} | "
            f"result={raw.get('call_result', '')}"
        )

    elif rtype == "sms":
        content = _truncate(raw.get("message_content", ""), 250)
        mt = raw.get("message_type", "")
        if mt == "发送":
            direction = "outgoing"
        elif mt == "接收":
            direction = "incoming"
        else:
            direction = mt or "?"
        return (
            f"[E{index}] sms | {date} | {direction} | {raw.get('contactName', '?')} "
            f"({raw.get('phoneNumber', '')}) | {raw.get('datetime', '')} | {content}"
        )

    elif rtype == "agent_chat":
        conv = raw.get("conversation", {}) or {}
        turns = _iter_agent_turns(conv)
        lines = []
        for tk, user, assistant in turns:
            user_content = _truncate(user.get("content", ""), 600)
            user_action = user.get("action", "")
            user_explain = (user.get("explain", "") or "").strip()
            if user_content:
                lines.append(f"  [{tk}] user({user_action}): {user_content}")
            if user_explain:
                lines.append(f"  [{tk}] user_explain: {_truncate(user_explain, 200)}")
            if include_assistant:
                assistant_content = _truncate(assistant.get("content", ""), 600)
                assistant_action = assistant.get("action", "")
                if assistant_content:
                    lines.append(f"  [{tk}] assistant({assistant_action}): {assistant_content}")
        header_date = raw.get("date") or date
        header = f"[E{index}] agent_chat | {header_date} | {len(turns)} turns"
        if not lines:
            return header + " | (no user content)"
        return header + "\n" + "\n".join(lines)

    elif rtype == "calendar":
        desc = _truncate(raw.get("description", ""), 500)
        return (
            f"[E{index}] calendar | {date} | title={raw.get('title', '')} | "
            f"time={raw.get('start_time', '')} → {raw.get('end_time', '')} | "
            f"datetime={raw.get('datetime', '')} | desc={desc}"
        )

    elif rtype == "note":
        content = _truncate(raw.get("content", ""), 1000)
        return (
            f"[E{index}] note | {date} | {raw.get('datetime', '')} | "
            f"title={raw.get('title', '')} | content={content}"
        )

    elif rtype == "push":
        content = _truncate(raw.get("content", ""), 200)
        has_factual = bool(raw.get("title", "") or raw.get("content", ""))
        tag = "" if has_factual else " [METADATA_ONLY]"
        return (
            f"[E{index}] push{tag} | {date} | {raw.get('datetime', '')} | "
            f"app={raw.get('source', '')} | title={raw.get('title', '')} | "
            f"content={content} | status={raw.get('push_status', '')} | "
            f"jump_path={raw.get('jump_path', '')}"
        )

    elif rtype == "photo":
        caption = _truncate(raw.get("caption", ""), 200)
        loc_str = _location_str(raw.get("location", {}))
        fields = [
            f"[E{index}] photo | {date} | {raw.get('datetime', '')}",
            f"location={loc_str}",
        ]
        title = (raw.get("title", "") or "").strip()
        face = _join_list(raw.get("faceRecognition", ""))
        tags = _join_list(raw.get("imageTag", []))
        ocr = (raw.get("ocrText", "") or "").strip()
        shoot = (raw.get("shoot_mode", "") or "").strip()
        size = (raw.get("image_size", "") or "").strip()
        if title:
            fields.append(f"title={title}")
        if face:
            fields.append(f"faces={face}")
        if tags:
            fields.append(f"tags={tags}")
        if ocr and ocr != "无":
            fields.append(f"ocr={_truncate(ocr, 200)}")
        if shoot:
            fields.append(f"shoot_mode={shoot}")
        if size:
            fields.append(f"size={size}")
        fields.append(f"caption={caption}")
        return " | ".join(fields)

    elif rtype == "chat":
        msgs = raw.get("chat_message", [])
        lines = []
        for mi, m in enumerate(msgs):
            sender = m.get("sender_name", "?")
            content = _truncate(m.get("content", ""), 200)
            lines.append(f"  msg{mi + 1} {sender}: {content}")
        return f"[E{index}] chat | {date} | {len(msgs)} msgs\n" + "\n".join(lines)

    elif rtype == "daily_event":
        title = _truncate(raw.get("title", ""), 200)
        summary = _truncate(raw.get("summarized_info", ""), 200)
        return (
            f"[E{index}] daily_event | {date} | title={title} | summary={summary}"
        )

    elif rtype == "health":
        data_type = raw.get("data_type", raw.get("type", ""))
        summary = _truncate(json.dumps(raw, ensure_ascii=False), 300)
        return f"[E{index}] health | {date} | type={data_type} | {summary}"

    elif rtype == "knowledge":
        content = _truncate(raw.get("content", raw.get("description", "")), 250)
        return (
            f"[E{index}] knowledge | {date} | title={raw.get('title', '')} | "
            f"content={content}"
        )

    elif rtype == "location":
        loc_str = _location_str(raw.get("location", {}))
        return f"[E{index}] location | {date} | {loc_str}"

    elif rtype == "screen":
        app = raw.get("app_name", "")
        return (
            f"[E{index}] screen | {date} | app={app} | "
            f"{_truncate(json.dumps(raw, ensure_ascii=False), 200)}"
        )

    else:
        return (
            f"[E{index}] {rtype} (source={src}) | {date} | "
            f"{_truncate(json.dumps(raw, ensure_ascii=False), 300)}"
        )


# ============================================================
# LLM prompt & call
# ============================================================

SYSTEM_PROMPT = """You are an evidence quality auditor for a long-term memory QA dataset. Each question comes with multiple evidence items from a person's digital footprint (calls, messages, calendar, notes, photos, AI chat, etc.).

Your job: evaluate each evidence item and decide whether to KEEP or REMOVE it. The goal is to remove ONLY clearly useless noise while preserving every item that could help answer the question.

**GUIDING PRINCIPLE — when in doubt, KEEP.** Removing an answer-bearing item makes the question unanswerable (unrecoverable); keeping a noisy item only adds a minor distraction. So err on the side of keeping: remove only when you are confident the item is irrelevant or fully redundant.

**CRITERIA to REMOVE (any one is sufficient):**

1. IRRELEVANT: Contains no information that helps answer the question.
   - Items about different people, dates, or topics unrelated to the question
   - Photos of unrelated scenes (flowers, food, landscapes with no connection)
2. REDUNDANT: The item's facts are entirely covered by another KEPT item AND it adds no extra information. If it carries any additional detail (a different date, person, number, or event), it is NOT redundant — keep it.
3. NO CONTENT: Items with literally no usable info (empty body, no metadata worth extracting). Note that [METADATA_ONLY] on a call still carries contact/time/duration — treat that metadata as factual, don't auto-remove.
4. WRONG CONTEXT: The date or context clearly doesn't match what the question asks about.

**CRITERIA to KEEP (all should apply):**

1. Contains facts that directly help answer the question (names, dates, locations, events, decisions, numbers)
2. Provides unique information not covered by other KEPT items
3. Has specific, meaningful content related to the question's topic, time, or entities

**TYPE-SPECIFIC RULES:**

- **note**: HIGH signal — direct factual records. Almost always keep, remove only if redundant with a more detailed note.
- **calendar**: HIGH signal — structured event info. Keep if event topic matches the question.
- **photo**: Keep only if caption, faceRecognition, location, or datetime answers the question. Remove generic/unrelated photos.
- **sms**: Keep if message content mentions names, events, or facts relevant to the question.
- **agent_chat**: Both USER and assistant messages are shown. The assistant's replies often contain the actual answer (advice, plans, facts the user asked about), so treat them as evidence too. Remove only if the whole conversation is unrelated to the question.
- **call**: Metadata-only, but that metadata (contact, time, duration, direction) is itself a valid fact and can be the answer. Keep if the contact, time, or duration is relevant to the question; remove only if clearly unrelated.
- **push**: System notifications (payment, transport card, app alerts). Often noise, but their title/content/amount can be the answer. Keep if the title/content matches the question; remove only if clearly unrelated.

**IMPORTANT:**
- When in doubt, KEEP. Prefer completeness over minimalism — a slightly noisy kept set is far less harmful than a wrongly removed answer.
- For Unanswerable questions, it's OK to mark all as REMOVE if truly no evidence relates to the question.
- Verify the KEPT set can answer the question. If items are complementary, keep both.
- Never remove ALL items for Answerable questions — keep the most relevant one.

Output ONLY a JSON object (no other text):
{"decisions": [{"id": "E0", "keep": true, "reason": "short reason in English"}, ...]}"""


def _evidence_content_score(ev: Dict) -> int:
    """Estimate factual content richness of an evidence item.

    Higher score = more likely to contain useful information.
    Used as a fallback to decide which item to keep when LLM removes all.
    """
    raw = ev.get("raw_data", {})
    rtype = raw.get("type", "")

    if rtype == "note":
        return len(raw.get("content", "")) + len(raw.get("title", ""))
    elif rtype == "calendar":
        return len(raw.get("description", "")) + len(raw.get("title", ""))
    elif rtype == "agent_chat":
        conv = raw.get("conversation", {})
        score = 0
        for tk in conv:
            turn = conv[tk] or {}
            score += len((turn.get("user") or {}).get("content", ""))
            score += len((turn.get("assistant") or {}).get("content", ""))
        return score
    elif rtype == "sms":
        return len(raw.get("message_content", ""))
    elif rtype == "chat":
        msgs = raw.get("chat_message", [])
        return sum(len(m.get("content", "")) for m in msgs)
    elif rtype == "photo":
        score = len(raw.get("caption", ""))
        if raw.get("faceRecognition"):
            score += 50
        return score
    elif rtype == "call":
        return 10  # low baseline
    elif rtype == "push":
        return 5  # very low baseline
    else:
        return len(json.dumps(raw, ensure_ascii=False))


def _safeguard_keep(
    decisions: List[Dict],
    evidence_items: List[Dict],
    question_types: List[str],
) -> List[Dict]:
    """If LLM removes ALL items for an answerable question, force-keep the best one."""
    is_unanswerable = "Unanswerable" in question_types
    if is_unanswerable:
        return decisions

    kept = [d for d in decisions if d["keep"]]
    if kept:
        return decisions

    # All removed — but question is answerable. Force-keep the most contentful item.
    best_idx = max(range(len(evidence_items)),
                   key=lambda i: _evidence_content_score(evidence_items[i]))
    new_decisions = []
    for d in decisions:
        if d["index"] == best_idx:
            new_decisions.append({
                **d,
                "keep": True,
                "reason": d["reason"] + " [forced keep: answerable QA needs ≥1 item]",
            })
        else:
            new_decisions.append(d)
    return new_decisions


def build_user_prompt(
    question: str,
    answer: str,
    question_types: List[str],
    ask_time: str,
    evidence_texts: List[str],
) -> str:
    """Build the user prompt for a single QA evaluation."""
    parts = [
        "=== QUESTION ===",
        f"Question: {question}",
        f"Expected Answer: {answer}",
        f"Question Types: {', '.join(question_types)}",
        f"Ask Time: {ask_time}",
        "",
        "=== EVIDENCE ITEMS ===",
    ]
    parts.extend(evidence_texts)
    parts.append("")
    parts.append(
        "Evaluate each evidence item. Output ONLY a JSON object: "
        '{"decisions": [{"id": "E0", "keep": true/false, "reason": "..."}, ...]}'
    )
    return "\n".join(parts)


async def evaluate_evidence(
    client: AsyncOpenAI,
    question: str,
    answer: str,
    question_types: List[str],
    ask_time: str,
    evidence_items: List[Dict],
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> Optional[List[Dict]]:
    """Evaluate evidence for a single QA. Returns list of decision dicts."""
    if not evidence_items:
        return []

    # Format evidence — include assistant turns (they often hold the answer)
    evidence_texts = [
        format_evidence(ev, i, include_assistant=True) for i, ev in enumerate(evidence_items)
    ]
    user_prompt = build_user_prompt(
        question, answer, question_types, ask_time, evidence_texts
    )

    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                result = json.loads(raw)
                decisions = result.get("decisions", [])

                # Validate decisions
                valid = []
                for d in decisions:
                    eid = d.get("id", "")
                    keep = d.get("keep", False)
                    reason = d.get("reason", "")
                    # Extract index from "E0", "E1", etc.
                    match = re.match(r"E(\d+)", str(eid))
                    if match:
                        valid.append(
                            {
                                "index": int(match.group(1)),
                                "id": str(eid),
                                "keep": bool(keep),
                                "reason": str(reason),
                            }
                        )

                # Ensure all evidence items have a decision
                expected_indices = set(range(len(evidence_items)))
                decided_indices = {d["index"] for d in valid}
                for idx in expected_indices - decided_indices:
                    valid.append(
                        {
                            "index": idx,
                            "id": f"E{idx}",
                            "keep": True,  # default keep if no decision
                            "reason": "LLM did not evaluate; default keep",
                        }
                    )

                valid.sort(key=lambda x: x["index"])
                valid = _safeguard_keep(valid, evidence_items, question_types)
                return valid

            except (json.JSONDecodeError, KeyError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(
                        f"  [WARN] Failed to parse LLM response after "
                        f"{max_retries} attempts: {e}"
                    )
                    # Default: keep all
                    return [
                        {
                            "index": i,
                            "id": f"E{i}",
                            "keep": True,
                            "reason": "parse error; default keep",
                        }
                        for i in range(len(evidence_items))
                    ]

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  [RETRY] Attempt {attempt + 1} failed: {e}. "
                          f"Waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  [ERROR] LLM call failed after "
                          f"{max_retries} attempts: {e}")
                    return [
                        {
                            "index": i,
                            "id": f"E{i}",
                            "keep": True,
                            "reason": f"llm error; default keep",
                        }
                        for i in range(len(evidence_items))
                    ]

    return None


async def evaluate_large_qa(
    client: AsyncOpenAI,
    question: str,
    answer: str,
    question_types: List[str],
    ask_time: str,
    evidence_items: List[Dict],
    semaphore: asyncio.Semaphore,
) -> Optional[List[Dict]]:
    """Handle large QAs by splitting into batches."""
    batch_size = LARGE_BATCH_THRESHOLD
    all_decisions = []

    for batch_start in range(0, len(evidence_items), batch_size):
        batch_end = min(batch_start + batch_size, len(evidence_items))
        batch = evidence_items[batch_start:batch_end]

        # Re-index for the batch
        batch_with_orig_idx = list(enumerate(batch, start=batch_start))
        batch_items = [item for _, item in batch_with_orig_idx]

        decisions = await evaluate_evidence(
            client, question, answer, question_types,
            ask_time, batch_items, semaphore,
        )

        if decisions is None:
            return None

        # Adjust indices back to original positions
        for d in decisions:
            d["index"] = batch_start + d["index"]
            d["id"] = f"E{d['index']}"

        all_decisions.extend(decisions)

    return all_decisions


# ============================================================
# Main processing
# ============================================================


def load_data() -> Tuple[Dict, Dict]:
    """Load conversation format and evidence mapping.

    Returns:
        (qa_lookup, evidence_mapping)
        qa_lookup: question_id -> {question, answer, question_type, ask_time}
        evidence_mapping: question_id -> [evidence items]
    """
    print(f"Loading conversation format from {CONV_PATH}...")
    with open(CONV_PATH, "r", encoding="utf-8") as f:
        conv_data = json.load(f)

    qa_lookup = {}
    for person in conv_data:
        for qa in person.get("qa", []):
            qid = qa.get("question_id", "")
            if qid:
                # Strip time prefix from question
                question = qa.get("question", "")
                if "）" in question and question.startswith("（"):
                    question = question.split("）", 1)[1]
                qa_lookup[qid] = {
                    "question": question,
                    "answer": qa.get("answer", ""),
                    "question_type": qa.get("question_type", []),
                    "ask_time": qa.get("ask_time", ""),
                }

    print(f"Loading evidence mapping from {EVI_MAPPING_PATH}...")
    with open(EVI_MAPPING_PATH, "r", encoding="utf-8") as f:
        evidence_mapping = json.load(f)

    print(f"Loaded {len(qa_lookup)} QAs, {len(evidence_mapping)} evidence entries")
    return qa_lookup, evidence_mapping


def load_checkpoint() -> Dict[str, List[Dict]]:
    """Load checkpoint if it exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
            return json.load(f)
    return {}


def save_checkpoint(decisions_map: Dict[str, List[Dict]]):
    """Save intermediate results to checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(decisions_map, f, ensure_ascii=False, indent=2)


def save_report(
    decisions_map: Dict[str, List[Dict]],
    evidence_mapping: Dict,
    qa_lookup: Dict,
):
    """Save a summary report of what was removed and why."""
    total_qa = len(decisions_map)
    total_before = 0
    total_after = 0
    removal_reasons = {}
    per_type_stats = {}

    for qid, decisions in decisions_map.items():
        ev_count = len(evidence_mapping.get(qid, []))
        total_before += ev_count
        kept = sum(1 for d in decisions if d["keep"])
        total_after += kept

        q_types = tuple(qa_lookup.get(qid, {}).get("question_type", ["unknown"]))
        if q_types not in per_type_stats:
            per_type_stats[q_types] = {"before": 0, "after": 0, "count": 0}
        per_type_stats[q_types]["before"] += ev_count
        per_type_stats[q_types]["after"] += kept
        per_type_stats[q_types]["count"] += 1

        for d in decisions:
            if not d["keep"]:
                reason = d.get("reason", "unknown")
                removal_reasons[reason] = removal_reasons.get(reason, 0) + 1

    report = {
        "total_qas_processed": total_qa,
        "total_evidence_before": total_before,
        "total_evidence_after": total_after,
        "removed_count": total_before - total_after,
        "removed_pct": (
            round((total_before - total_after) / total_before * 100, 1)
            if total_before > 0
            else 0
        ),
        "removal_reasons_top20": sorted(
            removal_reasons.items(), key=lambda x: -x[1]
        )[:20],
        "per_type_stats": {
            " | ".join(k): v
            for k, v in sorted(
                per_type_stats.items(), key=lambda x: -x[1]["count"]
            )
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Report saved to {REPORT_PATH}")
    print(f"Total QAs processed: {total_qa}")
    print(f"Evidence before: {total_before}")
    print(f"Evidence after:  {total_after}")
    print(f"Removed: {total_before - total_after} "
          f"({report['removed_pct']}%)")
    print(f"\nTop removal reasons:")
    for reason, count in sorted(
        removal_reasons.items(), key=lambda x: -x[1]
    )[:10]:
        print(f"  [{count}] {reason}")


def apply_filtering(
    decisions_map: Dict[str, List[Dict]],
    evidence_mapping: Dict,
    dry_run: bool = False,
):
    """Apply LLM decisions to filter evidence mapping."""
    filtered_mapping = {}
    for qid, evidence_list in evidence_mapping.items():
        if qid not in decisions_map:
            # QA was skipped or had no evidence; keep as-is
            filtered_mapping[qid] = evidence_list
            continue

        decisions = decisions_map[qid]
        # Build a set of indices to keep
        keep_indices = {d["index"] for d in decisions if d["keep"]}
        filtered = [
            ev for i, ev in enumerate(evidence_list) if i in keep_indices
        ]
        filtered_mapping[qid] = filtered

    if not dry_run:
        # Backup the original before overwriting
        backup_path = Path(str(OUTPUT_PATH) + ".bak")
        if OUTPUT_PATH.exists() and not backup_path.exists():
            shutil.copy2(OUTPUT_PATH, backup_path)
            print(f"Backed up original to {backup_path}")

        print(f"Writing filtered evidence mapping to {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(filtered_mapping, f, ensure_ascii=False, indent=2)
        print("Done.")
    else:
        print("[DRY RUN] Would write filtered mapping to "
              f"{OUTPUT_PATH} (skipped)")

    return filtered_mapping


async def process_all(
    qa_lookup: Dict,
    evidence_mapping: Dict,
    concurrency: int = 20,
    limit: int = 0,
    resume: bool = False,
):
    """Main async processing loop."""
    client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    semaphore = asyncio.Semaphore(concurrency)

    # Load checkpoint if resuming
    decisions_map = load_checkpoint() if resume else {}

    # Build task list
    tasks = []
    qids_to_process = []

    for qid, qa_info in qa_lookup.items():
        if qid in decisions_map:
            continue  # already processed

        evidence_items = evidence_mapping.get(qid, [])
        if len(evidence_items) < SMALL_QA_THRESHOLD:
            # Skip small QAs; keep all evidence
            decisions_map[qid] = [
                {
                    "index": i,
                    "id": f"E{i}",
                    "keep": True,
                    "reason": "small QA (<3 evidence); auto keep",
                }
                for i in range(len(evidence_items))
            ]
            continue

        if limit and limit > 0 and len(qids_to_process) >= limit:
            continue  # stop collecting once we hit the limit

        qids_to_process.append(qid)
        is_large = len(evidence_items) > LARGE_BATCH_THRESHOLD
        if is_large:
            task = evaluate_large_qa(
                client,
                qa_info["question"],
                str(qa_info["answer"]),
                qa_info["question_type"],
                qa_info.get("ask_time", ""),
                evidence_items,
                semaphore,
            )
        else:
            task = evaluate_evidence(
                client,
                qa_info["question"],
                str(qa_info["answer"]),
                qa_info["question_type"],
                qa_info.get("ask_time", ""),
                evidence_items,
                semaphore,
            )
        tasks.append((qid, task))

    total = len(tasks)
    skipped = len(qa_lookup) - total
    print(f"\nProcessing {total} QAs ({skipped} skipped/small)...")
    print(f"Concurrency: {concurrency}, Model: {LLM_MODEL}")
    print(f"Small QA threshold: <{SMALL_QA_THRESHOLD} evidence items")
    print(f"Large QA batching: >{LARGE_BATCH_THRESHOLD} evidence items\n")

    start_time = time.time()
    processed = 0
    errors = 0

    # Process in chunks to allow checkpoint saves
    chunk_size = min(CHECKPOINT_INTERVAL, total)
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk = tasks[chunk_start:chunk_end]

        # Run chunk concurrently
        chunk_coros = [task for _, task in chunk]
        results = await asyncio.gather(*chunk_coros, return_exceptions=True)

        for (qid, _), result in zip(chunk, results):
            if isinstance(result, Exception):
                print(f"  [ERROR] {qid}: {result}")
                errors += 1
                # Keep all evidence on error
                ev_items = evidence_mapping.get(qid, [])
                decisions_map[qid] = [
                    {
                        "index": i,
                        "id": f"E{i}",
                        "keep": True,
                        "reason": f"processing error; default keep",
                    }
                    for i in range(len(ev_items))
                ]
            elif result is not None:
                decisions_map[qid] = result
            else:
                # Shouldn't happen, but handle
                ev_items = evidence_mapping.get(qid, [])
                decisions_map[qid] = [
                    {
                        "index": i,
                        "id": f"E{i}",
                        "keep": True,
                        "reason": "null result; default keep",
                    }
                    for i in range(len(ev_items))
                ]

        processed += len(chunk)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(
            f"  Progress: {processed}/{total} ({processed / total * 100:.1f}%) | "
            f"Errors: {errors} | {rate:.1f} QA/s | ETA: "
            f"{(total - processed) / rate:.0f}s"
            if rate > 0
            else f"  Progress: {processed}/{total}"
        )

        # Save checkpoint
        save_checkpoint(decisions_map)
        print(f"  Checkpoint saved ({len(decisions_map)} entries)")

    elapsed = time.time() - start_time
    print(f"\nFinished processing in {elapsed:.1f}s "
          f"({total / elapsed:.1f} QA/s, {errors} errors)")

    return decisions_map


async def main():
    parser = argparse.ArgumentParser(
        description="Refine LifeBench evidence using LLM evaluation"
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Max concurrent LLM calls (default: 20)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of QAs to process (0 = all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview decisions without writing output"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint"
    )

    args = parser.parse_args()

    # Load data
    qa_lookup, evidence_mapping = load_data()

    # Process
    decisions_map = await process_all(
        qa_lookup=qa_lookup,
        evidence_mapping=evidence_mapping,
        concurrency=args.concurrency,
        limit=args.limit,
        resume=args.resume,
    )

    # Save report
    save_report(decisions_map, evidence_mapping, qa_lookup)

    # Apply filtering
    apply_filtering(decisions_map, evidence_mapping, dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
