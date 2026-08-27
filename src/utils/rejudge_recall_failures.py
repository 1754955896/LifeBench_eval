"""
Re-judge recall-evaluator failures WITHOUT touching recall_evaluator.py.

The recall judge in ``recall_evaluator.py`` silently converts LLM failures
(empty response / unparseable JSON / HTTP error) into ``evidence_matches=[]``
+ ``answerable=0`` — byte-identical to a genuine "not covered" verdict. It also
never sends ``max_tokens`` in its request payload.

This script fixes the data, not the code:

  1. Reuses recall_evaluator's prompt + evidence/result formatters (import).
  2. Re-judges every question whose ``evidence_matches`` are *entirely empty*
     (the failure signature) using:
       - an explicit ``max_tokens`` (the field recall_evaluator forgot),
       - a stricter JSON parser (fenced blocks, trailing commas, truncated
         JSON balance, answerable regex fallback),
       - more retries with exponential backoff.
  3. Merges recovered verdicts back into the original checkpoint and recomputes
     the SAME recall metrics, writing ``recall_results_fixed.json``.

Usage:
    python -m src.utils.rejudge_recall_failures --results-dir results/lifebench-graphiti_local
    python -m src.utils.rejudge_recall_failures --results-dir results/lifebench-graphiti_local --dry-run 5
"""

import argparse
import asyncio
import io
import json
import os
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.utils.recall_evaluator import (  # noqa: E402
    JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE,
    _fmt_evidence,
    _fmt_search_result,
    _build_coverage_results,
    load_evidence_mapping,
)


# ---------------------------------------------------------------------------
# Robust JSON parsing (stronger than recall_evaluator._parse_response)
# ---------------------------------------------------------------------------

def _to_int(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _extract_json_candidates(content):
    """Yield candidate JSON strings, most-preferred first."""
    if not content:
        return
    # 1. fenced code block
    for m in re.finditer(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL):
        yield m.group(1)
    # 2. first "{" .. last "}" (tolerates surrounding prose)
    s = content.find("{")
    if s == -1:
        return
    e = content.rfind("}")
    if e > s:
        yield content[s:e + 1]
    else:
        # truncated output with no closing brace -> let _balance_json repair it
        yield content[s:]


def _balance_json(text):
    """Append the missing closing brackets for a truncated JSON string.

    Scans respecting string literals + backslash escapes, so brackets inside
    quoted strings are not miscounted.
    """
    stack = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    return text + "".join(reversed(stack))


def _repair_json(text):
    """Try progressively more aggressive repairs, returning a parsed dict or None."""
    t = text.strip()
    attempts = [
        t,                                              # as-is
        re.sub(r",\s*([}\]])", r"\1", t),               # trailing comma before }/]
        re.sub(r",\s*$", "", t),                        # trailing comma at end
        _balance_json(t),                               # truncated -> balance
        _balance_json(re.sub(r",\s*([}\]])", r"\1", t)),  # comma strip + balance
    ]
    # strip trailing excess closers (e.g. "}}" -> "}") then re-balance
    cur = t
    for _ in range(3):
        m = re.match(r"(.*?)[}\]]+\s*$", cur)
        if not m:
            break
        cur = m.group(1).rstrip()
        attempts.append(cur)
        attempts.append(_balance_json(cur))
    for a in attempts:
        if not a:
            continue
        try:
            return json.loads(a)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _parse_answerable(value, content):
    """Coerce answerable to 0/1. Lenient about bool/int/str, with regex fallback."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        s = value.strip().lower()
        return 1 if s in ("1", "true", "yes", "是") else 0
    m = re.search(r'answerable["\s:]+([01])', content, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _coerce(obj, expected_ev, expected_results, content):
    """Dict -> (evidence_matches, answerable) or None if invalid."""
    matches_raw = obj.get("evidence_matches")
    if not isinstance(matches_raw, list):
        return None

    evidence_matches = []
    for entry in matches_raw:
        idxs = []
        if isinstance(entry, list):
            for v in entry:
                n = _to_int(v)
                if n is None:
                    continue
                idx = n - 1  # prompt uses 1-based numbering
                if 0 <= idx < expected_results:
                    idxs.append(idx)
        # dedupe preserving order
        seen, deduped = set(), []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        evidence_matches.append(deduped)

    while len(evidence_matches) < expected_ev:
        evidence_matches.append([])
    evidence_matches = evidence_matches[:expected_ev]

    answerable = _parse_answerable(obj.get("answerable"), content)
    return evidence_matches, answerable


def _parse_judge_json(content, expected_ev, expected_results):
    for cand in _extract_json_candidates(content):
        obj = _repair_json(cand)
        if obj is not None:
            parsed = _coerce(obj, expected_ev, expected_results, cand)
            if parsed is not None:
                return parsed
    return None


# ---------------------------------------------------------------------------
# Judge with max_tokens + retry (the fixed version)
# ---------------------------------------------------------------------------

def _load_config():
    return {
        "model": os.getenv("RECALL_JUDGE_MODEL") or os.getenv("LLM_MODEL", "deepseek-v4-pro"),
        "api_key": os.getenv("RECALL_JUDGE_API_KEY") or os.getenv("LLM_API_KEY", ""),
        "base_url": os.getenv("RECALL_JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "max_tokens": int(os.getenv("RECALL_JUDGE_MAX_TOKENS") or os.getenv("LLM_MAX_TOKENS", "4096")),
        "temperature": float(os.getenv("RECALL_JUDGE_TEMPERATURE") or os.getenv("LLM_TEMPERATURE", "0.0")),
    }


class RobustJudge:
    """Recall judge that sends max_tokens and recovers malformed JSON."""

    def __init__(self, config, max_retries=8):
        self.model = config["model"]
        self.api_key = config["api_key"]
        self.base_url = config["base_url"].rstrip("/")
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        self.max_retries = max_retries
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            conn = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=360)
            self._session = aiohttp.ClientSession(connector=conn, timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def judge(self, question, reference_answer, evidence_items, search_results):
        """Return {evidence_matches, answerable, status, reason}."""
        result_count = len(search_results)
        evidence_text = "\n\n".join(_fmt_evidence(e, i) for i, e in enumerate(evidence_items))
        results_text = "\n".join(_fmt_search_result(r, i) for i, r in enumerate(search_results))

        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            evidence_count=len(evidence_items),
            evidence_text=evidence_text,
            result_count=result_count,
            results_text=results_text,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        last_reason = "unknown"
        for attempt in range(1, self.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    resp.raise_for_status()
                    data = await resp.json()

                content = data["choices"][0]["message"]["content"] or ""
                if not content.strip():
                    last_reason = "empty response"
                else:
                    parsed = _parse_judge_json(content, len(evidence_items), result_count)
                    if parsed is not None:
                        evidence_matches, answerable = parsed
                        return {
                            "evidence_matches": evidence_matches,
                            "answerable": answerable,
                            "status": "ok",
                            "reason": "",
                        }
                    last_reason = "unparseable"
            except Exception as exc:
                last_reason = f"{type(exc).__name__}: {str(exc)[:100]}"

            if attempt < self.max_retries:
                await asyncio.sleep(min(2 ** attempt, 30))

        return {
            "evidence_matches": [[] for _ in evidence_items],
            "answerable": 0,
            "status": "failed",
            "reason": last_reason,
        }


# ---------------------------------------------------------------------------
# Metric recomputation (same formulas as recall_evaluator.evaluate_recall)
# ---------------------------------------------------------------------------

def _summarize(summary):
    per_question = summary["per_question"]
    n_q = len(per_question)

    total_ev = sum(pq["num_evidence"] for pq in per_question)
    covered_ev = sum(pq["covered_count"] for pq in per_question)
    coverage_rate = sum(pq["coverage_rate"] for pq in per_question) / n_q if n_q else 0.0
    coverage_rate_micro = covered_ev / total_ev if total_ev else 0.0

    answerable_count = sum(1 for pq in per_question if pq["answerable"] == 1)
    answerable_rate = answerable_count / n_q if n_q else 0.0

    fully_covered_answerable = sum(
        1 for pq in per_question
        if pq["answerable"] == 1 and pq["covered_count"] == pq["num_evidence"]
    )
    joint_rate = fully_covered_answerable / n_q if n_q else 0.0

    total_results = sum(pq["num_results"] for pq in per_question)
    relevant_results = sum(pq["relevant_count"] for pq in per_question)
    precision = relevant_results / total_results if total_results else 0.0
    redundancy = 1.0 - precision
    recall = coverage_rate
    avg_results = total_results / n_q if n_q else 0.0
    avg_relevant = relevant_results / n_q if n_q else 0.0

    total_tokens = sum(pq["total_tokens"] for pq in per_question)
    avg_tokens_per_unit = total_tokens / total_results if total_results else 0.0
    avg_tokens_per_question = total_tokens / n_q if n_q else 0.0

    covered_at_5 = sum(pq["covered_at_5"] for pq in per_question)
    covered_at_20 = sum(pq["covered_at_20"] for pq in per_question)
    recall_at_5 = sum(pq["recall_at_5"] for pq in per_question) / n_q if n_q else 0.0
    recall_at_20 = sum(pq["recall_at_20"] for pq in per_question) / n_q if n_q else 0.0

    relevant_at_5 = sum(pq["relevant_at_5"] for pq in per_question)
    relevant_at_20 = sum(pq["relevant_at_20"] for pq in per_question)
    precision_at_5 = sum(pq["precision_at_5"] for pq in per_question) / n_q if n_q else 0.0
    precision_at_20 = sum(pq["precision_at_20"] for pq in per_question) / n_q if n_q else 0.0

    summary["coverage_rate"] = coverage_rate
    summary["coverage_rate_micro"] = coverage_rate_micro
    summary["recall"] = recall
    summary["recall_at_5"] = recall_at_5
    summary["recall_at_20"] = recall_at_20
    summary["answerable_rate"] = answerable_rate
    summary["covered_and_answerable_rate"] = joint_rate
    summary["precision"] = precision
    summary["precision_at_5"] = precision_at_5
    summary["precision_at_20"] = precision_at_20
    summary["redundancy"] = redundancy
    summary["avg_results_per_question"] = avg_results
    summary["avg_relevant_per_question"] = avg_relevant
    summary["avg_tokens_per_memory_unit"] = avg_tokens_per_unit
    summary["avg_tokens_per_question"] = avg_tokens_per_question
    return summary


def _write_checkpoint(path, completed):
    tmp = str(path) + ".tmp"
    entries = [
        {
            "question_id": qid,
            "evidence_matches": v.get("evidence_matches", []),
            "answerable": v.get("answerable", 0),
            "status": v.get("status", "ok"),
            "reason": v.get("reason", ""),
        }
        for qid, v in completed.items()
    ]
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"format_version": 4, "completed": entries}, f, ensure_ascii=False)
    os.replace(tmp, str(path))


def _print_summary(summary, before):
    print()
    print("=" * 64)
    print("Re-judged Recall Results  (before -> after)")
    print("=" * 64)
    rows = [
        ("Recall (= macro coverage)", "recall"),
        ("Answerable rate", "answerable_rate"),
        ("Covered AND answerable", "covered_and_answerable_rate"),
        ("Precision (result purity)", "precision"),
        ("Recall@5", "recall_at_5"),
        ("Recall@20", "recall_at_20"),
    ]
    for label, key in rows:
        new = summary.get(key)
        old = before.get(key) if before else None
        if old is None:
            print(f"  {label:28s}: {new*100:6.2f}%")
        else:
            print(f"  {label:28s}: {old*100:6.2f}% -> {new*100:6.2f}%   (Δ {new-old:+.4f})")
    print(f"  questions judged: {len(summary['per_question'])}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(results_dir, concurrency, max_retries, dry_run):
    base = Path(results_dir)
    results_path = base / "search_results.json"
    checkpoint_path = base / "recall_checkpoint.json"
    output_path = base / "recall_results_fixed.json"
    fixed_ckpt_path = base / "recall_checkpoint_fixed.json"

    if not results_path.exists():
        raise FileNotFoundError(f"search_results.json not found at {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        search_data = json.load(f)

    evidence_lookup = load_evidence_mapping(project_root / "datasets" / "lifebench_raw")

    conv_path = project_root / "datasets" / "lifebench_locomo_format" / \
                "lifebench_locomo_conversation_format_v2.0_3380QA.json"
    with open(conv_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    answer_map = {}
    for person in conv_data:
        for qa in person["qa"]:
            answer_map[qa["question_id"]] = qa.get("answer", "")

    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ck = json.load(f)
        for entry in ck.get("completed", []):
            completed[entry["question_id"]] = {
                "evidence_matches": entry.get("evidence_matches", []),
                "answerable": entry.get("answerable", 0),
                "status": entry.get("status", "ok"),
            }

    items = []
    for sr in search_data:
        qid = sr["question_id"]
        ev = evidence_lookup.get(qid, [])
        if ev:
            items.append((sr, ev))

    def is_empty(sr):
        if len(sr.get("results", [])) == 0:
            return False  # genuine no-retrieval, nothing to judge
        entry = completed.get(sr["question_id"])
        if entry is None:
            return True
        return not any(entry.get("evidence_matches", []))

    targets = [(sr, ev) for sr, ev in items if is_empty(sr)]
    if dry_run:
        targets = targets[:dry_run]

    print(f"Total with evidence: {len(items)}")
    print(f"Empty-match questions to re-judge: {len(targets)}")

    if not targets:
        print("Nothing to re-judge.")
        return

    config = _load_config()
    print(f"Judge: {config['model']} @ {config['base_url']}  (max_tokens={config['max_tokens']})")
    print(f"Concurrency: {concurrency}  |  retries/judge: {max_retries}")

    judge = RobustJudge(config, max_retries=max_retries)
    sem = asyncio.Semaphore(concurrency)

    recovered = 0
    still_failed = 0
    failed_list = []

    pbar = tqdm(total=len(targets), desc="Re-judging")

    async def rejudge_one(sr, ev):
        nonlocal recovered, still_failed
        async with sem:
            qid = sr["question_id"]
            out = await judge.judge(
                question=sr.get("query", ""),
                reference_answer=answer_map.get(qid, ""),
                evidence_items=ev,
                search_results=sr.get("results", []),
            )
            if out["status"] == "ok":
                completed[qid] = {
                    "evidence_matches": out["evidence_matches"],
                    "answerable": out["answerable"],
                    "status": "ok",
                }
                recovered += 1
            else:
                completed[qid] = {
                    "evidence_matches": out["evidence_matches"],
                    "answerable": 0,
                    "status": "failed",
                    "reason": out["reason"],
                }
                still_failed += 1
                failed_list.append((qid, out["reason"]))
            pbar.update(1)

    if targets:
        await asyncio.gather(*[rejudge_one(sr, ev) for sr, ev in targets])

    pbar.close()
    await judge.close()

    print(f"\nValid verdicts obtained: {recovered}  |  still failing: {still_failed}")
    if failed_list:
        print("Still-failing questions:")
        for qid, reason in failed_list[:20]:
            print(f"  {qid}: {reason}")

    # Recompute with the same formulas as recall_evaluator.
    summary = _build_coverage_results(items, completed)
    summary = _summarize(summary)
    summary["rejudge_metadata"] = {
        "num_targeted": len(targets),
        "num_recovered": recovered,
        "num_still_failed": still_failed,
        "failed_questions": [{"question_id": q, "reason": r} for q, r in failed_list],
    }

    before = {}
    original_results = base / "recall_results.json"
    if original_results.exists():
        with open(original_results, "r", encoding="utf-8") as f:
            before = json.load(f)

    _print_summary(summary, before)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_checkpoint(fixed_ckpt_path, completed)

    print(f"Fixed recall stats -> {output_path}")
    print(f"Fixed checkpoint   -> {fixed_ckpt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-judge recall-judge failures with stronger JSON parsing + max_tokens"
    )
    parser.add_argument("--results-dir", default="results/lifebench-graphiti_local")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--max-retries", type=int, default=8,
                        help="Retries per judge LLM call before marking failed")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Only re-judge first N empty-match questions (smoke test)")
    args = parser.parse_args()
    asyncio.run(run(args.results_dir, args.concurrency, args.max_retries, args.dry_run))


if __name__ == "__main__":
    main()