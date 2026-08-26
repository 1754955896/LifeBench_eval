"""
Re-judge judge-failure questions from a direct-evidence baseline run.

The LLMJudge in ``src/evaluators/llm_judge.py`` omits ``max_tokens`` from its
request payload. On deepseek-v4-flash this yields empty responses for questions
with long golden answers / many score points, which the judge silently turns
into ``is_correct=False`` (reasoning == "empty response").

This script re-runs the judge for ONLY those failed questions, with an explicit
``max_tokens``, then merges the recovered verdicts back and recomputes the
aggregate metrics.

Usage:
    python -m src.utils.rejudge_failures --results-dir results/lifebench-direct_evidence
    python -m src.utils.rejudge_failures --results-dir results/lifebench-direct_evidence --dry-run 5
"""
import argparse
import asyncio
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import aiohttp
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.evaluators.llm_judge import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    _normalize_types,
    _render_score_points,
    _weighted_score,
)


def _parse_robust(content: str):
    """Parse judge response into {label, point_hits, reasoning}. Copied from LLMJudge."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except Exception:
            pass

    m = re.search(r'"?label"?\s*:\s*"?(CORRECT|WRONG)"?', content, re.IGNORECASE)
    if not m:
        return None

    hits = [
        v.lower() == "true"
        for v in re.findall(r'"?hit"?\s*:\s*(true|false)', content, re.IGNORECASE)
    ]
    return {
        "label": m.group(1).upper(),
        "point_hits": [{"hit": h} for h in hits],
        "reasoning": "(regex-recovered: malformed JSON)",
    }

JUDGE_FAIL_REASONS = {"empty response", "bad label", "unparseable", "retries exhausted"}

MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")


async def _judge(
    session: aiohttp.ClientSession,
    question: str,
    reference_answer: str,
    generated_answer: str,
    question_types: list,
    ask_time: str,
    score_points: list,
    max_tokens: int,
    max_retries: int = 3,
) -> dict:
    """One judge call -> {is_correct, point_hits, reasoning}. Mirrors LLMJudge._judge
    but adds an explicit ``max_tokens`` to the payload."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        ask_time=ask_time,
        question_types=", ".join(question_types) if question_types else "（未标注）",
        reference_answer=reference_answer,
        score_points=_render_score_points(score_points),
        generated_answer=generated_answer,
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    url = f"{BASE_URL}/chat/completions"

    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status >= 500:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status
                    )
                resp.raise_for_status()
                data = await resp.json()

            if isinstance(data, dict) and "choices" in data:
                content = data["choices"][0]["message"]["content"]
            else:
                content = str(data)

            if not content:
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                return {"is_correct": False, "point_hits": [], "reasoning": "empty response"}

            obj = _parse_robust(content)
            if obj is None:
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                return {"is_correct": False, "point_hits": [], "reasoning": "unparseable"}

            label = str(obj.get("label", "")).strip().upper()
            if label not in ("CORRECT", "WRONG"):
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                return {"is_correct": False, "point_hits": [], "reasoning": "bad label"}

            point_hits = obj.get("point_hits", [])
            if not isinstance(point_hits, list):
                point_hits = []
            return {
                "is_correct": label == "CORRECT",
                "point_hits": point_hits,
                "reasoning": str(obj.get("reasoning", "")),
            }
        except Exception as exc:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            return {"is_correct": False, "point_hits": [], "reasoning": str(exc)}

    return {"is_correct": False, "point_hits": [], "reasoning": "retries exhausted"}


def _majority_vote(verdicts: list, score_points: list, num_runs: int) -> dict:
    """Aggregate num_runs verdicts -> {is_correct, point_hits, reasoning, judge_votes}."""
    correct_votes = sum(int(v["is_correct"]) for v in verdicts)
    is_correct = correct_votes * 2 >= num_runs

    point_hits = []
    for index, score_point in enumerate(score_points):
        hit_votes = 0
        for candidate in verdicts:
            candidate_hits = candidate.get("point_hits", [])
            if index < len(candidate_hits):
                hit = candidate_hits[index]
                hit_votes += int(
                    hit.get("hit", False) if isinstance(hit, dict) else bool(hit)
                )
        point_hits.append(
            {
                "description": (score_point or {}).get("description", ""),
                "hit": hit_votes * 2 >= num_runs,
            }
        )

    representative = next(
        (v for v in verdicts if v["is_correct"] == is_correct), verdicts[0]
    )
    return {
        "is_correct": is_correct,
        "point_hits": point_hits,
        "reasoning": representative.get("reasoning", ""),
        "judge_votes": {"correct": correct_votes, "wrong": num_runs - correct_votes},
    }


def _recompute(detailed_results: list) -> dict:
    """Recompute aggregate + per-type stats from detailed_results."""
    total = len(detailed_results)
    correct = sum(1 for d in detailed_results if d["is_correct"])
    accuracy = correct / total if total else 0.0

    scored = [d["weighted_score"] for d in detailed_results if d["weighted_score"] is not None]
    weighted_score = sum(scored) / len(scored) if scored else None

    type_stats = defaultdict(lambda: {"n": 0, "correct": 0, "score_sum": 0.0, "score_n": 0})
    for d in detailed_results:
        for t in d.get("question_types", ["_untyped"]):
            ts = type_stats[t]
            ts["n"] += 1
            ts["correct"] += int(d["is_correct"])
            if d["weighted_score"] is not None:
                ts["score_sum"] += d["weighted_score"]
                ts["score_n"] += 1

    qts = {}
    for t, ts in type_stats.items():
        qts[t] = {
            "count": ts["n"],
            "correct": ts["correct"],
            "accuracy": ts["correct"] / ts["n"] if ts["n"] else 0.0,
            "weighted_score": ts["score_sum"] / ts["score_n"] if ts["score_n"] else None,
        }
    return {
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "weighted_score": weighted_score,
        "question_type_stats": qts,
    }


async def run(results_dir: str, num_runs: int, concurrency: int, max_tokens: int,
              dry_run: int, max_retries: int, outer_retries: int) -> None:
    base = Path(results_dir)
    answer_path = base / "answer_results.json"
    eval_path = base / "eval_results.json"
    with open(answer_path, encoding="utf-8") as f:
        answers = json.load(f)
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    ans_by_qid = {a["question_id"]: a for a in answers}
    detailed = eval_data["detailed_results"]

    to_rejudge = [d for d in detailed if d["reasoning"] in JUDGE_FAIL_REASONS]
    if dry_run > 0:
        to_rejudge = to_rejudge[:dry_run]
    print(f"Judge-fail questions to re-judge: {len(to_rejudge)}")

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)

    recovered = {}

    async with aiohttp.ClientSession(connector=connector) as session:
        async def rejudge_one(d):
            a = ans_by_qid[d["question_id"]]
            meta = a.get("metadata", {}) or {}
            question_types = _normalize_types(meta.get("question_type"))
            ask_time = meta.get("ask_time", "") or "（未提供）"
            score_points = meta.get("score_points") or []
            async with semaphore:
                verdicts = []
                for _ in range(num_runs):
                    verdicts.append(
                        await _judge(
                            session,
                            question=a["question"],
                            reference_answer=a["golden_answer"],
                            generated_answer=a["answer"],
                            question_types=question_types,
                            ask_time=ask_time,
                            score_points=score_points,
                            max_tokens=max_tokens,
                            max_retries=max_retries,
                        )
                    )
            agg = _majority_vote(verdicts, score_points, num_runs)
            weighted = _weighted_score(score_points, agg["point_hits"])
            return {
                **d,
                "is_correct": agg["is_correct"],
                "weighted_score": weighted,
                "point_hits": agg["point_hits"],
                "reasoning": agg["reasoning"],
                "judge_votes": agg["judge_votes"],
            }

        # Multi-pass: questions that still return empty on one pass get re-tried
        # on the next pass, up to outer_retries times.
        remaining = list(to_rejudge)
        for outer in range(1, outer_retries + 1):
            if not remaining:
                break
            print(f"  pass {outer}/{outer_retries}: {len(remaining)} questions")
            batch_still = {}
            for coro in asyncio.as_completed([rejudge_one(d) for d in remaining]):
                new = await coro
                qid = new["question_id"]
                if new["reasoning"] in JUDGE_FAIL_REASONS:
                    batch_still[qid] = new
                else:
                    recovered[qid] = new
            remaining = list(batch_still.values())
            print(f"    recovered={len(recovered)} cumulative, still-failed={len(remaining)}")
        still_failed = {d["question_id"]: d["reasoning"] for d in remaining}

    # Merge recovered verdicts back into detailed_results
    merged = [recovered.get(d["question_id"], d) for d in detailed]

    result = {
        "total_questions": len(merged),
        "correct": sum(1 for d in merged if d["is_correct"]),
        "accuracy": sum(1 for d in merged if d["is_correct"]) / len(merged),
        "weighted_score": None,
        "detailed_results": merged,
        "question_type_stats": {},
        "metadata": {
            **eval_data.get("metadata", {}),
            "rejudged": {
                "num_attempted": len(to_rejudge),
                "num_recovered": len(recovered),
                "num_still_failed": len(still_failed),
            },
        },
    }
    stats = _recompute(merged)
    result["weighted_score"] = stats["weighted_score"]
    result["question_type_stats"] = stats["question_type_stats"]

    out_path = base / "eval_results_rejudged.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("Re-judged result (corrected for judge failures)")
    print(f"{'=' * 60}")
    print(f"Attempted: {len(to_rejudge)} | Recovered: {len(recovered)} | "
          f"Still failed: {len(still_failed)}")
    if still_failed:
        print(f"Still-failed reasons: {still_failed}")
    print(f"Correct: {result['correct']}/{result['total_questions']}")
    print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
    print(f"Weighted Score: {result['weighted_score'] * 100:.2f}%")
    print("\nPer-type accuracy (corrected):")
    for t, s in sorted(stats["question_type_stats"].items()):
        print(f"  {t}: {s['accuracy'] * 100:.2f}% ({s['correct']}/{s['count']})")
    print(f"\nSaved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-judge judge-failure questions")
    parser.add_argument("--results-dir", default="results/lifebench-direct_evidence")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=8,
                        help="Retries per judge LLM call before giving up on empty/unparseable")
    parser.add_argument("--outer-retries", type=int, default=4,
                        help="Full passes over still-failing questions")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Only re-judge first N failures (smoke test)")
    args = parser.parse_args()
    asyncio.run(run(args.results_dir, args.num_runs, args.concurrency,
                    args.max_tokens, args.dry_run, args.max_retries, args.outer_retries))


if __name__ == "__main__":
    main()