"""
Text-match based Recall@K evaluation for LifeBench search results.

Zero-LLM alternative to ``recall_evaluator.py``: judges whether an evidence
item is covered by the top-K retrieved chunks using pure text matching
(substring or token-overlap).  Suitable for memory systems that store raw
text (e.g. memu), unreliable for systems that return LLM-distilled memories
(e.g. memos).

Statistics follow the same question/evidence scope as ``recall_evaluator``
(only questions with evidence are counted; unanswerable questions skipped).
Results are additionally broken down by evidence ``source``, because
sources that were never ingested (calendar/note/push) can never match —
reporting them separately avoids systematically deflating the aggregate.

Usage:
    python -m src.utils.recall_text_match --results-dir results/lifebench-memu_cloud

    # Custom K values / matching method:
    python -m src.utils.recall_text_match --results-dir results/lifebench-memu_cloud \\
        --k-values 5,10,15,20 --method substring
    python -m src.utils.recall_text_match --results-dir results/lifebench-memu_cloud \\
        --method token --token-threshold 0.5
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.recall_evaluator import load_evidence_mapping  # noqa: E402


# ---------------------------------------------------------------------------
# Evidence → candidate texts
# ---------------------------------------------------------------------------


def extract_candidates(ev: dict) -> List[str]:
    """Return candidate texts for an evidence item (any hit counts).

    ``agent_chat`` evidence spans multiple turns; each user/agent utterance
    is a separate candidate because retrieved chunks are single messages.
    """
    raw = ev.get("raw_data", {}) or {}
    source = ev.get("source", "")
    if source == "calendar":
        return [((raw.get("title", "") or "") + " " + (raw.get("description", "") or "")).strip()]
    if source == "sms":
        return [raw.get("message_content", "") or ""]
    if source == "note":
        return [((raw.get("title", "") or "") + " " + (raw.get("content", "") or "")).strip()]
    if source == "push":
        return [((raw.get("title", "") or "") + " " + (raw.get("content", "") or "")).strip()]
    if source == "call":
        return [raw.get("contactName", "") or ""]
    if source == "photo":
        return [raw.get("description", "") or ""]
    if source == "email":
        return [raw.get("content", "") or ""]
    if source == "agent_chat":
        conv = raw.get("conversation", {}) or {}
        parts = []
        for turn in conv.values():
            if isinstance(turn, dict):
                u = turn.get("user", {}) or {}
                a = turn.get("assistant", {}) or {}
                if u.get("content"):
                    parts.append(u["content"])
                if a.get("content"):
                    parts.append(a["content"])
        return parts
    return [json.dumps(raw, ensure_ascii=False)]


def _tokenize(text: str) -> set:
    """Split into CJK chunks and alphanumeric words (case-insensitive)."""
    return set(re.findall(r"[一-鿿]+|[a-z0-9]+", text.lower()))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_evidence(
    candidates: List[str],
    result_texts: List[str],
    method: str = "substring",
    token_threshold: float = 0.5,
) -> tuple:
    """Return (hit: bool, first_rank: int) for an evidence item.

    ``first_rank`` is 1-indexed position of the first covering result,
    or 0 if no result covers it.
    """
    for idx, rt in enumerate(result_texts):
        for cand in candidates:
            if not cand:
                continue
            if method == "substring":
                if re.sub(r"\s+", "", cand) in re.sub(r"\s+", "", rt):
                    return True, idx + 1
            else:  # token overlap
                cand_tokens = _tokenize(cand)
                if not cand_tokens:
                    continue
                rt_tokens = _tokenize(rt)
                overlap = len(cand_tokens & rt_tokens) / len(cand_tokens)
                if overlap >= token_threshold:
                    return True, idx + 1
    return False, 0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_text_recall(
    results_dir: str,
    k_values: List[int],
    method: str = "substring",
    token_threshold: float = 0.5,
    output_path: Optional[str] = None,
) -> dict:
    base = Path(__file__).resolve().parent.parent.parent
    results_path = Path(results_dir) / "search_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"search_results.json not found at {results_path}")

    with open(results_path, encoding="utf-8") as f:
        search_data = json.load(f)

    raw_dir = base / "datasets" / "lifebench_raw"
    evidence_lookup = load_evidence_mapping(raw_dir)
    print(f"  Loaded evidence for {len(evidence_lookup)} question_ids")

    max_k = max(k_values)

    # Pre-build top-K result texts per question
    q_results: Dict[str, List[str]] = {}
    for sr in search_data:
        q_results[sr["question_id"]] = [
            r.get("content", "") for r in sr.get("results", [])[:max_k]
        ]

    # (question_id, source, first_rank) per evidence item
    hits: List[tuple] = []
    n_questions = 0
    # per-question: which evidence ranks each result covers (for precision)
    result_hits: List[Dict[int, int]] = []  # {result_idx: evidence_count}
    for sr in search_data:
        qid = sr["question_id"]
        evidence = evidence_lookup.get(qid, [])
        if not evidence:
            continue
        n_questions += 1
        result_texts = q_results.get(qid, [])
        counts: Dict[int, int] = defaultdict(int)
        for ev in evidence:
            candidates = [c for c in extract_candidates(ev) if c.strip()]
            if not candidates:
                continue
            hit, rank = match_evidence(
                candidates, result_texts, method, token_threshold
            )
            hits.append((qid, ev.get("source", "?"), rank))
            if rank > 0:
                counts[rank - 1] += 1
        result_hits.append(counts)

    # ── aggregates ─────────────────────────────────────────────────────
    total_ev = len(hits)
    by_source: Dict[str, List[int]] = defaultdict(list)
    for _, src, rank in hits:
        by_source[src].append(rank)

    def _f1(recall: float, precision: float) -> float:
        return 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0

    summary: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        recall = sum(1 for _, _, r in hits if 1 <= r <= k) / total_ev
        # Strict precision: share of result slots within the first K that
        # cover at least one evidence.  Denominator uses the actual number
        # of results min(K, len(results)) so systems returning fewer than K
        # results (e.g. memu ~8-9) are not diluted by empty slots.
        hit_slots = sum(
            sum(1 for i in range(min(k, len(rh))) if rh.get(i, 0) > 0)
            for rh in result_hits
        )
        total_slots = sum(min(k, len(rh)) for rh in result_hits)
        prec = hit_slots / total_slots if total_slots else 0.0
        summary[f"recall@{k}"] = {
            "overall": recall,
            "by_source": {
                src: sum(1 for r in ranks if 1 <= r <= k) / len(ranks)
                for src, ranks in by_source.items()
            },
        }
        summary[f"precision@{k}"] = {
            "overall": prec,
            "by_source": {},
        }
        summary[f"f1@{k}"] = {
            "overall": _f1(recall, prec),
            "by_source": {},
        }

    per_question: List[Dict] = []
    by_qid: Dict[str, List[tuple]] = defaultdict(list)
    for qid, src, rank in hits:
        by_qid[qid].append((src, rank))
    for qid, ev_list in by_qid.items():
        pq = {"question_id": qid, "evidence": []}
        for src, rank in ev_list:
            pq["evidence"].append({"source": src, "first_rank": rank})
        for k in k_values:
            found = sum(1 for _, r in ev_list if 1 <= r <= k)
            pq[f"recall@{k}"] = found / len(ev_list) if ev_list else 0.0
        per_question.append(pq)

    result = {
        "method": method,
        "token_threshold": token_threshold if method == "token" else None,
        "k_values": k_values,
        "questions": n_questions,
        "evidence_total": total_ev,
        "evidence_by_source": {src: len(r) for src, r in by_source.items()},
        "summary": summary,
        "per_question": per_question,
    }

    # ── print ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Recall@K (text-match, method={method})")
    print(f"{'=' * 60}")
    print(f"Questions: {n_questions} | Evidence: {total_ev} | "
          f"By source: {result['evidence_by_source']}")
    print()
    print("  Recall@K (by source):")
    print(f"  {'K':>3s} {'Overall':>10s}  " + " ".join(
        f"{src:>12s}" for src in by_source
    ))
    for k in k_values:
        s = summary[f"recall@{k}"]
        row = f"  {k:>3d} {s['overall']*100:>9.2f}% "
        row += " ".join(f"{s['by_source'].get(src, 0)*100:>11.2f}%" for src in by_source)
        print(row)
    print()
    print("  Recall / Precision / F1 (overall):")
    print(f"  {'K':>3s} {'Recall':>10s} {'Precision':>12s} {'F1':>10s}")
    for k in k_values:
        r = summary[f"recall@{k}"]["overall"]
        p = summary[f"precision@{k}"]["overall"]
        f1 = summary[f"f1@{k}"]["overall"]
        print(f"  {k:>3d} {r*100:>9.2f}% {p*100:>11.2f}% {f1*100:>9.2f}%")

    if output_path is None:
        output_path = str(Path(results_dir) / "recall_results_text_match.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Text-match Recall@K for LifeBench")
    parser.add_argument("--results-dir", required=True,
                        help="Path to results directory containing search_results.json")
    parser.add_argument("--k-values", default="5,10,15,20",
                        help="Comma-separated K values (default: 5,10,15,20)")
    parser.add_argument("--method", default="substring", choices=["substring", "token"],
                        help="Matching method (default: substring)")
    parser.add_argument("--token-threshold", type=float, default=0.5,
                        help="Token-overlap ratio threshold for --method token (default: 0.5)")
    parser.add_argument("--output", default=None,
                        help="Path to save detailed results JSON")
    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    evaluate_text_recall(
        results_dir=args.results_dir,
        k_values=k_values,
        method=args.method,
        token_threshold=args.token_threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
