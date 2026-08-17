"""
Fix recall judge: re-run the recall (coverage) judge for QA that ended up with
zero covered evidence, to separate genuine "no coverage" from judge failures.

The recall_checkpoint.json does not record a failure flag — when the coverage
judge returns None (unparseable / HTTP error after retries) it writes the same
all-empty ``evidence_ranks`` as a legitimate "nothing covered" result. Re-running
the judge on those zero-coverage QA lets us split them:

    * judge now returns non-empty coverage  ->  the previous run was a failure
    * judge now returns empty coverage      ->  genuine "no coverage"
    * judge still returns None              ->  still failing

Usage:
    python src/utils/fix_recall.py --results-dir results/lifebench-mem0
"""
import argparse
import asyncio
import copy
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime

# Remove script directory from sys.path to avoid shadowing stdlib modules
# (this directory contains logging.py which breaks aiohttp's `import logging`).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_abs_paths = [os.path.abspath(p) for p in sys.path]
for i in reversed([i for i, p in enumerate(_abs_paths) if p == os.path.abspath(_SCRIPT_DIR)]):
    sys.path.pop(i)

# Load recall_evaluator by explicit file path (its RecallJudge + helpers).
# Loading it this way (instead of `import recall_evaluator`) keeps the script-dir
# shadowing fix above intact while still reusing the judge logic verbatim.
_recall_path = os.path.join(_SCRIPT_DIR, "recall_evaluator.py")
_spec = importlib.util.spec_from_file_location("_recall_evaluator", _recall_path)
_recall = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recall)

RecallJudge = _recall.RecallJudge
load_evidence_mapping = _recall.load_evidence_mapping
_build_recall_results = _recall._build_recall_results
_save_checkpoint = _recall._save_checkpoint

# Load .env manually (robust against dotenv / path edge cases).
_ENV_PATH = os.path.join(_SCRIPT_DIR, "..", "..", ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val

from tqdm import tqdm


def _is_zero_coverage(entry: dict) -> bool:
    """True when no evidence item has any covering search result."""
    er = entry.get("evidence_ranks") or []
    return all(not r for r in er)


def _compute_metrics(items, completed, k_values):
    """Aggregate (recall, precision) per k, mirroring recall_evaluator's math."""
    all_results = []
    for sr, ev in items:
        qid = sr["question_id"]
        entry = completed.get(qid)
        if entry is None:
            continue
        all_results.append((
            qid,
            entry["evidence_ranks"],
            entry["result_counts"],
            entry.get("min_sufficient_rank", -1),
            entry.get("supporting_ranks", []),
        ))
    n_q = len(all_results)
    metrics = {}
    for k in k_values:
        total_ev = sum(len(r[1]) for r in all_results)
        first_ranks = []
        for _, ev_ranks, _, _, _ in all_results:
            first_ranks.extend([r[0] if r else 0 for r in ev_ranks])
        recall = sum(1 for r in first_ranks if 1 <= r <= k) / total_ev if total_ev else 0.0
        prec_sum = sum(
            1 for _, _, rc, _, _ in all_results for c in rc[:k] if c > 0
        )
        precision = prec_sum / (k * n_q) if n_q else 0.0
        metrics[k] = (recall, precision)
    return metrics


async def run(args):
    results_dir = os.path.abspath(args.results_dir)
    base = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

    results_path = os.path.join(results_dir, "search_results.json")
    checkpoint_path = os.path.join(results_dir, "recall_checkpoint.json")
    recall_results_path = os.path.join(results_dir, "recall_results.json")

    if not os.path.exists(results_path):
        print(f"ERROR: search_results.json not found at {results_path}")
        sys.exit(1)

    # ---- Load data ----------------------------------------------------------
    with open(results_path, "r", encoding="utf-8") as f:
        search_data = json.load(f)

    raw_dir = os.path.join(base, "datasets", "lifebench_raw")
    evidence_lookup = load_evidence_mapping(raw_dir)

    conv_path = os.path.join(
        base, "datasets", "lifebench_locomo_format",
        "lifebench_locomo_conversation_format_v2.0_3380QA.json",
    )
    with open(conv_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    answer_map = {}
    for person in conv_data:
        for qa in person["qa"]:
            answer_map[qa["question_id"]] = qa.get("answer", "")

    sr_map = {}
    items = []
    evidence_by_qid = {}
    for sr in search_data:
        qid = sr["question_id"]
        sr_map[qid] = sr
        evidence = evidence_lookup.get(qid, [])
        if evidence:
            items.append((sr, evidence))
            evidence_by_qid[qid] = evidence

    # ---- Load checkpoint ----------------------------------------------------
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
            if isinstance(ckpt, dict) and "completed" in ckpt:
                for entry in ckpt["completed"]:
                    completed[entry["question_id"]] = {
                        "evidence_ranks": entry.get("evidence_ranks", entry.get("ranks", [])),
                        "result_counts": entry.get("result_counts", []),
                        "min_sufficient_rank": entry.get("min_sufficient_rank", -1),
                        "supporting_ranks": entry.get("supporting_ranks", []),
                    }

    # ---- k values -----------------------------------------------------------
    k_values = [5, 10, 15, 20]
    if os.path.exists(recall_results_path):
        with open(recall_results_path, "r", encoding="utf-8") as f:
            rr = json.load(f)
            if isinstance(rr.get("k_values"), list) and rr["k_values"]:
                k_values = [int(k) for k in rr["k_values"]]
    max_k = max(k_values)

    # ---- Identify zero-coverage QA -----------------------------------------
    zero_qids = [
        qid for qid, entry in completed.items()
        if _is_zero_coverage(entry) and qid in evidence_by_qid
    ]
    orig_zero = len(zero_qids)
    total_q = len(completed)
    print(f"Total completed QA: {total_q}")
    print(f"Zero-coverage QA to re-judge: {orig_zero}")

    if not zero_qids:
        print("No zero-coverage QA — nothing to do.")
        return

    before_metrics = _compute_metrics(items, completed, k_values)

    # ---- Backup -------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in (checkpoint_path, recall_results_path):
        if os.path.exists(p):
            shutil.copy2(p, p + f".bak_{ts}")
    print(f"Backup created: *.bak_{ts}\n")

    # ---- Judge setup --------------------------------------------------------
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        print("ERROR: LLM_API_KEY not set in environment")
        sys.exit(1)

    judge = RecallJudge({"api_key": api_key, "base_url": base_url})
    print(f"Judge model: {judge.model}  |  Base URL: {base_url}")
    print(f"Concurrency: {args.concurrency}  |  Max passes: {args.max_passes}\n")
    sem = asyncio.Semaphore(args.concurrency)

    completed_before = copy.deepcopy(completed)

    newly_covered = set()
    remaining = list(zero_qids)

    # ---- Multi-pass re-judge ------------------------------------------------
    for pass_num in range(1, args.max_passes + 1):
        if not remaining:
            break
        print(f"Pass {pass_num}/{args.max_passes}: re-judging {len(remaining)} QA")

        pbar = tqdm(total=len(remaining), desc=f"Pass {pass_num}", unit="qa")

        async def rejudge_one(qid):
            async with sem:
                question = sr_map[qid].get("query", "")
                answer = answer_map.get(qid, "")
                evidence = evidence_by_qid[qid]
                search_results = sr_map[qid].get("results", [])
                result = await judge.judge(
                    question=question,
                    reference_answer=answer,
                    evidence_items=evidence,
                    search_results=search_results,
                    max_rank=max_k,
                )
                pbar.update(1)
                return qid, result

        results_list = await asyncio.gather(*[rejudge_one(qid) for qid in remaining])
        pbar.close()

        next_remaining = []
        for qid, result in results_list:
            if result is None:
                next_remaining.append(qid)
                continue
            ev_ranks, res_counts = result
            completed[qid]["evidence_ranks"] = ev_ranks
            completed[qid]["result_counts"] = res_counts
            if not _is_zero_coverage(completed[qid]):
                newly_covered.add(qid)

        _save_checkpoint(checkpoint_path, completed)
        print(f"  newly covered so far: {len(newly_covered)}  |  still failing: {len(next_remaining)}")
        remaining = next_remaining

    await judge.close()

    still_failed = set(remaining)
    genuine_zero = set(zero_qids) - newly_covered - still_failed

    # ---- Write final outputs ------------------------------------------------
    _save_checkpoint(checkpoint_path, completed)
    summary = _build_recall_results(items, completed, k_values, max_k)
    with open(recall_results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    after_metrics = _compute_metrics(items, completed, k_values)

    # ---- Report -------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Recall judge re-run summary")
    print(f"{'=' * 60}")
    print(f"  Zero-coverage QA re-judged:       {orig_zero}")
    print(f"    -> newly covered (were failures): {len(newly_covered)}")
    print(f"    -> genuine zero coverage:        {len(genuine_zero)}")
    print(f"    -> still failing (None):         {len(still_failed)}")
    print(f"  Successfully evaluated recall:     {total_q - len(still_failed)}/{total_q}")
    if still_failed:
        sample = sorted(still_failed)[:5]
        print(f"  Still-failing samples: {sample}")

    print(f"\n  {'k':>3}  {'Recall@k (before -> after)':>30}  {'Precision@k (before -> after)':>34}")
    for k in k_values:
        rb, pb = before_metrics[k]
        ra, pa = after_metrics[k]
        print(f"  {k:>3}  {rb:>12.4f} -> {ra:<12.4f}  {pb:>14.4f} -> {pa:<12.4f}")

    print(f"\nSaved: {checkpoint_path}")
    print(f"Saved: {recall_results_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-run recall judge for zero-coverage QA")
    parser.add_argument("--results-dir", required=True,
                        help="Results directory containing search_results.json / recall_checkpoint.json")
    parser.add_argument("--concurrency", type=int, default=15,
                        help="Max concurrent LLM calls (default: 15)")
    parser.add_argument("--max-passes", type=int, default=3,
                        help="Max re-judge passes (default: 3)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
