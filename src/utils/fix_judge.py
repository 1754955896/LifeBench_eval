"""
Fix failed LLM Judge evaluations by re-running judge calls with retries.
Runs multiple passes until all failures are resolved or max passes reached.

Usage: python fix_judge.py [--max-passes N] [--concurrency N]
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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

import aiohttp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是记忆系统 benchmark 的评分员，务实而不吹毛求疵。判答案对错的唯一标准是
「问题问的核心，答对了没有」——只要核心答对就算对，不要求答案面面俱到、不要求
它解释依据或覆盖所有相关要点。但答错、编造、答非所问绝不给分。被评内容为中文。"""

USER_PROMPT_TEMPLATE = """给一条"系统生成的答案"打分。你要产出两样东西：(A) 整体对错 label，(B) 逐评分点命中。

## 输入
- 问题：{question}
- 提问时间（问题被提出的时刻）：{ask_time}
- 题型：{question_types}
- 参考答案：{reference_answer}
- 评分点（评分清单；分值已隐藏，不要臆测权重）：
{score_points}
- 待评的生成答案：{generated_answer}

## 判分总原则（务实、偏宽，核心对就算对）
- **答对核心就算对**：只要生成答案把问题问的**核心内容/答案**答对了，就判 CORRECT，
  **哪怕**它漏了次要要点、没说全所有相关原因、没引用证据出处、没解释推理过程、
  没复述问题里已给的前提、也没逐字给出参考答案里的某个日期标签或事件名。这些都**不影响对错**。
- 换种说法、改写、日期/数字格式或合理的四舍五入差异、补充的正确额外信息，都**不扣分**。
- **但绝不放水**（下列判 WRONG）：核心事实答错、张冠李戴、凭空编造、答非所问、
  该答却拒答（答"无法确定"但记录里其实有答案）。只是"话题沾边"而没答到正确核心，也是错。
  "漏次要点"宽容、但"用笼统泛化概括顶替问题问的具体内容"（如问做了什么具体事却只答'学习了一下/处理了点事'）＝没答到核心＝WRONG。
- **关键区分：漏 ≠ 答错**。漏掉次要内容/没说全＝宽容不扣分；但答案里**明确给错**的关键事实
  （把日期/数字/人物/地点**答成了错的值**，如日期差了两天、对象张冠李戴）＝判 WRONG，即便核心沾边。
  合理的四舍五入或近似（如 44 与 44.1、半小时与约30分钟）不算答错。

## 时间语义
{ask_time} 是提问时刻；相关事件都在它之前。答案应反映**截至该时点**的状态；
相对时间（"上个月"/"目前"等）按提问时间解析。

## A. 整体 label（看核心，宽松）—— 这是主判定
判生成答案是否**正确回答了问题的核心**。核心对 → CORRECT；核心错/缺/编造/错误拒答 → WRONG。
按题型确定什么是"核心"：
- Unanswerable：核心＝该不该拒答。恰当拒答/「记录中没有」/「无法确定」＝CORRECT；编造具体答案＝WRONG。
- Knowledge_update：核心＝**提问时点**的状态值是否答对（不是全局最新）。
- Conflict：核心＝**最终裁决值**是否答对。给对值即 CORRECT，**不要求**答案展示它如何识别/排除了矛盾信息；
  但若答了矛盾里那个错误值（如盲目取最新而最新非正确值）＝WRONG。
- Temporal：核心＝问的日期/间隔/时长/排序是否对（格式宽松）。
- Multi_hop / Causal：核心＝问题主线的答案/因果是否对。主要事实或因果对即 CORRECT；
  漏掉次要环节不致错；只有主线答错或基本没答到才 WRONG。
- Pattern_recognition / Hidden_info：核心＝归纳的方向/结论是否对。结论对即 CORRECT，
  **不必穷举**所有支撑细节；方向反了或结论错才 WRONG。
- Evolution_tracking：核心＝某主线**随时间的演化轨迹/阶段/关键里程碑**是否抓对。覆盖主要阶段且顺序/方向对
  即 CORRECT，**不必穷举**每个节点；只给单点静态事实、漏掉主干演化、或把阶段顺序/方向答反才 WRONG。
- Single_hop：核心＝问的那个事实是否对。

## B. 逐点 score_points（算 weighted，也放松）—— 只衡量"答得多全"，不决定 A
对每个评分点判 hit/miss（与 A 解耦：B 漏几个点不影响 A 是否 CORRECT）：
- hit ＝ 答案传达了该点的**事实内容**（改写、等价表述、合理近似都算）。
- **不要因为**该点附带的「引用证据出处 / 说明推理或排除过程 / 复述问题前提 / 逐字给出某日期或事件名标签」
  这类**与答案对错无关的元要求**没满足，就判 miss——只看该点的**事实内容**答没答到。
- 该点的事实确实没答到、或答错 → miss。话题沾边但没给该点事实 → miss。
- 若上面"评分点"为空（无清单），point_hits 输出空数组 []。

## 多题型优先级
1. Unanswerable 最高——若本题期望恰当拒答，按 UA 判。
2. Knowledge_update 压过"最新＝真"的直觉。
3. 其余按各适用题型综合判核心。

## 输出（只输出 JSON，不要别的）
{{"reasoning": "<一两句中文说明，先说核心对错的理由>",
  "point_hits": [{{"description": "<照抄该评分点>", "hit": true}}],
  "label": "CORRECT"}}

point_hits 的条目数量和顺序必须与输入的评分点完全一致。label 必须正好是 "CORRECT" 或 "WRONG"。
reasoning 和 description 字段内**不要使用英文双引号**（会破坏 JSON）；如需引用原文请用中文引号「」。」"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_types(value) -> List[str]:
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _render_score_points(score_points: List[dict]) -> str:
    if not score_points:
        return "（本题无评分点清单，只判整体 label）"
    lines = []
    for i, sp in enumerate(score_points, 1):
        desc = (sp or {}).get("description", "")
        lines.append(f"    {i}. {desc}")
    return "\n".join(lines)


def _weighted_score(score_points: List[dict], point_hits: List) -> Optional[float]:
    if not score_points:
        return None
    total = sum(float((sp or {}).get("score", 0) or 0) for sp in score_points)
    if total <= 0:
        return None
    if len(point_hits) != len(score_points):
        return None
    got = 0.0
    for sp, ph in zip(score_points, point_hits):
        hit = ph.get("hit") if isinstance(ph, dict) else bool(ph)
        if hit:
            got += float((sp or {}).get("score", 0) or 0)
    return got / total


def _parse_robust(content: str):
    """Parse judge response into {label, point_hits, reasoning}."""
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


def is_failing(r: dict) -> bool:
    """Check if a result entry still has a failed judge."""
    reason = r.get("reasoning", "")
    return any(kw in reason for kw in ("empty response", "bad label", "unparseable"))


# ---------------------------------------------------------------------------
# Judge API call
# ---------------------------------------------------------------------------

async def judge_single(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    model: str,
    api_key: str,
    base_url: str,
    entry: dict,
    max_retries: int = 3,
) -> dict:
    """Call judge API with retries. Returns {is_correct, point_hits, reasoning, weighted_score}."""
    question = entry["question"]
    reference_answer = entry["golden_answer"]
    generated_answer = entry["generated_answer"]
    meta = entry.get("_meta", {})
    question_types = _normalize_types(meta.get("question_type"))
    ask_time = meta.get("ask_time", "") or "（未提供）"
    score_points = meta.get("score_points") or []

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        ask_time=ask_time,
        question_types=", ".join(question_types) if question_types else "（未标注）",
        reference_answer=reference_answer,
        score_points=_render_score_points(score_points),
        generated_answer=generated_answer,
    )

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    for attempt in range(1, max_retries + 1):
        try:
            async with sem:
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
                    await asyncio.sleep(2 * attempt)
                    continue
                return {
                    "is_correct": False, "point_hits": [],
                    "reasoning": "empty response (retried)", "weighted_score": None,
                }

            obj = _parse_robust(content)
            if obj is None:
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
                    continue
                return {
                    "is_correct": False, "point_hits": [],
                    "reasoning": "unparseable (retried)", "weighted_score": None,
                }

            label = str(obj.get("label", "")).strip().upper()
            if label not in ("CORRECT", "WRONG"):
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
                    continue
                reasoning_text = str(obj.get("reasoning", ""))
                if "正确" in reasoning_text or "答对" in reasoning_text:
                    label = "CORRECT"
                elif "错误" in reasoning_text or "答错" in reasoning_text:
                    label = "WRONG"
                else:
                    return {
                        "is_correct": False, "point_hits": [],
                        "reasoning": "bad label (retried)", "weighted_score": None,
                    }

            point_hits = obj.get("point_hits", [])
            if not isinstance(point_hits, list):
                point_hits = []
            weighted = _weighted_score(score_points, point_hits)

            return {
                "is_correct": label == "CORRECT",
                "point_hits": point_hits,
                "reasoning": str(obj.get("reasoning", "")),
                "weighted_score": weighted,
            }

        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 * attempt)
                continue
            return {
                "is_correct": False, "point_hits": [],
                "reasoning": str(e), "weighted_score": None,
            }

    return {
        "is_correct": False, "point_hits": [],
        "reasoning": "exhausted retries", "weighted_score": None,
    }


# ---------------------------------------------------------------------------
# Stats recompute
# ---------------------------------------------------------------------------

def _recompute_stats(detailed: List[dict]) -> dict:
    total = len(detailed)
    correct = sum(1 for r in detailed if r["is_correct"])
    accuracy = correct / total if total else 0.0
    scored = [r["weighted_score"] for r in detailed if r["weighted_score"] is not None]
    weighted_score = sum(scored) / len(scored) if scored else None

    type_stats: Dict[str, dict] = defaultdict(
        lambda: {"n": 0, "correct": 0, "score_sum": 0.0, "score_n": 0}
    )
    for r in detailed:
        for t in r.get("question_types", ["_untyped"]):
            ts = type_stats[t]
            ts["n"] += 1
            ts["correct"] += int(r["is_correct"])
            if r["weighted_score"] is not None:
                ts["score_sum"] += r["weighted_score"]
                ts["score_n"] += 1

    question_type_stats = {}
    for t, ts in type_stats.items():
        question_type_stats[t] = {
            "name": t,
            "count": ts["n"],
            "correct": ts["correct"],
            "accuracy": ts["correct"] / ts["n"] if ts["n"] else 0.0,
        }

    remaining = sum(1 for r in detailed if is_failing(r))
    return {
        "total": total, "correct": correct, "accuracy": accuracy,
        "weighted_score": weighted_score, "question_type_stats": question_type_stats,
        "remaining": remaining,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Fix failed LLM Judge evaluations")
    parser.add_argument("--max-passes", type=int, default=5,
                        help="Maximum re-judge passes (default: 5)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Concurrent API calls (default: 5)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results", "lifebench-cognee")
    eval_path = os.path.join(results_dir, "eval_results.json")
    answer_path = os.path.join(results_dir, "answer_results.json")

    # ---- Step 1: Backup original eval_results.json ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = eval_path + f".bak_{ts}"
    shutil.copy2(eval_path, backup_path)
    print(f"Backup created: {backup_path}")

    # ---- Step 2: Load data ----
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    with open(answer_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    meta_by_qid = {a["question_id"]: a.get("metadata", {}) for a in answer_data}

    detailed = eval_data["detailed_results"]
    model = eval_data.get("metadata", {}).get("model", "deepseek-v4-flash")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        print("ERROR: LLM_API_KEY not set in environment")
        sys.exit(1)

    # Snapshot original stats for final comparison
    orig_total = eval_data["total_questions"]
    orig_correct = eval_data["correct"]
    orig_accuracy = eval_data["accuracy"]
    orig_ws = eval_data.get("weighted_score")

    print(f"Model: {model}  |  Base URL: {base_url}")
    print(f"Total questions: {orig_total}  |  Accuracy: {orig_accuracy:.4f}")
    print(f"Concurrency: {args.concurrency}  |  Max passes: {args.max_passes}")

    # ---- Step 3: Multi-pass re-judging ----
    total_fixed = 0

    for pass_num in range(1, args.max_passes + 1):
        failed = [(i, r) for i, r in enumerate(detailed) if is_failing(r)]
        if not failed:
            print(f"\nPass {pass_num}: No failures remaining — done.")
            break

        print(f"\n{'─' * 50}")
        print(f"Pass {pass_num}/{args.max_passes}: {len(failed)} entries to re-judge")

        # Attach metadata
        for _, r in failed:
            r["_meta"] = meta_by_qid.get(r["question_id"], {})

        # Run re-judge
        sem = asyncio.Semaphore(args.concurrency)
        connector = aiohttp.TCPConnector(limit=args.concurrency + 5)
        async with aiohttp.ClientSession(connector=connector) as session:
            pbar = tqdm(total=len(failed), desc=f"Pass {pass_num}", unit="qa")

            async def rejudge_one(pack):
                orig_idx, entry = pack
                result = await judge_single(
                    session, sem, model, api_key, base_url, entry, max_retries=3,
                )
                pbar.update(1)
                if is_failing(result):
                    pbar.write(f"  Still failed [{entry['question_id'][:35]}]: {result['reasoning'][:60]}")
                return orig_idx, result

            tasks = [rejudge_one(p) for p in failed]
            results_list = await asyncio.gather(*tasks)
            pbar.close()

        # Patch results in-place
        fix_map = dict(results_list)
        pass_fixed = 0
        for orig_idx, new in fix_map.items():
            r = detailed[orig_idx]
            was_failing = is_failing(r)
            detailed[orig_idx] = {
                "question_id": r["question_id"],
                "question": r["question"],
                "golden_answer": r["golden_answer"],
                "generated_answer": r["generated_answer"],
                "question_types": r.get("question_types", []),
                "is_correct": new["is_correct"],
                "weighted_score": new["weighted_score"],
                "point_hits": new.get("point_hits", []),
                "reasoning": new["reasoning"],
            }
            if was_failing and not is_failing(detailed[orig_idx]):
                pass_fixed += 1

        total_fixed += pass_fixed
        stats = _recompute_stats(detailed)
        print(f"  Fixed this pass: {pass_fixed}  |  Still failing: {stats['remaining']}  |  Accuracy: {stats['accuracy']:.4f}")

    # ---- Step 4: Write back to eval_results.json ----
    stats = _recompute_stats(detailed)
    fixed_data = {
        "total_questions": stats["total"],
        "correct": stats["correct"],
        "accuracy": stats["accuracy"],
        "weighted_score": stats["weighted_score"],
        "detailed_results": detailed,
        "question_type_stats": stats["question_type_stats"],
        "metadata": {
            **eval_data.get("metadata", {}),
            "fix_info": {
                "total_fixed": total_fixed,
                "remaining_failures": stats["remaining"],
                "passes": pass_num,
                "backup": os.path.basename(backup_path),
            },
        },
    }

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, indent=2, ensure_ascii=False, default=str)

    # ---- Step 5: Print comparison ----
    print(f"\n{'=' * 60}")
    print(f"Before / After")
    print(f"{'=' * 60}")
    print(f"  Total questions:         {orig_total:>6}  →  {stats['total']:>6}")
    print(f"  Correct:                 {orig_correct:>6}  →  {stats['correct']:>6}  (+{stats['correct'] - orig_correct})")
    print(f"  Accuracy:                {orig_accuracy:.4f}  →  {stats['accuracy']:.4f}  ({stats['accuracy']*100:.2f}%)")
    print(f"  Weighted score:          {orig_ws or 0:.4f}  →  {stats['weighted_score'] or 0:.4f}")
    print(f"  Total fixed:             {total_fixed}")
    print(f"  Remaining failures:      {stats['remaining']}")
    print(f"  Backup:                  {backup_path}")
    print(f"\nSaved to: {eval_path}")


if __name__ == "__main__":
    asyncio.run(main())
