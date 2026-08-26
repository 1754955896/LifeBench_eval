"""
LLM Judge evaluator - type-aware evaluation for LifeBench.

Based on LifeMem's evaluator with:
- ONE unified type-aware prompt covers all question types
- Configurable odd-numbered judge runs with majority voting (temperature=0)
- Binary is_correct + per-point weighted score
"""

import asyncio
import json
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional

import aiohttp
from tqdm import tqdm

from src.evaluators.base import BaseEvaluator
from src.evaluators.registry import register_evaluator
from src.models.evaluation import EvaluationResult, QuestionTypeStats


def _normalize_types(value) -> List[str]:
    """Normalize question_type to list of strings."""
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _render_score_points(score_points: List[dict]) -> str:
    """Render score points as a numbered, weight-free list for the judge prompt."""
    if not score_points:
        return "（本题无评分点清单，只判整体 label）"
    lines = []
    for i, sp in enumerate(score_points, 1):
        desc = (sp or {}).get("description", "")
        lines.append(f"    {i}. {desc}")
    return "\n".join(lines)


def _weighted_score(
    score_points: List[dict], point_hits: List
) -> Optional[float]:
    """Weighted hit ratio in [0,1]; None if no usable rubric or count mismatch."""
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
reasoning 和 description 字段内**不要使用英文双引号**（会破坏 JSON）；如需引用原文请用中文引号「」。"""


@register_evaluator("llm_judge")
class LLMJudge(BaseEvaluator):
    """Type-aware, score_points-based LLM judge."""

    def __init__(self, config: dict):
        super().__init__(config)
        llm_config = config.get("llm", {})
        self.model = llm_config.get("model", "deepseek-chat")
        self.api_key = llm_config.get("api_key", "")
        self.base_url = llm_config.get(
            "base_url", "https://api.deepseek.com"
        )
        self.num_runs = int(config.get("num_runs", 1))
        if self.num_runs < 1:
            raise ValueError("evaluation.num_runs must be at least 1")
        if self.num_runs % 2 == 0:
            raise ValueError("evaluation.num_runs must be odd for majority vote")
        self.max_retries = int(config.get("max_retries", 3))
        if self.max_retries < 1:
            raise ValueError("evaluation.max_retries must be at least 1")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=100)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def evaluate(self, answer_results: List) -> EvaluationResult:
        print(f"\n{'=' * 60}")
        print(
            "Stage 4/4: Evaluate  "
            f"[LLM Judge, model={self.model}, runs={self.num_runs}]"
        )
        print(f"{'=' * 60}")

        semaphore = asyncio.Semaphore(10)
        pbar = tqdm(total=len(answer_results), desc="Evaluate")

        async def eval_one(ar):
            async with semaphore:
                result = await self._evaluate_single(ar)
                pbar.update(1)
                return result

        results = await asyncio.gather(*[eval_one(ar) for ar in answer_results])
        pbar.close()

        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total else 0.0

        scored = [r["weighted_score"] for r in results if r["weighted_score"] is not None]
        weighted_score = sum(scored) / len(scored) if scored else None

        # Per-type stats
        type_stats: Dict[str, dict] = defaultdict(
            lambda: {"n": 0, "correct": 0, "score_sum": 0.0, "score_n": 0}
        )
        for r in results:
            types = r.get("question_types", ["_untyped"])
            for t in types:
                ts = type_stats[t]
                ts["n"] += 1
                ts["correct"] += int(r["is_correct"])
                if r["weighted_score"] is not None:
                    ts["score_sum"] += r["weighted_score"]
                    ts["score_n"] += 1

        question_type_stats = {}
        for t, ts in type_stats.items():
            question_type_stats[t] = QuestionTypeStats(
                name=t,
                count=ts["n"],
                correct=ts["correct"],
                accuracy=ts["correct"] / ts["n"] if ts["n"] else 0.0,
                weighted_score=(
                    ts["score_sum"] / ts["score_n"] if ts["score_n"] else None
                ),
            )

        print(f"\n✅ Evaluation complete:")
        print(f"   - Total questions: {total}")
        print(f"   - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        if weighted_score is not None:
            print(f"   - Weighted score: {weighted_score:.4f} ({weighted_score * 100:.2f}%)")

        return EvaluationResult(
            total_questions=total,
            correct=correct,
            accuracy=accuracy,
            weighted_score=weighted_score,
            detailed_results=results,
            question_type_stats=question_type_stats,
            metadata={
                "model": self.model,
                "evaluator": "llm_judge",
                "num_runs": self.num_runs,
                "aggregation": "majority_vote",
            },
        )

    async def _evaluate_single(self, ar) -> dict:
        """Evaluate a single answer result using LLM judge."""
        meta = ar.metadata or {}
        question_types = _normalize_types(meta.get("question_type"))
        ask_time = meta.get("ask_time", "") or "（未提供）"
        score_points = meta.get("score_points") or []

        verdicts = []
        for _ in range(self.num_runs):
            verdicts.append(
                await self._judge(
                    question=ar.question,
                    reference_answer=ar.golden_answer,
                    generated_answer=ar.answer,
                    question_types=question_types,
                    ask_time=ask_time,
                    score_points=score_points,
                )
            )

        correct_votes = sum(int(v["is_correct"]) for v in verdicts)
        is_correct = correct_votes * 2 >= self.num_runs
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
                    "hit": hit_votes * 2 >= self.num_runs,
                }
            )

        representative = next(
            (v for v in verdicts if v["is_correct"] == is_correct), verdicts[0]
        )
        verdict = {
            "is_correct": is_correct,
            "point_hits": point_hits,
            "reasoning": representative.get("reasoning", ""),
        }

        weighted = _weighted_score(score_points, verdict.get("point_hits", []))

        return {
            "question_id": ar.question_id,
            "question": ar.question,
            "golden_answer": ar.golden_answer,
            "generated_answer": ar.answer,
            "question_types": question_types,
            "is_correct": verdict["is_correct"],
            "weighted_score": weighted,
            "point_hits": verdict.get("point_hits", []),
            "reasoning": verdict.get("reasoning", ""),
            "judge_votes": {
                "correct": correct_votes,
                "wrong": self.num_runs - correct_votes,
            },
        }

    async def _judge(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        question_types: List[str],
        ask_time: str,
        score_points: List[dict],
    ) -> dict:
        """One judge call -> {is_correct, point_hits, reasoning}.

        Retries the LLM call when it returns an empty response, unparseable
        output, or a bad label (transient judge failures).
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            ask_time=ask_time,
            question_types=", ".join(question_types) if question_types else "（未标注）",
            reference_answer=reference_answer,
            score_points=_render_score_points(score_points),
            generated_answer=generated_answer,
        )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }

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

                if isinstance(data, dict) and "choices" in data:
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = str(data)

                if not content:
                    print(
                        f"  ⚠️ LLM Judge: empty response "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(1.0 * attempt)
                        continue
                    return {"is_correct": False, "point_hits": [], "reasoning": "empty response"}

                obj = self._parse_robust(content)
                if obj is None:
                    print(
                        f"  ⚠️ LLM Judge: unparseable "
                        f"(attempt {attempt}/{self.max_retries}); raw: {content[:160]}..."
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(1.0 * attempt)
                        continue
                    return {"is_correct": False, "point_hits": [], "reasoning": "unparseable"}

                label = str(obj.get("label", "")).strip().upper()
                if label not in ("CORRECT", "WRONG"):
                    print(
                        f"  ⚠️ LLM Judge: bad/empty label "
                        f"(attempt {attempt}/{self.max_retries}); raw: {content[:160]}..."
                    )
                    if attempt < self.max_retries:
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

            except Exception as e:
                print(
                    f"  ⚠️ LLM Judge failed (attempt {attempt}/{self.max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"is_correct": False, "point_hits": [], "reasoning": str(e)}

        return {"is_correct": False, "point_hits": [], "reasoning": "retries exhausted"}

    def _parse_robust(self, content: str):
        """Parse the judge response into {label, point_hits, reasoning}."""
        # Try JSON in code blocks first
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # Try raw JSON
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                pass

        # Regex fallback
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
