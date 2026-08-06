"""
Recall@K evaluation for LifeBench search results.

Evaluates whether ground-truth evidence items are covered by the top-K
retrieved chunks, using an LLM judge to determine semantic coverage.

Usage:
    python -m src.utils.recall_evaluator --results-dir results/lifebench-hindsight

    # Custom K values and max concurrent:
    python -m src.utils.recall_evaluator --results-dir results/lifebench-hindsight \\
        --k-values 5,10,15,20 --concurrency 20
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------

SOURCE_LABELS = {
    "calendar": "日历事件",
    "sms": "短信",
    "note": "笔记",
    "agent_chat": "智能助手对话",
    "call": "通话记录",
    "photo": "照片",
    "push": "推送通知",
    "email": "邮件",
}


def _fmt_evidence(ev: dict, index: int) -> str:
    """Render one evidence item for the LLM prompt."""
    source = ev.get("source", "")
    label = SOURCE_LABELS.get(source, source)
    session_date = ev.get("session_date", "")
    raw = ev.get("raw_data", {}) or {}

    parts = [f"证据{index + 1}. [{label}] 日期: {session_date}"]

    if source == "calendar":
        title = raw.get("title", "")
        desc = raw.get("description", "")
        start = raw.get("start_time", "")
        end = raw.get("end_time", "")
        parts.append(f"   标题: {title}")
        if desc:
            parts.append(f"   描述: {desc}")
        if start:
            parts.append(f"   时间: {start} ~ {end}")

    elif source == "sms":
        contact = raw.get("contactName", "")
        phone = raw.get("phoneNumber", "")
        content = raw.get("message_content", "")
        dt = raw.get("datetime", "")
        parts.append(f"   联系人: {contact} ({phone})")
        parts.append(f"   内容: {content}")
        if dt:
            parts.append(f"   时间: {dt}")

    elif source == "note":
        title = raw.get("title", "")
        content = raw.get("content", "")
        dt = raw.get("datetime", "")
        parts.append(f"   标题: {title}")
        parts.append(f"   内容: {content}")
        if dt:
            parts.append(f"   时间: {dt}")

    elif source == "agent_chat":
        conv = raw.get("conversation", {})
        if isinstance(conv, dict):
            for turn_key, turn_val in conv.items():
                if isinstance(turn_val, dict):
                    user = turn_val.get("user", {}) or {}
                    agent = turn_val.get("agent", {}) or {}
                    user_action = user.get("action", "")
                    user_content = user.get("content", "")
                    agent_content = agent.get("content", "")
                    if user_action:
                        parts.append(f"   {turn_key} user({user_action}): {user_content}")
                    else:
                        parts.append(f"   {turn_key} user: {user_content}")
                    if agent_content:
                        parts.append(f"   {turn_key} agent: {agent_content}")
        else:
            parts.append(f"   内容: {json.dumps(raw, ensure_ascii=False)}")

    elif source == "call":
        contact = raw.get("contactName", "")
        phone = raw.get("phoneNumber", "")
        duration = raw.get("durationSeconds", "")
        dt = raw.get("datetime", "")
        parts.append(f"   联系人: {contact} ({phone})")
        parts.append(f"   通话时长: {duration}秒")
        if dt:
            parts.append(f"   时间: {dt}")

    elif source == "photo":
        desc = raw.get("description", "")
        location = raw.get("location", "")
        dt = raw.get("datetime", "")
        if desc:
            parts.append(f"   描述: {desc}")
        if location:
            parts.append(f"   地点: {location}")
        if dt:
            parts.append(f"   时间: {dt}")

    elif source == "push":
        title = raw.get("title", "")
        content = raw.get("content", "")
        dt = raw.get("datetime", "")
        parts.append(f"   标题: {title}")
        parts.append(f"   内容: {content}")
        if dt:
            parts.append(f"   时间: {dt}")

    else:
        parts.append(f"   raw_data: {json.dumps(raw, ensure_ascii=False)}")

    return "\n".join(parts)


def _fmt_search_result(result: dict, index: int) -> str:
    """Render one search result for the LLM prompt."""
    content = result.get("content", "")
    score = result.get("score", 0)
    return f"检索{index + 1}. [score: {score:.4f}] {content}"


# ---------------------------------------------------------------------------
# LLM Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = textwrap.dedent("""\
你是记忆检索系统的评估员。你需要判断检索结果与证据项之间的**全部覆盖关系**。

判断"覆盖"的标准：
- 检索结果包含了该证据项的核心事实信息：事件内容、人物、时间、关键数据等
- 不要求逐字相同，语义等价、改写、合理归纳都算覆盖
- 检索结果必须是"实质包含"，不能只是话题沾边但没给出具体信息
- 例：证据项说"2月16日粤绣在线分享，展示《牡丹图》"，检索结果提到"粤绣分享会展示了牡丹图"
  或"2月在线分享粤绣作品"都算覆盖；只提"参加了粤绣活动"但没说具体内容＝不算覆盖

输出必须是严格的JSON，不要额外文字。""")

JUDGE_USER_TEMPLATE = textwrap.dedent("""\
## 问题
{question}

## 参考答案
{reference_answer}
（参考答案标识了问题的核心事实。判断覆盖时，优先关注证据中与答案事实相关的部分，
忽略证据中与答案无关的次要细节。答案本身不参与覆盖判断，仅用于帮你聚焦关键信息。）

## 证据项（共{evidence_count}条，编号证据1～证据{evidence_count}）
{evidence_text}

## 检索结果（共{result_count}条，按相关性从高到低排序）
{results_text}

## 任务
对每条证据项，找出所有覆盖它的检索结果。检索结果序号从1开始。
多条检索结果可能同时覆盖同一条证据项；一条检索结果也可能覆盖多条证据项。
如果无任何结果覆盖某证据项，对应位置填空数组[]。

## 输出格式
{{"evidence_ranks": [[3, 7, 12], [1], [], [5, 8]]}}

evidence_ranks 是一个数组，长度为{evidence_count}。
- evidence_ranks[0] 对应证据1，evidence_ranks[1] 对应证据2，依此类推。
- 每个元素是一个整数数组（可能为空），数组内的检索结果序号从1开始。

只输出JSON。""")


# ---------------------------------------------------------------------------
# Quality judge prompt — can the question be answered from retrieved results?
# ---------------------------------------------------------------------------

QUALITY_SYSTEM = textwrap.dedent("""\
你是记忆检索系统的质量评估员。判断给定的检索结果是否包含了回答问题的充足信息。

标准：
- 检索结果中包含回答问题所需的核心事实信息 → 充足
- 检索结果缺失关键事实、只有话题沾边的内容、或完全无关 → 不充足
- **不需要覆盖所有证据项**：只要检索结果中的信息足以支撑参考答案的核心内容，即为充足
- 不要求检索结果本身已经组织成完整答案，只要求其中包含了足够的信息片段
- 证据项和参考答案仅供你理解"回答该问题需要什么样的信息"，不参与充足性判断

只输出JSON，不要额外文字。""")

QUALITY_USER_TEMPLATE = textwrap.dedent("""\
## 问题
{question}

## 参考答案（仅供理解问题核心，也是判断充足性的最终标准）
{reference_answer}

## 证据项（回答该问题可能需要覆盖的关键信息，共{evidence_count}条）
{evidence_text}

## 检索结果（共{result_count}条，按相关性从高到低排序）
{results_text}

## 任务
1. 逐条累加阅读检索结果（从检索1开始），判断读到第几条时检索结果中的信息已经**足以支撑参考答案**。输出第一条使信息充足的检索结果序号（从1开始）。如果所有结果读完仍不足以支撑参考答案，则输出0。
2. 列出所有对回答问题**有实质帮助**的检索结果的序号（supporting_ranks），按升序排列。这些是包含关键事实、可直接用于回答问题的结果。如果一条都没有，输出空数组[]。

**注意**：不需要覆盖全部证据项。只要检索结果提供了足够信息能得出参考答案的核心结论，就是充足。

## 输出格式
{{"min_sufficient_rank": 3, "supporting_ranks": [1, 3]}}

只输出JSON。""")


# ---------------------------------------------------------------------------
# Evidence mapping loader
# ---------------------------------------------------------------------------

MAPPING_FILE = "question_id_to_evidence_mapping.json"
BUILD_SCRIPT = "build_question_evidence_mapping.py"


def load_evidence_mapping(raw_dir: Path) -> Dict[str, List[dict]]:
    """
    Load question_id -> evidence_list mapping from JSON file.
    If the mapping file doesn't exist, run the build script to generate it.
    """
    mapping_path = raw_dir / MAPPING_FILE
    build_path = raw_dir / BUILD_SCRIPT

    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        if not build_path.exists():
            raise FileNotFoundError(
                f"Build script not found: {build_path}. "
                f"Cannot generate evidence mapping."
            )
        print(f"Running build script: {build_path}")
        result = subprocess.run(
            [sys.executable, str(build_path)],
            cwd=str(raw_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Build script failed:\n{result.stderr}"
            )
        print(result.stdout)
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Mapping file still not found after running build script."
            )

    print(f"Loading evidence mapping from: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class RecallJudge:
    """Calls LLM to judge evidence coverage in search results."""

    def __init__(self, config: dict):
        self.model = "deepseek-v4-pro"
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.deepseek.com")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.0)
        self.max_retries = config.get("max_retries", 3)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            conn = aiohttp.TCPConnector(limit=100)
            self._session = aiohttp.ClientSession(connector=conn)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def judge(
        self,
        question: str,
        reference_answer: str,
        evidence_items: List[dict],
        search_results: List[dict],
        max_rank: int,
    ) -> Optional[Tuple[List[List[int]], List[int]]]:
        """
        Returns (evidence_ranks, result_evidence_counts) or None on failure.
        evidence_ranks[i]: list of ALL ranks (1-indexed) covering evidence i, [] = none.
        result_evidence_counts[j]: how many evidence items search result j+1 covers.
        """
        if not evidence_items:
            return [], []

        result_count = min(len(search_results), max_rank)

        evidence_text = "\n\n".join(
            _fmt_evidence(ev, i) for i, ev in enumerate(evidence_items)
        )
        results_text = "\n".join(
            _fmt_search_result(r, i) for i, r in enumerate(search_results[:max_rank])
        )

        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            evidence_count=len(evidence_items),
            evidence_text=evidence_text,
            result_count=result_count,
            results_text=results_text,
        )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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

                content = data["choices"][0]["message"]["content"]
                result = self._parse_response(content, len(evidence_items), result_count)
                if result is not None:
                    return result

                if attempt < self.max_retries:
                    await asyncio.sleep(2 * attempt)
                    continue

            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  [RETRY {attempt}/{self.max_retries}] recall judge: {type(e).__name__}: {str(e)[:120]}")
                    await asyncio.sleep(2 * attempt)
                    continue
                print(f"  [ERROR] Recall judge failed after {self.max_retries} attempts: {type(e).__name__}: {e}")

        return None

    def _parse_response(
        self, content: str, expected_ev: int, expected_results: int
    ) -> Optional[Tuple[List[List[int]], List[int]]]:
        """Parse LLM JSON response → (evidence_ranks, result_evidence_counts)."""
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        obj = None
        if m:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        if obj is None:
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end > start:
                try:
                    obj = json.loads(content[start: end + 1])
                except json.JSONDecodeError:
                    pass
        if obj is None:
            print(f"  [WARN] Unparseable response: {content[:200]}...")
            return None

        # Parse evidence_ranks: list of lists of ints
        ev_raw = obj.get("evidence_ranks")
        if not isinstance(ev_raw, list):
            # Fallback to old evidence_first_ranks format
            ev_raw = obj.get("evidence_first_ranks", [])
            if isinstance(ev_raw, list) and ev_raw and isinstance(ev_raw[0], (int, float)):
                # Convert [3, 0, 1] → [[3], [], [1]]
                ev_raw = [[int(r)] if int(r) > 0 else [] for r in ev_raw]

        evidence_ranks: List[List[int]] = []
        if isinstance(ev_raw, list):
            for entry in ev_raw:
                if isinstance(entry, list):
                    ranks = sorted(set(int(r) for r in entry if int(r) > 0))
                elif isinstance(entry, (int, float)):
                    v = int(entry)
                    ranks = [v] if v > 0 else []
                else:
                    ranks = []
                evidence_ranks.append(ranks)

        # Pad or truncate
        while len(evidence_ranks) < expected_ev:
            evidence_ranks.append([])
        evidence_ranks = evidence_ranks[:expected_ev]

        # Always derive result_evidence_counts from evidence_ranks.
        # LLMs sometimes make off-by-one errors in the redundant
        # result_evidence_counts field; evidence_ranks is the
        # authoritative signal.
        result_counts = [0] * expected_results
        for ranks in evidence_ranks:
            for r in ranks:
                if 1 <= r <= expected_results:
                    result_counts[r - 1] += 1

        return evidence_ranks, result_counts

    async def judge_quality(
        self,
        question: str,
        reference_answer: str,
        evidence_items: List[dict],
        search_results: List[dict],
        max_rank: int,
    ) -> Optional[Tuple[int, List[int]]]:
        """Returns (min_sufficient_rank, supporting_ranks) or None on failure."""
        result_count = min(len(search_results), max_rank)
        if result_count == 0:
            return 0, []

        evidence_text = "\n\n".join(
            _fmt_evidence(ev, i) for i, ev in enumerate(evidence_items)
        )
        results_text = "\n".join(
            _fmt_search_result(r, i) for i, r in enumerate(search_results[:max_rank])
        )

        user_prompt = QUALITY_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            evidence_count=len(evidence_items),
            evidence_text=evidence_text,
            result_count=result_count,
            results_text=results_text,
        )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": QUALITY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
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

                content = data["choices"][0]["message"]["content"]
                result = self._parse_quality_response(content)
                if result is not None:
                    return result

                if attempt < self.max_retries:
                    await asyncio.sleep(2 * attempt)
                    continue

            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  [RETRY {attempt}/{self.max_retries}] quality judge: {type(e).__name__}: {str(e)[:120]}")
                    await asyncio.sleep(2 * attempt)
                    continue
                print(f"  [ERROR] Quality judge failed after {self.max_retries} attempts: {type(e).__name__}: {e}")

        return None

    def _parse_quality_response(self, content: str) -> Optional[Tuple[int, List[int]]]:
        """Parse quality judge response → (min_sufficient_rank, supporting_ranks) or None."""
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        obj = None
        if m:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        if obj is None:
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end > start:
                try:
                    obj = json.loads(content[start: end + 1])
                except json.JSONDecodeError:
                    pass
        if obj is None:
            print(f"  [WARN] Unparseable quality response: {content[:200]}...")
            return None

        rank = obj.get("min_sufficient_rank")
        if isinstance(rank, (int, float)):
            rank = max(0, int(rank))
        else:
            return None

        raw = obj.get("supporting_ranks", [])
        if isinstance(raw, list):
            supporting = sorted(set(int(r) for r in raw if isinstance(r, (int, float)) and int(r) > 0))
        else:
            supporting = []

        return rank, supporting


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

async def evaluate_recall(
    results_dir: str,
    k_values: List[int],
    concurrency: int,
    llm_config: dict,
    resume: bool = True,
    output_path: Optional[str] = None,
    skip_quality: bool = False,
) -> Dict[int, float]:
    """
    Compute recall@K for memory retrieval results.

    Supports checkpoint/resume to avoid re-running completed LLM calls.
    """
    base = Path(__file__).parent.parent.parent
    results_path = Path(results_dir) / "search_results.json"
    checkpoint_path = Path(results_dir) / "recall_checkpoint.json"

    if not results_path.exists():
        raise FileNotFoundError(f"search_results.json not found at {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        search_data = json.load(f)

    # Load evidence mapping (auto-generate if missing)
    raw_dir = base / "datasets" / "lifebench_raw"
    evidence_lookup = load_evidence_mapping(raw_dir)
    print(f"  Loaded evidence for {len(evidence_lookup)} question_ids")

    # Build conv qid → answer map
    conv_path = base / "datasets" / "lifebench_locomo_format" / \
                "lifebench_locomo_conversation_format_v2.0_3380QA.json"
    with open(conv_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    answer_map: Dict[str, str] = {}
    for person in conv_data:
        for qa in person["qa"]:
            answer_map[qa["question_id"]] = qa.get("answer", "")

    judge = RecallJudge(llm_config)
    semaphore = asyncio.Semaphore(concurrency)
    max_k = max(k_values)

    items = []
    for sr in search_data:
        qid = sr["question_id"]
        evidence = evidence_lookup.get(qid, [])
        if evidence:
            items.append((sr, evidence))

    # Load checkpoint
    completed: Dict[str, Dict[str, List[int]]] = {}
    if resume and checkpoint_path.exists():
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
            print(f"Resumed {len(completed)} completed from checkpoint")

    pending = [
        (sr, ev) for sr, ev in items
        if sr["question_id"] not in completed
    ]

    print(f"\nEvaluating recall & precision: {len(pending)} pending, {len(completed)} cached")
    print(f"  (skipped {len(search_data) - len(items)} with 0 evidence)")
    print(f"  Max K: {max_k}, Concurrency: {concurrency}")

    pbar = tqdm(total=len(pending), desc="Judging")
    failures = 0

    save_every = max(1, concurrency * 5)
    done_since_save = 0
    checkpoint_lock = asyncio.Lock()

    async def judge_one(sr: dict, evidence: List[dict]):
        nonlocal done_since_save
        async with semaphore:
            qid = sr["question_id"]
            question = sr.get("query", "")
            answer = answer_map.get(qid, "")
            search_results = sr.get("results", [])

            if not search_results:
                ev_ranks = [[] for _ in evidence]
                res_counts = [0] * min(len(search_results), max_k)
                min_sufficient_rank = -1
                supporting_ranks = []
            else:
                recall_task = judge.judge(
                    question=question,
                    reference_answer=answer,
                    evidence_items=evidence,
                    search_results=search_results,
                    max_rank=max_k,
                )
                if skip_quality:
                    quality_task = asyncio.sleep(0)
                else:
                    quality_task = judge.judge_quality(
                        question=question,
                        reference_answer=answer,
                        evidence_items=evidence,
                        search_results=search_results,
                        max_rank=len(search_results),
                    )
                recall_result, quality_result = await asyncio.gather(
                    recall_task, quality_task,
                )
                if recall_result is None:
                    ev_ranks = [[] for _ in evidence]
                    res_counts = [0] * min(len(search_results), max_k)
                else:
                    ev_ranks, res_counts = recall_result
                if skip_quality:
                    min_sufficient_rank = -1
                    supporting_ranks = []
                elif quality_result is not None:
                    min_sufficient_rank, supporting_ranks = quality_result
                else:
                    min_sufficient_rank = -1
                    supporting_ranks = []

            completed[qid] = {
                "evidence_ranks": ev_ranks,
                "result_counts": res_counts,
                "min_sufficient_rank": min_sufficient_rank,
                "supporting_ranks": supporting_ranks,
            }
            pbar.update(1)

            async with checkpoint_lock:
                nonlocal done_since_save
                done_since_save += 1
                if done_since_save >= save_every:
                    _save_checkpoint(checkpoint_path, completed)
                    done_since_save = 0

    if pending:
        tasks = [judge_one(sr, ev) for sr, ev in pending]
        await asyncio.gather(*tasks)

    pbar.close()
    await judge.close()

    # Final save
    _save_checkpoint(checkpoint_path, completed)

    # Reconstruct ordered results
    all_results: List[Tuple[str, List[List[int]], List[int], int, List[int]]] = []
    for sr, ev in items:
        qid = sr["question_id"]
        default = {
            "evidence_ranks": [[] for _ in ev],
            "result_counts": [0] * max_k,
            "min_sufficient_rank": -1,
            "supporting_ranks": [],
        }
        entry = completed.get(qid, default)
        all_results.append((
            qid, entry["evidence_ranks"], entry["result_counts"],
            entry.get("min_sufficient_rank", -1),
            entry.get("supporting_ranks", []),
        ))

    # Compute metrics
    print(f"\n{'=' * 60}")
    print("Recall & Precision @K Results")
    print(f"{'=' * 60}")

    summary = {"k_values": k_values, "per_question": []}

    for qid, ev_ranks, res_counts, min_sr, supporting in all_results:
        entry = {
            "question_id": qid,
            "evidence_ranks": ev_ranks,
            "result_evidence_counts": res_counts,
            "min_sufficient_rank": min_sr,
            "supporting_ranks": supporting,
        }
        for k in k_values:
            first_ranks = [r[0] if r else 0 for r in ev_ranks]
            found = sum(1 for r in first_ranks if 1 <= r <= k)
            entry[f"strict_recall@{k}"] = found / len(ev_ranks) if ev_ranks else 0.0
            relevant_in_k = sum(1 for c in res_counts[:k] if c > 0)
            entry[f"strict_precision@{k}"] = relevant_in_k / k
            # Quality@K: answerable if 1 <= min_sufficient_rank <= k
            entry[f"quality@{k}"] = 1 if (min_sr >= 1 and min_sr <= k) else 0
        # Quality@all: answerable at any rank (based on full result set)
        entry["quality@all"] = 1 if min_sr >= 1 else 0
        summary["per_question"].append(entry)

    # Compute aggregate & print
    total_with_evidence = len(all_results)
    for k in k_values:
        total_ev = sum(len(r[1]) for r in all_results)
        first_ranks_all = []
        for _, ev_ranks, _, _, _ in all_results:
            first_ranks_all.extend([r[0] if r else 0 for r in ev_ranks])
        strict_recall_sum = sum(1 for r in first_ranks_all if 1 <= r <= k)
        strict_recall = strict_recall_sum / total_ev if total_ev else 0.0

        strict_prec_sum = 0.0
        for _, _, res_counts, _, _ in all_results:
            strict_prec_sum += sum(1 for c in res_counts[:k] if c > 0)
        n_q = len(all_results)
        strict_prec = strict_prec_sum / (k * n_q) if n_q else 0.0

        quality_count = sum(
            1 for _, _, _, min_sr, _ in all_results
            if min_sr >= 1 and min_sr <= k
        )
        quality = quality_count / n_q if n_q else 0.0

        all_found = sum(
            1 for _, ev_ranks, _, _, _ in all_results
            if ev_ranks and all((ev_ranks[i] or [0])[0] >= 1 and (ev_ranks[i] or [0])[0] <= k
                               for i in range(len(ev_ranks)))
        )
        partial_found = sum(
            1 for _, ev_ranks, _, _, _ in all_results
            if ev_ranks and any(r and r[0] >= 1 and r[0] <= k for r in ev_ranks)
        )

        print(f"  --- Recall@{k:>2d} ---")
        print(f"    Strict:     {strict_recall:.4f} ({strict_recall*100:.1f}%)")
        print(f"  --- Precision@{k:>2d} ---")
        print(f"    Strict:     {strict_prec:.4f} ({strict_prec*100:.1f}%)")
        print(f"  --- Quality@{k:>2d} ---")
        print(f"    Answerable: {quality:.4f} ({quality*100:.1f}%)")
        print(f"  Hit@{k:>2d} (all): {all_found/total_with_evidence:.4f} ({all_found/total_with_evidence*100:.1f}%)")
        print(f"  Hit@{k:>2d} (any): {partial_found/total_with_evidence:.4f} ({partial_found/total_with_evidence*100:.1f}%)")
        print()

    # Quality@all — across full result set
    quality_all = sum(
        1 for _, _, _, min_sr, _ in all_results
        if min_sr >= 1
    ) / n_q if n_q else 0.0
    print(f"  --- Quality@all ---")
    print(f"    Answerable: {quality_all:.4f} ({quality_all*100:.1f}%)")
    print()

    # Save detailed output
    if output_path is None:
        output_path = str(Path(results_dir) / "recall_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to: {output_path}")


def _save_checkpoint(path: Path, completed: Dict[str, Dict]):
    """Atomically save completed judgments to checkpoint file."""
    tmp = str(path) + ".tmp"
    entries = [
        {
            "question_id": qid,
            "evidence_ranks": v.get("evidence_ranks", []),
            "result_counts": v.get("result_counts", []),
            "min_sufficient_rank": v.get("min_sufficient_rank", -1),
            "supporting_ranks": v.get("supporting_ranks", []),
        }
        for qid, v in completed.items()
    ]
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"completed": entries}, f, ensure_ascii=False)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recall@K evaluation for LifeBench")
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to results directory containing search_results.json"
    )
    parser.add_argument(
        "--k-values", default="5,10,15,20",
        help="Comma-separated K values (default: 5,10,15,20)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=15,
        help="Max concurrent LLM calls (default: 15)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore checkpoint and start fresh"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save detailed per-question results JSON"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be evaluated without calling LLM"
    )
    parser.add_argument(
        "--no-quality", action="store_true",
        help="Skip retrieval quality (answerability) judge"
    )
    parser.add_argument(
        "--env-file", default=None,
        help="Path to .env file (default: auto-detect from project root)"
    )
    args = parser.parse_args()

    # Load .env
    base = Path(__file__).parent.parent.parent
    env_path = args.env_file or str(base / ".env")
    load_dotenv(env_path)

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    llm_config = {
        "model": "deepseek-v4-pro",
        "api_key": os.getenv("LLM_API_KEY", ""),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
    }

    print(f"Recall evaluator starting:")
    print(f"  Results dir: {args.results_dir}")
    print(f"  K values: {k_values}")
    print(f"  LLM: {llm_config['model']} @ {llm_config['base_url']}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Quality: {'disabled' if args.no_quality else 'enabled'}")
    if args.dry_run:
        print(f"  DRY RUN - no LLM calls")
    print()

    if args.dry_run:
        base = Path(__file__).parent.parent.parent
        results_path = Path(args.results_dir) / "search_results.json"
        with open(results_path, "r", encoding="utf-8") as f:
            search_data = json.load(f)
        raw_dir = base / "datasets" / "lifebench_raw"
        evidence_lookup = load_evidence_mapping(raw_dir)
        with_evidence = sum(1 for sr in search_data if evidence_lookup.get(sr["question_id"]))
        without_evidence = len(search_data) - with_evidence
        total_evidence = sum(len(evidence_lookup.get(sr["question_id"], [])) for sr in search_data)
        print(f"  Questions total: {len(search_data)}")
        print(f"  With evidence: {with_evidence}")
        print(f"  Without evidence (skipped): {without_evidence}")
        print(f"  Total evidence items: {total_evidence}")
        print(f"  Estimated LLM calls: {with_evidence} (recall) + {0 if args.no_quality else with_evidence} (quality)")
        return

    asyncio.run(evaluate_recall(
        results_dir=args.results_dir,
        k_values=k_values,
        concurrency=args.concurrency,
        llm_config=llm_config,
        resume=not args.no_resume,
        output_path=args.output,
        skip_quality=args.no_quality,
    ))


if __name__ == "__main__":
    main()
