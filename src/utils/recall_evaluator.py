"""
Coverage + answerability evaluation for LifeBench search results.

Single-pass LLM judge: for every question, feed the question, reference answer,
ALL evidence items, and ALL search results in ONE prompt. The judge performs
two tasks together:

  1. Evidence-result matching — for every evidence item, list which search
     result(s) contain its "main info" (the facts needed to answer the
     question). A result covers an evidence if it contains that evidence's
     main info — one result, or several results combined.
  2. Answerability            — whether the results as a whole can answer the
     question (binary 0 / 1).

Coverage, recall and precision are then *derived from the matching*:
  - covered[i] = 1 if evidence i matched at least one result
  - recall     = macro-average coverage (mean of per-question coverage)
  - precision  = (distinct results covering >=1 evidence) / (total results)

No ranking is measured. The judge sees every result, and "coverage" is
deliberately union-based so systems that split one fact across multiple
retrieval units (graph fragments, edge+entity pairs, consecutive turns) are
not penalized.

Usage:
    python -m src.utils.recall_evaluator --results-dir results/lifebench-hindsight

    # Concurrency / resume:
    python -m src.utils.recall_evaluator --results-dir results/lifebench-hindsight \\
        --concurrency 20
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Token counting (for memory-unit granularity reporting)
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback heuristic below
    _TOKENIZER = None


def _count_tokens(text: str) -> int:
    """Token count for one memory unit (CJK-aware fallback if tiktoken missing)."""
    if not text:
        return 0
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    cjk = len(re.findall(r"[一-鿿]", text))
    rest = re.sub(r"[一-鿿]", " ", text)
    return cjk + len(re.findall(r"\S+", rest))


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
                    assistant = turn_val.get("assistant", {}) or {}
                    user_action = user.get("action", "")
                    user_content = user.get("content", "")
                    assistant_content = assistant.get("content", "")
                    if user_action:
                        parts.append(f"   {turn_key} user({user_action}): {user_content}")
                    else:
                        parts.append(f"   {turn_key} user: {user_content}")
                    if assistant_content:
                        parts.append(f"   {turn_key} assistant: {assistant_content}")
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
# Single LLM Judge prompt (evidence coverage + answerability)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = textwrap.dedent("""\
你是记忆检索系统的评估员。对每个问题，你需要同时完成两项判定：证据-结果匹配、可回答性。

【关于检索单元粒度——务必先读】
不同记忆系统返回的"检索结果"粒度差异很大，可能是：
- 一条简洁的事实（如"3月2日参加了陶艺课"）；
- 一整天的日记/时间线（把当天多个事件混在一起，夹杂大量与本题无关的内容）；
- 一个图谱节点或碎片（只含半句话，需要多条拼合）；
- 一段原始对话或一条蒸馏后的记忆。
判断时**只关心"答案/证据的关键事实点是否出现在单元里"**，不要因为单元粗大、夹杂无关
内容、或表述冗长就降低判定。关键事实点埋在无关内容里，也算覆盖/可回答。

【先思考，再下结论】在给出判定前，请按以下步骤在心里推理，不要跳过：
1. 读参考答案，提炼出答案的"关键事实点"（事件、人物、时间、关键数据等），这是你判断的锚点。
2. 逐条读证据项，明确每条证据"回答本题所需的主要信息"是什么（忽略与答案无关的次要细节）。
3. 逐条读检索结果，为每条证据找出哪些检索结果包含了它的主要信息（可一条、可多条拼合）。
4. 最后单独判断可回答性（见"二"）。可回答性是独立判断，不要因为"证据没有全部覆盖"就判 0。

一、证据-结果匹配：为每条证据，找出覆盖其"主要信息"的检索结果编号。
判断"覆盖"的标准：
- **只要一个记忆单元（检索结果）包含了该证据回答本题所需的主要信息，就认为覆盖**：
  即事件内容、人物、时间、关键数据等关键事实点。
- **不因粒度/冗长而减分**：单元可能是整日日记、长对话或图谱碎片，夹杂大量无关内容；
  只要主要信息确实出现在该单元里（哪怕只占一小段），就判为覆盖。
- 不要求包含证据的全部细节，也不要求逐字相同；语义等价、改写、合理归纳都算覆盖。
- 一条证据可能被 0 条、1 条或多条检索结果覆盖：
  - 若被多条结果拼合覆盖（例如一条给时间、另一条给事件内容），把所有这些结果的编号都列出。
  - 若未被覆盖，列出空数组 []。
- 一条检索结果也可能同时覆盖多条证据，允许它重复出现在不同证据的匹配列表里。
- 检索结果必须是"实质包含主要信息"，不能只是话题沾边但没给出具体信息。
- 例：证据项"2月16日粤绣在线分享，展示《牡丹图》"——一条结果提"粤绣分享会"、另一条提
  "展示了牡丹图"，两条拼合还原主要信息，则这两条编号都列入该证据的匹配列表。

二、可回答性：判断检索结果整体是否足以回答该问题（0 或 1）。
判断"可回答"的标准：
- **不要求覆盖所有证据**：只要参考答案的"关键事实点"已被检索结果（单条或多条拼合）
  提及或覆盖，就足以回答问题 → 1。
- **不因单元粗大/碎片化而判 0**：只要答案的关键事实点能在检索结果里找到，即使它埋在
  一长段夹杂无关内容的日记/对话里，也判为可回答。
- 只有当答案的关键事实点缺失、检索结果只有话题沾边或完全无关的内容时 → 0。
- 不要求检索结果本身已经组织成完整答案，只要求其中包含了回答所需的关键信息片段。

只输出JSON，不要额外文字。""")

JUDGE_USER_TEMPLATE = textwrap.dedent("""\
## 问题
{question}

## 参考答案
{reference_answer}
（参考答案标识了问题的核心事实。判断覆盖和可回答性时，优先关注与答案事实相关的部分，
忽略证据中与答案无关的次要细节。答案本身不参与覆盖判断，仅用于帮你聚焦关键信息。）

## 证据项（共{evidence_count}条，编号证据1～证据{evidence_count}）
{evidence_text}

## 检索结果（共{result_count}条，编号检索1～检索{result_count}）
{results_text}

## 任务
先按系统提示里的"思考流程"在心里推理，再输出结果：
1. 对每条证据项，找出覆盖其"主要信息"的检索结果编号（编号从1开始：检索1 记为 1，检索2 记为 2）。
   - 只要某个检索结果包含了该证据回答本题所需的主要信息（可一条、可多条拼合），
     就把该编号列入这条证据的匹配列表。
   - 检索单元可能是整日日记、图谱节点、长对话等粗粒度/碎片形式，夹杂无关内容很正常——
     只要主要信息确实出现在该单元里，就列入匹配，不要因无关内容多而漏掉。
   - 若没有任何结果覆盖该证据，则该证据对应空数组 []。
2. 判断检索结果整体是否足以回答该问题，输出 0 或 1。注意：**不要求覆盖所有证据**，
   只要答案的关键事实点已被提及/覆盖即可判 1，不要因单元粗大/碎片化而低估可回答性。

## 输出格式
{{"evidence_matches": [[1, 3], [], [2]], "answerable": 1, "reasoning": "answerable的判断依据，一句话"}}

- evidence_matches 是一个数组，长度必须等于证据数（{evidence_count}）。
  evidence_matches[0] 对应证据1，evidence_matches[1] 对应证据2，依此类推。
  每个元素是一个整数数组，列出覆盖该证据主要信息的检索结果编号（1-based）。
  - 编号从1开始：检索1 记为 1，检索2 记为 2，依此类推。
  - 一条证据未被覆盖时，该元素为 []。
  - 一条证据被多条结果拼合覆盖时，列出所有这些编号。
- answerable 必须是整数 0 或 1。
- reasoning 是 answerable 判断依据的一句话，简要说明即可，不要换行。

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

CHECKPOINT_FORMAT_VERSION = 4


class RecallJudge:
    """Calls LLM to judge evidence coverage + answerability in one pass."""

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
    ) -> Optional[Tuple[List[List[int]], int]]:
        """
        Returns (evidence_matches, answerable) or None.

        evidence_matches[i]: list of 0-based search-result indices that cover
                             evidence i's main info. Empty list = not covered.
        answerable: 0 or 1.
        """
        if not evidence_items:
            return [], 0

        result_count = len(search_results)

        evidence_text = "\n\n".join(
            _fmt_evidence(ev, i) for i, ev in enumerate(evidence_items)
        )
        results_text = "\n".join(
            _fmt_search_result(r, i) for i, r in enumerate(search_results)
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
                parsed = self._parse_response(content, len(evidence_items), result_count)
                if parsed is not None:
                    return parsed

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

    @staticmethod
    def _to_int(v) -> Optional[int]:
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _repair_json(text: str) -> Optional[str]:
        """Try to fix common LLM JSON bracket errors (flash models in particular).

        When bracket counts are off, strip excess ``]`` before the final ``}``.
        """
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        open_b = text.count("[")
        close_b = text.count("]")
        if close_b > open_b:
            excess = close_b - open_b
            rev = text[::-1]
            for _ in range(excess):
                m = re.search(r'(\s*)\]', rev)
                if m:
                    pos = m.start()
                    rev = rev[:pos] + rev[pos + 1:]
                else:
                    break
            text = rev[::-1]

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            return None

    def _parse_response(
        self, content: str, expected_ev: int, expected_results: int
    ) -> Optional[Tuple[List[List[int]], int]]:
        """Parse LLM JSON response → (evidence_matches, answerable) or None.

        evidence_matches[i] = list of 0-based result indices covering evidence i.
        """
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        obj = None
        if m:
            raw = m.group(1)
            fixed = self._repair_json(raw)
            if fixed:
                try:
                    obj = json.loads(fixed)
                except json.JSONDecodeError:
                    pass
        if obj is None:
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end > start:
                raw = content[start: end + 1]
                fixed = self._repair_json(raw)
                if fixed:
                    try:
                        obj = json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
        if obj is None:
            print(f"  [WARN] Unparseable response: {content[:200]}...")
            return None

        matches_raw = obj.get("evidence_matches")
        if not isinstance(matches_raw, list):
            print(f"  [WARN] Missing/invalid evidence_matches: {repr(matches_raw)[:120]}")
            return None

        evidence_matches: List[List[int]] = []
        for entry in matches_raw:
            idxs: List[int] = []
            if isinstance(entry, list):
                for v in entry:
                    n = self._to_int(v)
                    if n is None:
                        continue
                    # prompt uses 1-based numbering; convert to 0-based
                    idx = n - 1
                    if 0 <= idx < expected_results:
                        idxs.append(idx)
                seen = set()
                deduped = []
                for i in idxs:
                    if i not in seen:
                        seen.add(i)
                        deduped.append(i)
                idxs = deduped
            evidence_matches.append(idxs)

        while len(evidence_matches) < expected_ev:
            evidence_matches.append([])
        evidence_matches = evidence_matches[:expected_ev]

        answerable = self._parse_answerable(obj.get("answerable"), content)
        return evidence_matches, answerable

    @staticmethod
    def _parse_answerable(value, content: str) -> int:
        """Coerce answerable to 0/1. Lenient about bool/int/str, with regex fallback."""
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if int(value) == 1 else 0
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("1", "true", "yes", "是"):
                return 1
            return 0
        m = re.search(r'answerable["\s:]+([01])', content, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return 0


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

async def evaluate_recall(
    results_dir: str,
    concurrency: int,
    llm_config: dict,
    resume: bool = True,
    output_path: Optional[str] = None,
    clean: bool = False,
) -> None:
    """
    Compute evidence coverage rate + answerability rate for search results.

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

    if output_path is None:
        output_path = str(Path(results_dir) / "recall_results.json")

    if clean:
        for p in (checkpoint_path, Path(output_path)):
            if p.exists():
                p.unlink()
                print(f"  [clean] removed {p.name}")
        resume = False

    judge = RecallJudge(llm_config)
    semaphore = asyncio.Semaphore(concurrency)

    items = []
    for sr in search_data:
        qid = sr["question_id"]
        evidence = evidence_lookup.get(qid, [])
        if evidence:
            items.append((sr, evidence))

    # Load checkpoint (only if format matches)
    completed: Dict[str, Dict] = {}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if isinstance(ckpt, dict) and ckpt.get("format_version") == CHECKPOINT_FORMAT_VERSION:
            for entry in ckpt.get("completed", []):
                completed[entry["question_id"]] = {
                    "evidence_matches": entry.get("evidence_matches", []),
                    "answerable": entry.get("answerable", 0),
                }
            print(f"Resumed {len(completed)} completed from checkpoint")
        else:
            print("  [WARN] checkpoint format mismatch/legacy — starting fresh")

    pending = [
        (sr, ev) for sr, ev in items
        if sr["question_id"] not in completed
    ]

    print(f"\nEvaluating coverage / answerability: {len(pending)} pending, {len(completed)} cached")
    print(f"  (skipped {len(search_data) - len(items)} with 0 evidence)")
    print(f"  Concurrency: {concurrency}")

    pbar = tqdm(total=len(pending), desc="Judging")

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
                evidence_matches = [[] for _ in evidence]
                answerable = 0
            else:
                result = await judge.judge(
                    question=question,
                    reference_answer=answer,
                    evidence_items=evidence,
                    search_results=search_results,
                )
                if result is None:
                    evidence_matches = [[] for _ in evidence]
                    answerable = 0
                else:
                    evidence_matches, answerable = result

            completed[qid] = {
                "evidence_matches": evidence_matches,
                "answerable": answerable,
            }
            pbar.update(1)

            async with checkpoint_lock:
                nonlocal done_since_save
                done_since_save += 1
                if done_since_save >= save_every:
                    _save_checkpoint(checkpoint_path, completed)
                    _save_partial_results(output_path, items, completed)
                    done_since_save = 0

    if pending:
        tasks = [judge_one(sr, ev) for sr, ev in pending]
        await asyncio.gather(*tasks)

    pbar.close()
    await judge.close()

    # Final save
    _save_checkpoint(checkpoint_path, completed)

    summary = _build_coverage_results(items, completed)
    per_question = summary["per_question"]
    n_q = len(per_question)

    print(f"\n{'=' * 60}")
    print("Coverage / Answerability Results")
    print(f"{'=' * 60}")

    total_ev = sum(pq["num_evidence"] for pq in per_question)
    covered_ev = sum(pq["covered_count"] for pq in per_question)
    # macro-average: mean of per-question coverage rate (robust to evidence imbalance)
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
    recall = coverage_rate  # recall == macro-average coverage
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

    print(f"  Evidence coverage rate (macro): {coverage_rate:.4f} ({coverage_rate*100:.1f}%)  "
          f"(mean per-question)")
    print(f"  Evidence coverage rate (micro): {coverage_rate_micro:.4f} ({coverage_rate_micro*100:.1f}%)  "
          f"({covered_ev}/{total_ev})")
    print(f"  Recall (= macro coverage):      {recall:.4f} ({recall*100:.1f}%)")
    print(f"  Recall@5:                       {recall_at_5:.4f} ({recall_at_5*100:.1f}%)  "
          f"({covered_at_5}/{total_ev})")
    print(f"  Recall@20:                      {recall_at_20:.4f} ({recall_at_20*100:.1f}%)  "
          f"({covered_at_20}/{total_ev})")
    print(f"  Answerable rate:                {answerable_rate:.4f} ({answerable_rate*100:.1f}%)  "
          f"({answerable_count}/{n_q})")
    print(f"  Covered AND answerable:         {joint_rate:.4f} ({joint_rate*100:.1f}%)  "
          f"({fully_covered_answerable}/{n_q})")
    print(f"  Precision (result purity):      {precision:.4f} ({precision*100:.1f}%)  "
          f"({relevant_results}/{total_results} relevant)")
    print(f"  Precision@5:                    {precision_at_5:.4f} ({precision_at_5*100:.1f}%)  "
          f"({relevant_at_5}/{min(5, total_results)})")
    print(f"  Precision@20:                   {precision_at_20:.4f} ({precision_at_20*100:.1f}%)  "
          f"({relevant_at_20}/{min(20, total_results)})")
    print(f"  Redundancy:                     {redundancy:.4f} ({redundancy*100:.1f}%)")
    print(f"  Avg results/question:           {avg_results:.2f}  (avg relevant {avg_relevant:.2f})")
    print(f"  Avg tokens/memory unit:         {avg_tokens_per_unit:.1f}")
    print(f"  Avg tokens/question:            {avg_tokens_per_question:.1f}")
    print()

    by_source = summary["by_source"]
    if by_source:
        print("  Coverage rate by evidence source:")
        for src in sorted(by_source):
            s = by_source[src]
            print(f"    {src:>12s}: {s['coverage_rate']*100:6.2f}%  ({s['covered']}/{s['total']})")

    # Save detailed output
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


def _save_checkpoint(path: Path, completed: Dict[str, Dict]):
    """Atomically save completed judgments to checkpoint file."""
    tmp = str(path) + ".tmp"
    entries = [
        {
            "question_id": qid,
            "evidence_matches": v.get("evidence_matches", []),
            "answerable": v.get("answerable", 0),
        }
        for qid, v in completed.items()
    ]
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"format_version": CHECKPOINT_FORMAT_VERSION, "completed": entries},
                  f, ensure_ascii=False)
    os.replace(tmp, str(path))


def _build_coverage_results(
    items: List[tuple],
    completed: Dict[str, Dict],
) -> dict:
    """Build coverage results dict from completed judgments. Always includes all items."""
    per_question = []
    by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "covered": 0})

    for sr, ev in items:
        qid = sr["question_id"]
        num_results = len(sr.get("results", []))
        default = {
            "evidence_matches": [[] for _ in ev],
            "answerable": 0,
        }
        entry = completed.get(qid, default)
        evidence_matches = entry.get("evidence_matches", [[] for _ in ev])
        answerable = entry.get("answerable", 0)

        # Normalize length (checkpoint corruption safety)
        if len(evidence_matches) != len(ev):
            evidence_matches = list(evidence_matches) + [
                [] for _ in range(max(0, len(ev) - len(evidence_matches)))
            ]
            evidence_matches = evidence_matches[: len(ev)]

        # Derive covered + relevant directly from the matching
        covered: List[int] = []
        relevant = [0] * num_results
        for matches in evidence_matches:
            covered.append(1 if matches else 0)
            for idx in matches:
                if isinstance(idx, int) and 0 <= idx < num_results:
                    relevant[idx] = 1

        covered_count = sum(covered)
        relevant_count = sum(relevant)
        sources = [e.get("source", "?") for e in ev]
        total_tokens = sum(_count_tokens(r.get("content", "")) for r in sr.get("results", []))

        # Recall@K: an evidence is covered only if a match lands in the top-K results
        covered_at_5 = sum(
            1 for matches in evidence_matches
            if any(isinstance(idx, int) and 0 <= idx < 5 for idx in matches)
        )
        covered_at_20 = sum(
            1 for matches in evidence_matches
            if any(isinstance(idx, int) and 0 <= idx < 20 for idx in matches)
        )

        # Precision@K: fraction of the top-K results that cover >=1 evidence
        relevant_at_5 = sum(relevant[:5])
        relevant_at_20 = sum(relevant[:20])

        pq = {
            "question_id": qid,
            "num_evidence": len(ev),
            "covered_count": covered_count,
            "coverage_rate": covered_count / len(ev) if ev else 0.0,
            "covered": covered,
            "evidence_matches": evidence_matches,
            "sources": sources,
            "answerable": answerable,
            "num_results": num_results,
            "relevant_count": relevant_count,
            "precision": relevant_count / num_results if num_results else 0.0,
            "total_tokens": total_tokens,
            "avg_tokens_per_unit": total_tokens / num_results if num_results else 0.0,
            "covered_at_5": covered_at_5,
            "covered_at_20": covered_at_20,
            "recall_at_5": covered_at_5 / len(ev) if ev else 0.0,
            "recall_at_20": covered_at_20 / len(ev) if ev else 0.0,
            "relevant_at_5": relevant_at_5,
            "relevant_at_20": relevant_at_20,
            "precision_at_5": relevant_at_5 / min(5, num_results) if num_results else 0.0,
            "precision_at_20": relevant_at_20 / min(20, num_results) if num_results else 0.0,
        }
        per_question.append(pq)

        for src, cov in zip(sources, covered):
            by_source[src]["total"] += 1
            if cov == 1:
                by_source[src]["covered"] += 1

    by_source_out = {
        src: {
            "total": s["total"],
            "covered": s["covered"],
            "coverage_rate": s["covered"] / s["total"] if s["total"] else 0.0,
        }
        for src, s in by_source.items()
    }

    return {"per_question": per_question, "by_source": by_source_out}


def _save_partial_results(
    output_path: str,
    items: List[tuple],
    completed: Dict[str, Dict],
):
    """Write current recall_results.json from completed work so far."""
    summary = _build_coverage_results(items, completed)
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Coverage + answerability evaluation for LifeBench")
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to results directory containing search_results.json"
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
        "--clean", action="store_true",
        help="Delete existing recall_checkpoint.json and recall_results.json first, "
             "then run from scratch"
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
        "--env-file", default=None,
        help="Path to .env file (default: auto-detect from project root)"
    )
    args = parser.parse_args()

    # Load .env
    base = Path(__file__).parent.parent.parent
    env_path = args.env_file or str(base / ".env")
    load_dotenv(env_path)

    llm_config = {
        "model": "deepseek-v4-pro",
        "api_key": os.getenv("LLM_API_KEY", ""),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
    }

    print(f"Coverage evaluator starting:")
    print(f"  Results dir: {args.results_dir}")
    print(f"  LLM: {llm_config['model']} @ {llm_config['base_url']}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Resume: {not args.no_resume}")
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
        print(f"  Estimated LLM calls: {with_evidence} (one combined judge call per question)")
        return

    asyncio.run(evaluate_recall(
        results_dir=args.results_dir,
        concurrency=args.concurrency,
        llm_config=llm_config,
        resume=not args.no_resume,
        output_path=args.output,
        clean=args.clean,
    ))


if __name__ == "__main__":
    main()
