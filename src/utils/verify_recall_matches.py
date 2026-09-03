"""
Verify coarse (GLM) evidence-result pair matches one-by-one with the env
default DeepSeek judge, and annotate Token Precision per verified pair.

Background
----------
`recall_evaluator.py` judged coverage/answerability with a deliberately
lenient single-prompt LLM ("topic-related structured labels count as
coverage"). For structured-fragment systems this inflates
recall: topic-matched but fact-free units are marked as covering evidence.

This script takes the coarse matches already saved in
`recall_checkpoint_fixed.json` and re-checks EVERY positive
(evidence -> result) pair individually: a strict verifier decides whether
the unit actually contains that evidence's key facts (same date/event/
person/value — NOT merely a shared topic keyword), and if yes, extracts the
verbatim minimal spans that substantiate the evidence. Token Precision for a
pair = tokens(its verified spans) / tokens(unit).

Only tightening is performed: GLM-negative/empty matches are never re-opened
(the cost of checking all evidence x all units is prohibitive), so corrected
recall <= coarse recall. Verdict semantics per pair = "this unit ALONE
substantively contains the evidence's main info"; evidence relying on
cross-unit composition may flip to uncovered.

Outputs (written next to the input files, e.g. results/lifebench-<sys>/):
  - recall_verify_checkpoint.json      per-pair verdicts, resumable
  - recall_verified_detail.json        per-question / per-pair detail
  - recall_results_verified.json       same schema as recall_results_fixed.json
                                       with corrected match-derived fields and
                                       added token_precision columns
                                       (micro = sum relevant / sum all tokens;
                                        macro = mean per-question TP)

Usage:
    python -m src.utils.verify_recall_matches --results-dir results/lifebench-hindsight
    # smoke test: verify only the first N pending pairs (real LLM calls)
    python -m src.utils.verify_recall_matches --results-dir ... --dry-run 20
"""

import argparse
import asyncio
import io
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    _count_tokens,
    _fmt_evidence,
    load_evidence_mapping,
)
from src.utils.rejudge_recall_failures import (  # noqa: E402
    _extract_json_candidates,
    _load_config,
    _repair_json,
)

CHECKPOINT_FORMAT_VERSION = 1
VERIFY_CHECKPOINT = "recall_verify_checkpoint.json"
VERIFY_DETAIL = "recall_verified_detail.json"
VERIFY_RESULTS = "recall_results_verified.json"


# ---------------------------------------------------------------------------
# Verifier prompt: one (question, evidence, unit) triple per call
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = textwrap.dedent("""\
你是记忆检索系统的"配对核实员"。一个宽松模型曾把"证据"与"检索结果"配成对，其中混有
大量"话题沾边"的误配。你的任务是一对一核实配对是否真正成立。原则：**严格核实，宁可
漏过，不可放过**——只有该结果真的承载了这条证据的具体事实才算成立。

背景说明：一条"证据"是该问题背后的一条原始记忆（日历/短信/笔记/对话/照片等，含日期
与原始内容）；一个"检索结果/记忆单元"是记忆系统为该问题返回的一个文本块，粒度可能很
粗：可能是一整天日记、一个以 Entity 开头的实体属性块、或一个图谱片段，夹杂大量与本题
无关的内容。粗粒度本身不扣分，只要证据的事实确实出现在其中。

成立（covered=1）的判据：
- 单元内确实出现该证据所记录的具体事实——**同一条**事件、同一人物/对象、同一数值/名称/
  结果，且**不是同主题下的另一个具体事件**。日期表述允许合理对应（"今天/2025-1-12"），
  但不允许把证据指向别的日期的事件。
- 该事实可以埋在一长段无关内容里，可以是被动语态改写、**忠实概括**（见下条），但必须
  能还原到该证据的具体事实，而不是只沾个主题。
- **忠实概括也算成立**：蒸馏型记忆系统常把多条同类记录合并成习惯性/跨度性表述——
  时间泛化为"每天/每晚/每周/每月/当周/当月/约一周/截至某日/当晚/近期/经常"等，
  但事件内容、人物、对象、数值、目的与证据一致，且单元中没有指向其他具体事件的
  矛盾细节。例如证据是"1月16日晨间跟王大夫视频做肩颈拉伸、酸胀缓解"，单元是
  "每天早晚各做5分钟肩颈拉伸、跟随王大夫的视频、以缓解肩颈酸胀"——同一事实的
  泛化，成立。

不成立（covered=0）的判据（务必逐条对照）：
1. 单元只通过标题/标签/实体名/主题词与证据相关（例如都叫"高级技师备考""体检"），但没
   有该证据本身的具体事实（例如证据是 1月12日的体检结果，单元里只有别的日期的体检）。
2. 单元记录的是同主题下的**另一个具体事件**——有自己的日期（哪怕是 1月12日以外的明确
   日期）或自己的内容细节，不是对证据的概括。例如证据是"3月13日学习高低压开关柜检修、
   记连锁保护逻辑"，单元只有"8月16日当天提到高级技师备考"——8月16日的事件与3月13日
   的证据不是概括关系；又例如单元说"某天做过一次体检"但没有证据的具体内容，也是泛指
   而非概括（无法还原证据的数值/结果）。
3. 单元只给出泛泛模糊提及（"关注过健康/学习过"），无法定位到该证据的具体事实，也不
   构成可还原证据细节的概括。
4. 单元内容与证据出自不同的对话/人/场景，只是相似话题。例如证据是用户 A 对助手说的话，
   单元是另一个人的同样内容——不算。

若成立，你必须输出单元内支撑该证据的**最小逐字片段**（spans）：
- 从单元原文中逐字复制，绝不改写、不 paraphrase、不按记忆补全、不拼接原文中没有的字符。
- 片段尽量短，但信息完整（足以让读者还原该证据的关键事实）。
- 一个证据的事实可能分散在多处，可输出多个片段；每段都必须是单元中实际连续出现的原文。
- 若事实在单元中多次重复出现，只给出第一次出现的位置即可。

只输出JSON，不要额外文字。""")

VERIFY_USER_TEMPLATE = textwrap.dedent("""\
## 问题
{question}

## 参考答案（仅用于理解该证据对本题的作用；不是判定对象，不需要在单元里逐字出现）
{reference_answer}

## 待核实的证据
{evidence}

## 待核实的检索单元
[单元开始]
{unit_content}
[单元结束]

## 任务
按系统提示的判据，判断该单元是否真实包含这条证据的具体事实，并输出：

{{"covered": 0或1, "spans": ["逐字片段", ...], "reasoning": "一句话说明成立/不成立的依据"}}

- covered=0 时，spans 必须为空数组 []。
- spans 的每个元素必须是"待核实的检索单元"原文（[单元开始]与[单元结束]之间）中实际连续
  出现的子串，逐字一致、含标点，不能是改写/拼接/补全。
- covered=1 时 spans 至少包含一个片段。

只输出JSON。""")

MAX_UNIT_CHARS = 20000
MAX_ANSWER_CHARS = 1500


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…[截断]"


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def _parse_verdict(content: str, unit_text: str) -> Optional[Tuple[int, List[str], str]]:
    """(covered, validated_spans, reasoning) or None."""
    for cand in _extract_json_candidates(content):
        obj = _repair_json(cand)
        if obj is None:
            continue
        covered_raw = obj.get("covered")
        if isinstance(covered_raw, bool):
            covered = 1 if covered_raw else 0
        elif isinstance(covered_raw, (int, float)):
            covered = 1 if int(covered_raw) == 1 else 0
        elif isinstance(covered_raw, str) and covered_raw.strip().lower() in ("1", "true", "是"):
            covered = 1
        else:
            covered = 0

        spans_raw = obj.get("spans", [])
        spans: List[str] = []
        if isinstance(spans_raw, list):
            for s in spans_raw:
                if not isinstance(s, str):
                    continue
                s = s.strip()
                if not s:
                    continue
                if s in unit_text:
                    if s not in spans:
                        spans.append(s)
        reasoning = obj.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)[:200] if reasoning else ""
        return covered, spans, reasoning
    return None


# ---------------------------------------------------------------------------
# Streaming loader for huge search_results.json (only keep needed questions)
# ---------------------------------------------------------------------------

def _iter_json_array_records(path: Path):
    """Stream top-level JSON array of objects from a possibly huge file.

    Yields one decoded record at a time; memory usage is bounded by the
    largest single record plus the read buffer.
    """
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0
    array_started = False
    in_array = False
    read_chunk = 16 * 1024 * 1024
    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(read_chunk)
            if chunk:
                buf += chunk
            # trim consumed prefix occasionally to keep buffer bounded
            if pos > 64 * 1024 * 1024:
                buf = buf[pos:]
                pos = 0

            if not array_started:
                i = 0
                while i < len(buf) and buf[i].isspace():
                    i += 1
                if i >= len(buf):
                    if not chunk:
                        break
                    continue
                if buf[i] != "[":
                    raise ValueError(f"expected '[' at start of {path}")
                pos = i + 1
                array_started = True
                continue

            # skip whitespace / comma separators
            while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ","):
                pos += 1
            if pos >= len(buf):
                if not chunk:
                    break
                continue
            if buf[pos] == "]":
                if not chunk:
                    return
                pos += 1
                # drain rest of file
                while f.read(read_chunk):
                    pass
                return
            if not in_array:
                in_array = True

            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if pos >= len(buf) - 4:
                    # record truncated at buffer end -> need more data
                    if not chunk:
                        raise
                    continue
                raise
            yield obj
            pos = end
            if not chunk and pos >= len(buf):
                break
    if in_array:
        return
    raise ValueError(f"file {path} ended before array closed")


def load_needed_search(search_path: Path, qids: set) -> Dict[str, dict]:
    """Load {qid: {query, results:[{content, score}]}} for qids in the set."""
    found: Dict[str, dict] = {}
    for rec in _iter_json_array_records(search_path):
        qid = rec.get("question_id")
        if qid not in qids:
            continue
        results = []
        for r in rec.get("results", []):
            content = r.get("content", "")
            if content:
                results.append({"content": content, "score": r.get("score", 0.0)})
        found[qid] = {"query": rec.get("query", ""), "results": results}
    return found


# ---------------------------------------------------------------------------
# Token precision helpers (char ranges over a unit, merged, counted w/ cl100k)
# ---------------------------------------------------------------------------

def _span_ranges(unit_text: str, spans: List[str]) -> List[Tuple[int, int]]:
    """First-occurrence char ranges of each span in unit_text (already validated)."""
    ranges = []
    for s in spans:
        idx = unit_text.find(s)
        if idx != -1:
            ranges.append((idx, idx + len(s)))
    ranges.sort()
    merged = []
    for s, e in ranges:
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _ranges_tokens(unit_text: str, ranges: List[Tuple[int, int]]) -> int:
    """Token count of merged char ranges (spliced text counted per fragment).

    A duplicated fact appears twice in the unit; only its first occurrence is
    counted as relevant (duplication is noise, which matches the "information
    density" semantics of Token Precision).
    """
    total = 0
    for s, e in ranges:
        total += _count_tokens(unit_text[s:e])
    return total


# ---------------------------------------------------------------------------
# Async verifier (mirrors RobustJudge in rejudge_recall_failures.py)
# ---------------------------------------------------------------------------

class PairVerifier:
    def __init__(self, config: dict, max_retries: int = 8):
        self.model = config["model"]
        self.api_key = config["api_key"]
        self.base_url = config["base_url"].rstrip("/")
        self.max_tokens = config["max_tokens"]
        self.temperature = 0.0
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            conn = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=360)
            self._session = aiohttp.ClientSession(connector=conn, timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def verify(
        self,
        question: str,
        reference_answer: str,
        evidence: dict,
        unit: dict,
    ) -> dict:
        """Verify one (evidence -> unit) pair. Returns verdict record."""
        unit_text = unit["content"]
        evidence_text = _fmt_evidence(evidence, 0)

        user_prompt = VERIFY_USER_TEMPLATE.format(
            question=question,
            reference_answer=_truncate(reference_answer, MAX_ANSWER_CHARS),
            evidence=evidence_text,
            unit_content=_truncate(unit_text, MAX_UNIT_CHARS),
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # deepseek-v4-flash emits a long hidden chain-of-thought
            # (reasoning_content) by default; it is unused cost, so disable it
            # (measured: completion 770 -> 140 tokens, wall 8.1s -> 2.4s).
            "thinking": {"type": "disabled"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        last_reason = "unknown"
        usage = {}
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
                usage = (data.get("usage") or {}).get("prompt_tokens", 0)
                usage = {
                    "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens", 0),
                    "completion_tokens": (data.get("usage") or {}).get("completion_tokens", 0),
                }
                content = data["choices"][0]["message"]["content"] or ""
                if not content.strip():
                    last_reason = "empty response"
                else:
                    parsed = _parse_verdict(content, unit_text)
                    if parsed is not None:
                        covered, spans, reasoning = parsed
                        if covered and not spans:
                            # no substantiation spans -> treat as not covered
                            covered, spans = 0, []
                            reasoning = (reasoning + "（无支撑片段，判定不成立）").strip()
                        return {
                            "covered": covered,
                            "spans": spans,
                            "reasoning": reasoning,
                            "status": "ok",
                            "reason": "",
                            "usage": usage,
                        }
                    last_reason = "unparseable"
            except Exception as exc:
                last_reason = f"{type(exc).__name__}: {str(exc)[:100]}"

            if attempt < self.max_retries:
                await asyncio.sleep(min(2 ** attempt, 30))

        return {
            "covered": 0,
            "spans": [],
            "reasoning": "",
            "status": "failed",
            "reason": last_reason,
            "usage": usage,
        }


# ---------------------------------------------------------------------------
# Per-question field recomputation (mirrors recall_evaluator._build_coverage_results)
# ---------------------------------------------------------------------------

def build_per_question(
    qid: str,
    evidence_list: List[dict],
    matches: List[List[int]],
    search: dict,
    answerable: int,
) -> dict:
    """Compute one per-question row from a match matrix (0-based result idxs)."""
    num_results = len(search.get("results", []))
    sources = [e.get("source", "?") for e in evidence_list]
    num_evidence = len(evidence_list)

    matches = list(matches) + [[] for _ in range(max(0, num_evidence - len(matches)))]
    matches = matches[:num_evidence]

    covered = [1 if m else 0 for m in matches]
    relevant = [0] * num_results
    for m in matches:
        for idx in m:
            if isinstance(idx, int) and 0 <= idx < num_results:
                relevant[idx] = 1

    covered_count = sum(covered)
    relevant_count = sum(relevant)
    total_tokens = sum(_count_tokens(r["content"]) for r in search.get("results", []))

    covered_at_5 = sum(1 for m in matches if any(isinstance(i, int) and 0 <= i < 5 for i in m))
    covered_at_20 = sum(1 for m in matches if any(isinstance(i, int) and 0 <= i < 20 for i in m))
    relevant_at_5 = sum(relevant[:5])
    relevant_at_20 = sum(relevant[:20])

    return {
        "question_id": qid,
        "num_evidence": num_evidence,
        "covered_count": covered_count,
        "coverage_rate": covered_count / num_evidence if num_evidence else 0.0,
        "covered": covered,
        "evidence_matches": matches,
        "sources": sources,
        "answerable": answerable,
        "num_results": num_results,
        "relevant_count": relevant_count,
        "precision": relevant_count / num_results if num_results else 0.0,
        "total_tokens": total_tokens,
        "avg_tokens_per_unit": total_tokens / num_results if num_results else 0.0,
        "covered_at_5": covered_at_5,
        "covered_at_20": covered_at_20,
        "recall_at_5": covered_at_5 / num_evidence if num_evidence else 0.0,
        "recall_at_20": covered_at_20 / num_evidence if num_evidence else 0.0,
        "relevant_at_5": relevant_at_5,
        "relevant_at_20": relevant_at_20,
        "precision_at_5": relevant_at_5 / min(5, num_results) if num_results else 0.0,
        "precision_at_20": relevant_at_20 / min(20, num_results) if num_results else 0.0,
    }


def summarize(rows: List[dict]) -> dict:
    """Aggregate top-level metrics from per-question rows (same formulas as
    recall_evaluator / rejudge_recall_failures)."""
    n_q = len(rows)
    summary: Dict[str, float] = {}
    if n_q == 0:
        return summary

    total_ev = sum(pq["num_evidence"] for pq in rows)
    covered_ev = sum(pq["covered_count"] for pq in rows)
    coverage_rate = sum(pq["coverage_rate"] for pq in rows) / n_q
    coverage_rate_micro = covered_ev / total_ev if total_ev else 0.0

    answerable_count = sum(1 for pq in rows if pq["answerable"] == 1)
    answerable_rate = answerable_count / n_q

    fully_covered_answerable = sum(
        1 for pq in rows
        if pq["answerable"] == 1 and pq["covered_count"] == pq["num_evidence"]
    )
    joint_rate = fully_covered_answerable / n_q

    total_results = sum(pq["num_results"] for pq in rows)
    relevant_results = sum(pq["relevant_count"] for pq in rows)
    precision = relevant_results / total_results if total_results else 0.0
    redundancy = 1.0 - precision
    recall = coverage_rate
    avg_results = total_results / n_q if n_q else 0.0
    avg_relevant = relevant_results / n_q if n_q else 0.0

    total_tokens = sum(pq["total_tokens"] for pq in rows)
    avg_tokens_per_unit = total_tokens / total_results if total_results else 0.0
    avg_tokens_per_question = total_tokens / n_q if n_q else 0.0

    summary.update({
        "recall": recall,
        "coverage_rate": coverage_rate,
        "coverage_rate_micro": coverage_rate_micro,
        "answerable_rate": answerable_rate,
        "covered_and_answerable_rate": joint_rate,
        "precision": precision,
        "redundancy": redundancy,
        "avg_results_per_question": avg_results,
        "avg_relevant_per_question": avg_relevant,
        "avg_tokens_per_memory_unit": avg_tokens_per_unit,
        "avg_tokens_per_question": avg_tokens_per_question,
        "num_questions": n_q,
        "total_evidence": total_ev,
        "covered_evidence": covered_ev,
    })
    summary["recall_at_5"] = sum(pq["recall_at_5"] for pq in rows) / n_q
    summary["recall_at_20"] = sum(pq["recall_at_20"] for pq in rows) / n_q
    summary["precision_at_5"] = sum(pq["precision_at_5"] for pq in rows) / n_q
    summary["precision_at_20"] = sum(pq["precision_at_20"] for pq in rows) / n_q
    return summary


# ---------------------------------------------------------------------------
# Token-precision aggregation
# ---------------------------------------------------------------------------

def annotate_token_precision(rows: List[dict], search_by_qid: Dict[str, dict],
                             verdicts_by_pair: Dict[str, dict]) -> List[dict]:
    """Add relevant_tokens / token_precision per row; returns updated rows."""
    for pq in rows:
        qid = pq["question_id"]
        results = search_by_qid.get(qid, {}).get("results", [])
        unit_tokens = [_count_tokens(r["content"]) for r in results]
        total_tokens = sum(unit_tokens)
        # unit -> merged ranges from all covered pairs of this question
        unit_ranges: Dict[int, List[Tuple[int, int]]] = {}
        for ev_idx, matches in enumerate(pq["evidence_matches"]):
            for res_idx in matches:
                key = f"{qid}|{ev_idx}|{res_idx}"
                v = verdicts_by_pair.get(key)
                if not v or v.get("status") != "ok" or not v.get("covered"):
                    continue
                unit_text = results[res_idx]["content"] if res_idx < len(results) else ""
                if not unit_text:
                    continue
                ranges = unit_ranges.setdefault(res_idx, [])
                for s, e in _span_ranges(unit_text, v.get("spans", [])):
                    ranges.append((s, e))
        relevant_tokens = 0
        for res_idx, ranges in unit_ranges.items():
            ranges.sort()
            merged = []
            for s, e in ranges:
                if merged and s <= merged[-1][1]:
                    if e > merged[-1][1]:
                        merged[-1] = (merged[-1][0], e)
                else:
                    merged.append((s, e))
            relevant_tokens += _ranges_tokens(results[res_idx]["content"], merged)
        pq["total_tokens"] = total_tokens  # same counting as _count_tokens on content
        pq["relevant_tokens"] = relevant_tokens
        pq["token_precision"] = relevant_tokens / total_tokens if total_tokens else 0.0
    return rows


def write_json_atomic(path: Path, obj: dict):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def pair_key(qid: str, ev_idx: int, res_idx: int) -> str:
    return f"{qid}|{ev_idx}|{res_idx}"


async def run(results_dir: str, concurrency: int, max_retries: int,
              dry_run: int, clean: bool, recheck_rejected: bool = False):
    base = Path(results_dir)
    ckpt_path = base / "recall_checkpoint_fixed.json"
    fixed_results_path = base / "recall_results_fixed.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")

    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    coarse = {e["question_id"]: e for e in ckpt.get("completed", [])}
    print(f"Loaded coarse checkpoint: {len(coarse)} questions")

    evidence_lookup = load_evidence_mapping(project_root / "datasets" / "lifebench_raw")
    conv_path = project_root / "datasets" / "lifebench_locomo_format" / \
                "lifebench_locomo_conversation_format_v2.0_3380QA.json"
    with open(conv_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    answer_map = {}
    for person in conv_data:
        for qa in person["qa"]:
            answer_map[qa["question_id"]] = qa.get("answer", "")

    # ---- collect pair list ------------------------------------------------
    pairs: List[Tuple[str, int, int]] = []  # (qid, ev_idx, res_idx)
    qids_needed = set(coarse.keys())
    for qid, entry in coarse.items():
        ev_list = evidence_lookup.get(qid, [])
        for ev_idx, matches in enumerate(entry.get("evidence_matches", [])):
            if ev_idx >= len(ev_list):
                continue
            for res_idx in matches:
                if isinstance(res_idx, int):
                    pairs.append((qid, ev_idx, res_idx))
    print(f"Positive pairs from coarse matches: {len(pairs)}")

    # ---- load search results (streaming, keep only needed qids) ----------
    search_path = base / "search_results.json"
    print(f"Streaming {search_path.name} ...")
    search_by_qid = load_needed_search(search_path, qids_needed)
    print(f"  kept {len(search_by_qid)} questions, "
          f"{sum(len(v['results']) for v in search_by_qid.values())} units")

    # sanity: unit index bounds
    bounded = []
    for qid, ev_idx, res_idx in pairs:
        n = len(search_by_qid.get(qid, {}).get("results", []))
        if res_idx < n:
            bounded.append((qid, ev_idx, res_idx))
    if len(bounded) != len(pairs):
        print(f"  [WARN] dropped {len(pairs) - len(bounded)} out-of-range pairs")
        pairs = bounded

    # ---- verdict checkpoint ----------------------------------------------
    verdict_path = base / VERIFY_CHECKPOINT
    completed: Dict[str, dict] = {}
    if (not clean) and verdict_path.exists():
        with open(verdict_path, "r", encoding="utf-8") as f:
            vck = json.load(f)
        if vck.get("format_version") == CHECKPOINT_FORMAT_VERSION:
            for entry in vck.get("completed", []):
                completed[pair_key(entry["question_id"], entry["ev_idx"], entry["res_idx"])] = entry
            print(f"Resumed {len(completed)} verdicts from checkpoint")
        else:
            print("  [WARN] verify-checkpoint format mismatch — starting fresh")

    if clean:
        for p in (verdict_path, base / VERIFY_DETAIL, base / VERIFY_RESULTS):
            if p.exists():
                p.unlink()
                print(f"  [clean] removed {p.name}")

    if recheck_rejected:
        # criterion v2 (faithful generalization counts as coverage): re-judge
        # only pairs previously rejected under v1 (covered=0, status=ok) plus
        # retry failed ones. Accepted pairs stay untouched.
        def _rejected(k):
            rec = completed.get(k)
            return rec is not None and rec.get("status") == "ok" and rec.get("covered") == 0
        pending = [(q, e, r) for (q, e, r) in pairs
                   if _rejected(pair_key(q, e, r))
                   or (completed.get(pair_key(q, e, r)) or {}).get("status") == "failed"]
    else:
        pending = [(q, e, r) for (q, e, r) in pairs
                   if pair_key(q, e, r) not in completed
                   or completed[pair_key(q, e, r)].get("status") == "failed"]
    if dry_run:
        pending = pending[:dry_run]

    print(f"\nPairs to verify: {len(pending)} pending, {len(completed)} cached"
          + ("   [DRY RUN]" if dry_run else ""))
    if pending and recheck_rejected:
        print("  mode: RECHECK-REJECTED — criterion v2 (faithful generalization "
              "counts as coverage); accepted pairs untouched")
    if pending:
        config = _load_config()
        print(f"Verifier: {config['model']} @ {config['base_url']}  "
              f"(max_tokens={config['max_tokens']}, concurrency={concurrency})")

        judge = PairVerifier(config, max_retries=max_retries)
        sem = asyncio.Semaphore(concurrency)
        pbar = tqdm(total=len(pending), desc="Verifying pairs")

        save_every = 50
        done_since_save = 0
        lock = asyncio.Lock()

        async def verify_one(qid: str, ev_idx: int, res_idx: int):
            nonlocal done_since_save
            async with sem:
                search = search_by_qid.get(qid, {})
                results = search.get("results", [])
                ev_list = evidence_lookup.get(qid, [])
                unit = results[res_idx]
                evidence = ev_list[ev_idx]
                verdict = await judge.verify(
                    question=search.get("query", ""),
                    reference_answer=answer_map.get(qid, ""),
                    evidence=evidence,
                    unit=unit,
                )
                rec = {
                    "question_id": qid,
                    "ev_idx": ev_idx,
                    "res_idx": res_idx,
                    "covered": verdict["covered"],
                    "spans": verdict["spans"],
                    "reasoning": verdict["reasoning"],
                    "status": verdict["status"],
                    "reason": verdict["reason"],
                }
                if verdict.get("usage"):
                    rec["prompt_tokens"] = verdict["usage"].get("prompt_tokens", 0)
                    rec["completion_tokens"] = verdict["usage"].get("completion_tokens", 0)
                completed[pair_key(qid, ev_idx, res_idx)] = rec
                pbar.update(1)

                async with lock:
                    done_since_save += 1
                    if done_since_save >= save_every:
                        save_completed(verdict_path, completed)
                        done_since_save = 0

        if pending:
            # process in bounded slices: a single gather over 20k+ tasks stalls
            # on Windows (event loop starved); slices keep fan-out manageable
            batch = 512
            for start in range(0, len(pending), batch):
                slice_ = pending[start:start + batch]
                await asyncio.gather(*[verify_one(q, e, r) for q, e, r in slice_])
        pbar.close()
        await judge.close()
        save_completed(verdict_path, completed)
        print(f"  verdicts now: {len(completed)} "
              f"(ok={sum(1 for v in completed.values() if v['status'] == 'ok')}, "
              f"failed={sum(1 for v in completed.values() if v['status'] == 'failed')})")

    verdicts_by_pair = completed

    # ---- build corrected match matrix ------------------------------------
    covered_votes = rejected = failed = 0
    corrected: Dict[str, List[List[int]]] = {}
    for qid, entry in coarse.items():
        orig = entry.get("evidence_matches", [])
        ev_list = evidence_lookup.get(qid, [])
        corrected_matches = []
        for ev_idx, matches in enumerate(orig):
            if ev_idx >= len(ev_list):
                corrected_matches.append([])
                continue
            kept = []
            for res_idx in matches:
                rec = verdicts_by_pair.get(pair_key(qid, ev_idx, res_idx))
                if rec is None:
                    # pair never targeted (shouldn't happen for non-empty matches)
                    kept.append(res_idx)
                    continue
                if rec.get("status") == "failed":
                    failed += 1
                    continue  # failed verdicts are dropped (strictness side)
                if rec.get("covered"):
                    covered_votes += 1
                    kept.append(res_idx)
                else:
                    rejected += 1
            corrected_matches.append(kept)
        corrected[qid] = corrected_matches

    # ---- per-question rows: before & after --------------------------------
    rows_before: List[dict] = []
    rows_after: List[dict] = []
    for qid in coarse:
        ev_list = evidence_lookup.get(qid, [])
        search = search_by_qid.get(qid, {"query": "", "results": []})
        ans = coarse[qid].get("answerable", 0)
        rows_before.append(build_per_question(qid, ev_list, coarse[qid].get("evidence_matches", []), search, ans))
        rows_after.append(build_per_question(qid, ev_list, corrected[qid], search, ans))

    rows_after = annotate_token_precision(rows_after, search_by_qid, verdicts_by_pair)

    stats_before = summarize(rows_before)
    stats_after = summarize(rows_after)

    # cross-check against the stored recall_results_fixed.json (baseline)
    if fixed_results_path.exists():
        with open(fixed_results_path, "r", encoding="utf-8") as f:
            fixed = json.load(f)
        diffs = []
        for key in ("coverage_rate", "coverage_rate_micro", "precision",
                    "answerable_rate", "avg_results_per_question",
                    "avg_tokens_per_memory_unit", "avg_tokens_per_question"):
            old = stats_before.get(key, 0.0)
            ref = fixed.get(key)
            if ref is not None and abs(old - ref) > 1e-9:
                diffs.append(f"{key}: recomputed={old:.6f} vs fixed={ref:.6f}")
        if diffs:
            print("\n[WARN] before-recompute differs from recall_results_fixed.json:")
            for d in diffs:
                print("  " + d)
        else:
            print("\n[OK] before metrics reproduce recall_results_fixed.json exactly")

    # ---- token-precision aggregates ---------------------------------------
    tp_micro_n = sum(pq["relevant_tokens"] for pq in rows_after)
    tp_micro_d = sum(pq["total_tokens"] for pq in rows_after)
    tp_micro = tp_micro_n / tp_micro_d if tp_micro_d else 0.0
    tp_macro = sum(pq["token_precision"] for pq in rows_after) / len(rows_after)

    print_stats(stats_before, stats_after, tp_micro, tp_macro,
                covered_votes, rejected, failed, len(pairs), dry_run)

    # ---- write outputs -----------------------------------------------------
    detail = {
        "questions": [],
        "token_precision": {"micro": tp_micro, "macro": tp_macro},
        "summary": {"pairs_total": len(pairs), "pairs_verified_covered": covered_votes,
                    "pairs_rejected": rejected, "pairs_failed": failed},
    }
    for pq in rows_after:
        qid = pq["question_id"]
        ev_list = evidence_lookup.get(qid, [])
        detail_q = {
            "question_id": qid,
            "num_evidence": pq["num_evidence"],
            "num_results": pq["num_results"],
            "total_tokens": pq["total_tokens"],
            "relevant_tokens": pq["relevant_tokens"],
            "token_precision": pq["token_precision"],
            "pairs": [],
        }
        results = search_by_qid.get(qid, {}).get("results", [])
        for ev_idx, matches in enumerate(corrected[qid]):
            for res_idx in matches:
                rec = verdicts_by_pair.get(pair_key(qid, ev_idx, res_idx), {})
                unit_tokens = _count_tokens(results[res_idx]["content"]) if res_idx < len(results) else 0
                spans = rec.get("spans", [])
                unit_text = results[res_idx]["content"] if res_idx < len(results) else ""
                pair_tokens = _ranges_tokens(unit_text, _span_ranges(unit_text, spans))
                detail_q["pairs"].append({
                    "ev_idx": ev_idx,
                    "evidence_source": ev_list[ev_idx].get("source") if ev_idx < len(ev_list) else None,
                    "res_idx": res_idx,
                    "covered": rec.get("covered", 0),
                    "spans": spans,
                    "reasoning": rec.get("reasoning", ""),
                    "pair_token_precision": pair_tokens / unit_tokens if unit_tokens else 0.0,
                    "pair_relevant_tokens": pair_tokens,
                    "unit_tokens": unit_tokens,
                })
        detail["questions"].append(detail_q)

    write_json_atomic(verdict_path, {"format_version": CHECKPOINT_FORMAT_VERSION,
                                     "completed": list(verdicts_by_pair.values())})
    write_json_atomic(base / VERIFY_DETAIL, detail)
    write_json_atomic(base / VERIFY_RESULTS, {
        "per_question": rows_after,
        "token_precision": {"micro": tp_micro, "macro": tp_macro},
        "rejudge_metadata": None,
        **stats_after,
    })
    print(f"\nVerification detail -> {base / VERIFY_DETAIL}")
    print(f"Corrected results   -> {base / VERIFY_RESULTS}")

    # ---- "new table" row ----------------------------------------------------
    print("\n--- New table row (mindmemos style) ---")
    print(f"  Recall (coverage macro): {stats_after['coverage_rate']*100:6.2f}%   "
          f"(was {stats_before['coverage_rate']*100:.2f}%)")
    print(f"  Recall (coverage micro): {stats_after['coverage_rate_micro']*100:6.2f}%   "
          f"(was {stats_before['coverage_rate_micro']*100:.2f}%)")
    print(f"  Token Precision (micro): {tp_micro*100:6.2f}%")
    print(f"  Token Precision (macro): {tp_macro*100:6.2f}%")
    print(f"  Avg results/question:    {stats_after['avg_results_per_question']:6.2f}")
    print(f"  Avg tokens/unit:         {stats_after['avg_tokens_per_memory_unit']:6.1f}")


def save_completed(verdict_path: Path, completed: Dict[str, dict]):
    write_json_atomic(verdict_path, {"format_version": CHECKPOINT_FORMAT_VERSION,
                                     "completed": list(completed.values())})


def print_stats(before: dict, after: dict, tp_micro: float, tp_macro: float,
                covered_votes: int, rejected: int, failed: int,
                total_pairs: int, dry_run: bool):
    print()
    print("=" * 68)
    print("Verified pair results        (before -> after)" + ("   [DRY RUN]" if dry_run else ""))
    print("=" * 68)
    rows = [
        ("Recall (= macro coverage)", "coverage_rate"),
        ("Coverage (micro)", "coverage_rate_micro"),
        ("Precision (result purity)", "precision"),
        ("Answerable rate", "answerable_rate"),
        ("Recall@5", "recall_at_5"),
        ("Recall@20", "recall_at_20"),
    ]
    for label, key in rows:
        b, a = before.get(key, 0.0), after.get(key, 0.0)
        print(f"  {label:28s}: {b*100:6.2f}% -> {a*100:6.2f}%   (Δ {(a-b)*100:+.2f}pp)")
    print(f"  {'Token Precision micro':28s}: {tp_micro*100:6.2f}%")
    print(f"  {'Token Precision macro':28s}: {tp_macro*100:6.2f}%")
    print()
    print(f"  pairs verified: {total_pairs}")
    print(f"    still covered : {covered_votes}   rejected: {rejected}   failed: {failed}")
    if dry_run:
        print("  NOTE: DRY RUN — only first pairs were really judged; "
              "all numbers are partial!")


def main():
    parser = argparse.ArgumentParser(
        description="Verify coarse (evidence->result) matches one-by-one with "
                    "DeepSeek and annotate Token Precision"
    )
    parser.add_argument("--results-dir", default="results/lifebench-mindmemos-schema",
                        help="Directory containing recall_checkpoint_fixed.json "
                             "and search_results.json")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", type=int, default=0,
                        help="Only verify first N pending pairs (real calls)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing verdict checkpoint and start fresh")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing verify outputs first")
    parser.add_argument("--recheck-rejected", action="store_true",
                        help="Re-judge only pairs rejected in a previous run "
                             "(criterion v2: faithful generalization counts)")
    args = parser.parse_args()
    if args.no_resume:
        args.clean = True
    asyncio.run(run(args.results_dir, args.concurrency, args.max_retries,
                    args.dry_run, args.clean, args.recheck_rejected))


if __name__ == "__main__":
    main()