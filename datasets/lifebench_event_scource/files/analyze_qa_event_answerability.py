#!/usr/bin/env python3
"""分析 LifeBench 三人的 QA：判断每个问题「仅凭事件记录（daily event）」能否回答。

背景
----
事件粒度数据集的构建方式，是用 daily_event 事件记录直接替换掉 conversation 里按天组织的
手机数据（lifebench_raw）。已确认：基于手机数据（evidence）回答每个问题都是可答的。但换成
事件记录后，部分 QA 会因关键信息遗漏而回答不出来。

本脚本对 fenghaoran / sunyuwei / yuxiaowei 三人的 lifebench_raw QA 逐条做如下分析：

    1. 由 question_id 从 question_id_to_evidence_mapping.json 取出该题的证据项
       （每项即一条手机数据 raw_data，含 daily_event_id）；
    2. 用每条手机数据的 daily_event_id 去 daily_event_{pinyin}.json 里找到对应的事件记录；
    3. 调用 LLM 判断：仅凭这些事件记录能否回答该问题；
    4. 过滤出「仅凭事件记录可回答」的问题，连同逐题明细、统计摘要一并保存为记录文件。

用法（在任意目录执行均可，路径基于脚本所在位置解析）：

    # 先做 dry-run，验证数据链路（不调用 LLM）
    python analyze_qa_event_answerability.py --dry-run

    # 真正调用 LLM 分析（默认三人，事件数据取 daily_event_clean）
    python analyze_qa_event_answerability.py

    # 用原始（未清洗）事件数据 daily_event
    python analyze_qa_event_answerability.py --event-source daily_event

LLM 配置：优先使用 RECALL_JUDGE_* 环境变量，未设置时回退到 LLM_*（与 src/utils/recall_evaluator.py 一致）。
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent.parent  # LifeBench_eval 根目录

RAW_MAPPING = BASE / "datasets" / "lifebench_raw" / "question_id_to_evidence_mapping.json"
CONV_PATH = BASE / "datasets" / "lifebench_locomo_format" / "lifebench_locomo_conversation_format_v2.0_3380QA.json"

DEFAULT_OUT = HERE / "qa_event_answerability_record.json"
DEFAULT_CHECKPOINT = HERE / ".qa_event_answerability_checkpoint.json"

# sample_id -> 拼音（文件名中缀）
PERSONS = {
    "孙雨薇": "sunyuwei",
    "于晓薇": "yuxiaowei",
    "冯浩然": "fenghaoran",
}

# 事件数据文件名模板
EVENT_FILE_TPL = "daily_event_{pinyin}.json"
EVENT_CLEAN_FILE_TPL = "daily_event_clean_{pinyin}.json"

CHECKPOINT_FORMAT_VERSION = 2

# 单条事件渲染后的最大字符数（防止超长描述撑爆 prompt）
MAX_CHARS_PER_ITEM = 2500


def _clip(text: str, limit: int = MAX_CHARS_PER_ITEM) -> str:
    """超长文本截断并加注记。"""
    if not text:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "……（已截断）"


def _fmt_event(ev: dict, index: int) -> str:
    """把一条 daily_event 渲染成给 LLM 看的中文文本。"""
    name = ev.get("name", "")
    typ = ev.get("type", "")
    dates = ev.get("date", [])
    desc = ev.get("description", "")
    participants = ev.get("participant", [])
    location = ev.get("location", "")

    parts = [f"事件{index}. {name}"]
    if typ:
        parts.append(f"   类型: {typ}")
    if dates:
        parts.append(f"   时间: {'、'.join(dates)}")
    if participants and isinstance(participants, list):
        ppl = "、".join(
            f"{p.get('name', '')}({p.get('relation', '')})"
            for p in participants if isinstance(p, dict)
        )
        if ppl:
            parts.append(f"   参与者: {ppl}")
    if location:
        parts.append(f"   地点: {location}")
    if desc:
        parts.append(f"   描述: {_clip(desc)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM Judge prompt（仅判事件记录可回答性）
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "你是 LifeBench 记忆数据集的质量评估员。给定一个问题及其参考答案，以及对应的事件记录（daily event），"
    "判断仅凭这些事件记录能否回答该问题。\n\n"
    "判定「可回答」(1) 的标准：\n"
    "- 以参考答案为锚点，提炼出回答该问题所需的「关键事实点」（事件、人物、时间、关键数据等）。\n"
    "- 只要事件记录里包含了这些关键事实点（可单条、可多条拼合；语义等价、改写、合理归纳都算），"
    "即可判为可回答。不要因为事件记录里有无关内容、表述冗长或需要拼合就降低判定。\n"
    "- 只有当关键事实点缺失、事件记录只有话题沾边或完全无关的内容时，才判为「不可回答」(0)。\n"
    "- 不要要求事件记录本身已组织成完整答案，只要包含回答所需的关键信息片段即可。\n\n"
    "只输出 JSON，不要额外文字。"
)

JUDGE_USER_TEMPLATE = """## 问题
{question}

## 参考答案
{reference_answer}

## 事件记录（共{event_count}条）
{event_text}

## 任务
判断仅凭上述事件记录能否回答该问题（0 或 1）。

## 输出格式
{{"answerable": 0, "reasoning": "一句话说明"}}

只输出 JSON。"""


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evidence_mapping() -> Dict[str, List[dict]]:
    if not RAW_MAPPING.exists():
        raise FileNotFoundError(f"未找到证据映射文件：{RAW_MAPPING}")
    print(f"加载证据映射：{RAW_MAPPING}")
    return load_json(RAW_MAPPING)


def load_qa_by_person() -> Dict[str, List[dict]]:
    """按 pinyin 加载 lifebench_raw_{pinyin}.json 的 qa 列表。"""
    qa_by_person: Dict[str, List[dict]] = {}
    for sample_id, pinyin in PERSONS.items():
        path = HERE / pinyin / f"lifebench_raw_{pinyin}.json"
        raw = load_json(path)
        sample = raw[0] if isinstance(raw, list) else raw
        qa_by_person[pinyin] = sample.get("qa", [])
        print(f"  {sample_id:6s} ({pinyin}) qa={len(qa_by_person[pinyin])}")
    return qa_by_person


def load_event_index(pinyin: str, event_source: str) -> Dict[str, dict]:
    """加载某人的 daily_event，返回 event_id -> event 的映射。"""
    if event_source == "daily_event_clean":
        tpl = EVENT_CLEAN_FILE_TPL
    elif event_source == "daily_event":
        tpl = EVENT_FILE_TPL
    else:
        raise SystemExit(f"未知 --event-source={event_source!r}，可选：daily_event_clean / daily_event")
    path = HERE / pinyin / tpl.format(pinyin=pinyin)
    events = load_json(path)
    index = {e["event_id"]: e for e in events}
    print(f"  {pinyin} [{event_source}] 事件数={len(events)}，event_id 索引={len(index)}")
    return index


def build_answer_map() -> Dict[str, str]:
    """question_id -> 参考答案。"""
    if not CONV_PATH.exists():
        return {}
    answer_map: Dict[str, str] = {}
    for person in load_json(CONV_PATH):
        for qa in person.get("qa", []):
            answer_map[qa["question_id"]] = qa.get("answer", "")
    return answer_map


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class AnswerabilityJudge:
    """调用 LLM 判断仅凭事件记录能否回答该问题。"""

    def __init__(self, config: dict):
        self.model = config.get("model", "deepseek-v4-flash")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.deepseek.com")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.0)
        self.max_retries = config.get("max_retries", 3)
        self._session: Optional[aiohttp.ClientSession] = None
        # 全局限速（N 次/秒），避免并发突发触发服务端限流（401/429）
        # 注意：AsyncLimiter 的 time_period 默认是 60 秒，须显式传 1.0 才是「N 次/秒」。
        self._limiter = AsyncLimiter(config.get("rate_limit", 3.0), 1.0)
        # 401/429 冷却：一旦某次请求命中限流，全局暂停一段时间再继续，避免持续触发
        self._cooldown_until = 0.0
        self._cooldown_lock = asyncio.Lock()

    async def _pause_until_allowed(self):
        """若当前处于限流冷却期，等待其结束。"""
        while True:
            async with self._cooldown_lock:
                remaining = self._cooldown_until - time.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 5.0))

    async def _note_rate_limited(self, seconds: float = 60.0):
        """记录一次限流，触发全局冷却。"""
        async with self._cooldown_lock:
            self._cooldown_until = max(self._cooldown_until, time.time() + seconds)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            conn = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=360)
            self._session = aiohttp.ClientSession(connector=conn, timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def judge(
        self,
        question: str,
        reference_answer: str,
        events: List[dict],
    ) -> Optional[Tuple[int, str]]:
        """返回 (answerable, reasoning) 或 None。"""
        event_text = "\n\n".join(_fmt_event(ev, i) for i, ev in enumerate(events))

        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference_answer=reference_answer,
            event_count=len(events),
            event_text=event_text or "（无）",
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
            backoff = 2 * attempt
            # 若处于限流冷却期，先等冷却结束，避免持续触发 401/429
            await self._pause_until_allowed()
            try:
                session = await self._get_session()
                async with self._limiter:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        # 401/429 多为限流/瞬时鉴权问题，触发全局冷却并拉长退避后重试
                        if resp.status in (401, 429):
                            await self._note_rate_limited(60.0)
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=resp.status
                            )
                        resp.raise_for_status()
                        data = await resp.json()

                content = data["choices"][0]["message"]["content"]
                parsed = self._parse_response(content)
                if parsed is not None:
                    return parsed

            except aiohttp.ClientResponseError as e:
                if e.status in (401, 429):
                    backoff = min(10 * attempt, 60)
                if attempt < self.max_retries:
                    print(f"  [RETRY {attempt}/{self.max_retries}] judge: {type(e).__name__} {e.status}: {str(e)[:100]}")
                    await asyncio.sleep(backoff)
                    continue
                print(f"  [ERROR] judge failed after {self.max_retries} attempts: {type(e).__name__} {e.status}: {e}")
            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  [RETRY {attempt}/{self.max_retries}] judge: {type(e).__name__}: {str(e)[:120]}")
                    await asyncio.sleep(backoff)
                    continue
                print(f"  [ERROR] judge failed after {self.max_retries} attempts: {type(e).__name__}: {e}")

        return None

    def _parse_response(self, content: str) -> Optional[Tuple[int, str]]:
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

        answerable = self._coerce_bool(obj.get("answerable"))
        reasoning = str(obj.get("reasoning", ""))
        return answerable, reasoning

    @staticmethod
    def _coerce_bool(value) -> int:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if int(value) == 1 else 0
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("1", "true", "yes", "是"):
                return 1
            return 0
        return 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def collect_items(
    evidence_lookup: Dict[str, List[dict]],
    qa_by_person: Dict[str, List[dict]],
    event_index_by_person: Dict[str, Dict[str, dict]],
    answer_map: Dict[str, str],
) -> List[dict]:
    """把每条 QA 组织成待判定条目：含证据（手机数据）、对应事件、参考答案等。"""
    items = []
    for pinyin, qa_list in qa_by_person.items():
        ev_idx = event_index_by_person[pinyin]
        for qa in qa_list:
            qid = qa.get("question_id", "")
            evidence = evidence_lookup.get(qid, [])

            # 由每条证据的 daily_event_id 找到对应事件（去重，保持出现顺序）
            events: List[dict] = []
            seen_event_ids = set()
            for ev in evidence:
                raw = ev.get("raw_data", {}) or {}
                deid = str(raw.get("daily_event_id", "") or "")
                if deid and deid not in seen_event_ids and deid in ev_idx:
                    events.append(ev_idx[deid])
                    seen_event_ids.add(deid)

            items.append({
                "pinyin": pinyin,
                "question_id": qid,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", answer_map.get(qid, "")),
                "ask_time": qa.get("ask_time", ""),
                "category": qa.get("category", []),
                "question_type": qa.get("question_type", []),
                "score_points": qa.get("score_points", []),
                "evidence": evidence,
                "daily_events": events,
                "has_evidence": bool(evidence),
            })
    return items


def _build_record(items: List[dict], completed: Dict[str, dict], event_source: str, model: str) -> dict:
    """由已完成判定组装最终记录文件。"""
    per_question = []
    stats = {
        "total": len(items),
        "answerable_by_event": 0,
        "not_answerable_by_event": 0,
        "unanswerable": 0,
        "no_judgment": 0,
    }
    by_person = defaultdict(lambda: {
        "total": 0, "answerable_by_event": 0, "not_answerable_by_event": 0,
        "unanswerable": 0, "no_judgment": 0,
    })

    filtered_ids = []

    for item in items:
        qid = item["question_id"]
        pinyin = item["pinyin"]
        entry = completed.get(qid)
        by_person[pinyin]["total"] += 1

        if not item.get("has_evidence"):
            # 无证据的 Unanswerable 题（设计上即无法回答），不参与事件可回答性判定
            answerable, reasoning = None, ""
            stats["unanswerable"] += 1
            by_person[pinyin]["unanswerable"] += 1
            verdict = "unanswerable"
        elif entry is None:
            answerable, reasoning = None, ""
            stats["no_judgment"] += 1
            by_person[pinyin]["no_judgment"] += 1
            verdict = "no_judgment"
        else:
            answerable = entry.get("answerable")
            reasoning = entry.get("reasoning", "")
            if answerable == 1:
                verdict = "answerable_by_event"
                stats["answerable_by_event"] += 1
                by_person[pinyin]["answerable_by_event"] += 1
                filtered_ids.append(qid)
            else:
                verdict = "not_answerable_by_event"
                stats["not_answerable_by_event"] += 1
                by_person[pinyin]["not_answerable_by_event"] += 1

        per_question.append({
            "question_id": qid,
            "person": pinyin,
            "verdict": verdict,
            "answerable_by_event": answerable,
            "reasoning": reasoning,
            "question": item["question"],
            "answer": item["answer"],
            "ask_time": item["ask_time"],
            "category": item["category"],
            "question_type": item["question_type"],
            "evidence": item["evidence"],
            "daily_events": item["daily_events"],
        })

    return {
        "meta": {
            "event_source": event_source,
            "persons": list(PERSONS.keys()),
            "model": model,
            "total_questions": len(items),
            "answerable_by_event": stats["answerable_by_event"],
            "note": "基于手机数据(evidence)可回答已作为已知前提，仅额外判定基于事件记录是否可回答。",
        },
        "summary": {
            "overall": stats,
            "by_person": dict(by_person),
        },
        "filtered_question_ids": filtered_ids,
        "per_question": per_question,
    }


async def run(
    event_source: str,
    concurrency: int,
    llm_config: dict,
    resume: bool,
    output_path: Path,
    checkpoint_path: Path,
    clean: bool,
    dry_run: bool,
) -> None:
    evidence_lookup = load_evidence_mapping()
    qa_by_person = load_qa_by_person()
    answer_map = build_answer_map()
    event_index_by_person = {
        pinyin: load_event_index(pinyin, event_source) for pinyin in PERSONS.values()
    }

    items = collect_items(evidence_lookup, qa_by_person, event_index_by_person, answer_map)

    # 数据链路统计
    n_ev_total = sum(len(it["evidence"]) for it in items)
    n_event_total = sum(len(it["daily_events"]) for it in items)
    n_no_evidence = sum(1 for it in items if not it["evidence"])
    n_no_event = sum(1 for it in items if not it["daily_events"])
    print(f"\n待分析问题数：{len(items)}")
    print(f"  证据（手机数据）总数：{n_ev_total}")
    print(f"  对应事件总数：{n_event_total}")
    print(f"  无证据问题（Unanswerable）：{n_no_evidence}")
    print(f"  有证据但无对应事件：{n_no_event - n_no_evidence}")

    if dry_run:
        # 只打印样例，不调用 LLM
        print("\n[DRY RUN] 不调用 LLM。前 3 条样例（问题 + 证据数 + 事件数）：")
        for it in items[:3]:
            print(f"  - {it['question_id']} ({it['pinyin']}) 证据={len(it['evidence'])} 事件={len(it['daily_events'])}")
            print(f"      Q: {it['question'][:80]}...")
        n_judge = sum(1 for it in items if it.get("has_evidence"))
        print(f"\n预估 LLM 调用次数：{n_judge}（另有 {len(items) - n_judge} 条无证据的 Unanswerable 题跳过）")
        return

    if clean:
        for p in (checkpoint_path, output_path):
            if p.exists():
                p.unlink()
                print(f"  [clean] 已删除 {p.name}")
        resume = False

    # 载入 checkpoint
    completed: Dict[str, dict] = {}
    if resume and checkpoint_path.exists():
        ckpt = load_json(checkpoint_path)
        if isinstance(ckpt, dict) and ckpt.get("format_version") == CHECKPOINT_FORMAT_VERSION:
            for entry in ckpt.get("completed", []):
                completed[entry["question_id"]] = {
                    "answerable": entry.get("answerable"),
                    "reasoning": entry.get("reasoning", ""),
                }
            print(f"  从 checkpoint 恢复 {len(completed)} 条已完成判定")
        else:
            print("  [WARN] checkpoint 格式不匹配，从头开始")

    pending = [it for it in items if it["question_id"] not in completed and it.get("has_evidence")]
    print(f"\n开始判定：{len(pending)} 条待判定，{len(completed)} 条已缓存")
    print(f"  （{sum(1 for it in items if not it.get('has_evidence'))} 条无证据的 Unanswerable 题跳过判定）")
    print(f"  LLM: {llm_config['model']} @ {llm_config['base_url']}")
    print(f"  并发：{concurrency}")

    judge = AnswerabilityJudge(llm_config)
    semaphore = asyncio.Semaphore(concurrency)
    save_every = max(1, concurrency * 5)
    done_since_save = 0
    lock = asyncio.Lock()
    pbar = tqdm(total=len(pending), desc="Judging")

    async def judge_one(item: dict):
        nonlocal done_since_save
        async with semaphore:
            qid = item["question_id"]
            result = await judge.judge(
                question=item["question"],
                reference_answer=item["answer"],
                events=item["daily_events"],
            )
            # 只有成功才写入 completed/checkpoint，失败项留待 resume 重试
            if result is not None:
                answerable, reasoning = result
                completed[qid] = {"answerable": answerable, "reasoning": reasoning}
            pbar.update(1)

            async with lock:
                nonlocal done_since_save
                done_since_save += 1
                if done_since_save >= save_every:
                    _save_checkpoint(checkpoint_path, completed)
                    done_since_save = 0

    if pending:
        tasks = [judge_one(it) for it in pending]
        await asyncio.gather(*tasks)

    pbar.close()
    await judge.close()
    _save_checkpoint(checkpoint_path, completed)

    record = _build_record(items, completed, event_source, llm_config["model"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # 汇总打印
    s = record["summary"]["overall"]
    total = s["total"] or 1
    print("\n" + "=" * 60)
    print("QA 事件记录可回答性分析结果（手机数据可回答为已知前提）")
    print("=" * 60)
    print(f"  问题总数：{s['total']}")
    print(f"  仅凭事件记录可回答：{s['answerable_by_event']}  ({s['answerable_by_event'] / total * 100:.1f}%)")
    print(f"  事件记录不可回答（信息遗漏）：{s['not_answerable_by_event']}  "
          f"({s['not_answerable_by_event'] / total * 100:.1f}%)")
    print(f"  Unanswerable（无证据）：{s['unanswerable']}")
    print(f"  判定失败：{s['no_judgment']}")
    print("\n  按人物：")
    for pinyin, st in record["summary"]["by_person"].items():
        tot = st["total"] or 1
        print(f"    {pinyin:12s} 共{st['total']:>3}  事件可答={st['answerable_by_event']:>3}  "
              f"事件不可答={st['not_answerable_by_event']:>3}  不可答={st['unanswerable']:>3}")
    print(f"\n记录文件已写入：{output_path}")


def _save_checkpoint(path: Path, completed: Dict[str, dict]):
    tmp = str(path) + ".tmp"
    entries = [
        {
            "question_id": qid,
            "answerable": v.get("answerable"),
            "reasoning": v.get("reasoning", ""),
        }
        for qid, v in completed.items()
    ]
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"format_version": CHECKPOINT_FORMAT_VERSION, "completed": entries},
                  f, ensure_ascii=False)
    os.replace(tmp, str(path))


def main():
    parser = argparse.ArgumentParser(description="分析三人 QA 的事件记录可回答性")
    parser.add_argument("--event-source", default="daily_event_clean",
                        choices=["daily_event_clean", "daily_event"],
                        help="事件数据来源（默认 daily_event_clean）")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="最大并发 LLM 调用数（默认 8）")
    parser.add_argument("--rate-limit", type=float, default=3.0,
                        help="全局每秒最大请求数（默认 3.0，避免触发服务端限流）")
    parser.add_argument("--output", default=str(DEFAULT_OUT),
                        help="记录文件输出路径")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="断点续传 checkpoint 路径")
    parser.add_argument("--no-resume", action="store_true", help="忽略 checkpoint 从头开始")
    parser.add_argument("--clean", action="store_true", help="先删除已有 checkpoint 与记录再运行")
    parser.add_argument("--dry-run", action="store_true", help="只验证数据链路，不调用 LLM")
    parser.add_argument("--env-file", default=None, help=".env 路径（默认自动从项目根目录读取）")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent.parent
    env_path = args.env_file or str(base / ".env")
    load_dotenv(env_path)

    llm_config = {
        "model": os.getenv("RECALL_JUDGE_MODEL") or os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        "api_key": os.getenv("RECALL_JUDGE_API_KEY") or os.getenv("LLM_API_KEY", ""),
        "base_url": os.getenv("RECALL_JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "max_tokens": int(os.getenv("RECALL_JUDGE_MAX_TOKENS") or os.getenv("LLM_MAX_TOKENS", "4096")),
        "temperature": float(os.getenv("RECALL_JUDGE_TEMPERATURE") or os.getenv("LLM_TEMPERATURE", "0.0")),
        "rate_limit": args.rate_limit,
    }

    print("QA 事件记录可回答性分析")
    print(f"  事件数据来源：{args.event_source}")
    print(f"  LLM：{llm_config['model']} @ {llm_config['base_url']}")
    print(f"  输出：{args.output}")
    print(f"  断点：{args.checkpoint}")
    if args.dry_run:
        print("  DRY RUN —— 不调用 LLM")
    print()

    asyncio.run(run(
        event_source=args.event_source,
        concurrency=args.concurrency,
        llm_config=llm_config,
        resume=not args.no_resume,
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        clean=args.clean,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
