"""
Narrative-only answerability baseline (upper bound) for LifeBench.

Feeds the *full conversation narrative* — exactly the input data the memory
systems ingest (speaker + text + blip_caption per message, all sessions) —
straight to the LLM, with no memory system and no retrieval. Generates an
answer, then evaluates with the same LLM judge that ``cli.py`` uses.

Purpose: separate "the system failed to store/retrieve the fact" from "the
question is not answerable from the ingested narrative at all". If the LLM
cannot answer a question from the full narrative, no memory system can be
expected to, and that question should be excluded (or separately reported)
when scoring recall / answerability.

This is the narrative analogue of ``direct_evidence_baseline.py`` (which feeds
the golden structured evidence instead of the narrative).

Usage:
    python -m src.utils.narrative_answerability_baseline                 # answer + eval all
    python -m src.utils.narrative_answerability_baseline --limit 50      # smoke test
    python -m src.utils.narrative_answerability_baseline --skip-answer   # only eval existing answers
    python -m src.utils.narrative_answerability_baseline --skip-eval     # only generate answers
"""
import argparse
import asyncio
import io
import json
import os
import re
import sys
from pathlib import Path

# Fix for Windows ASCII encoding issue with Chinese characters
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["PYTHONASYNCIODEBUG"] = "0"

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.evaluators.registry import create_evaluator
from src.models.answer import AnswerResult
from src.utils.config import load_yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = (
    project_root
    / "datasets"
    / "lifebench_locomo_format"
    / "lifebench_locomo_conversation_format_v2.0_3380QA.json"
)
DATASET_CONFIG_PATH = project_root / "config" / "datasets" / "lifebench.yaml"
OUTPUT_DIR = project_root / "results" / "lifebench-narrative_baseline"

# ---------------------------------------------------------------------------
# Answer-generation LLM config (matches the memory system's answer model)
# ---------------------------------------------------------------------------
ANSWER_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
ANSWER_API_KEY = os.getenv("LLM_API_KEY", "")
ANSWER_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
ANSWER_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "32768"))

ANSWER_SYSTEM_PROMPT = (
    "你是一个基于个人数字足迹记录（日历、备忘录、AI助手聊天、通话、短信、照片等）"
    "回答问题的助手。请严格依据提供的对话记录作答，不要编造记录中没有的信息。"
)

ANSWER_USER_TEMPLATE = """请根据下面的对话记录回答问题。

## 对话记录
{narrative}

## 问题
{question}

## 作答要求
1. 通读全部对话，综合多处信息作答：答案往往分散在多条消息里，不要只看开头。
2. 消息里的时间、地点、人物、照片描述等都是有效信息，不要忽略。
3. 模糊称谓按对话指代：对话里常用「老公」「婆婆」「闺蜜」等称谓代替姓名，请结合上下文推断其具体指代。不要因为用了模糊称谓就认为信息缺失。
4. 记录之间存在矛盾时，判断哪条更直接可信（本人陈述、带日期的消息通常更可信），给出裁决后的值，而不是拒绝回答。
5. 问题需要推理时（原因、可行性、规律总结、日期推算），依据记录做合理推断，不要轻易放弃。
6. 只有当记录里确实没有任何能回答该问题的信息时，才回答「无法确定」；只要有线索，就给出你的最佳答案。
7. 直接、简洁地给出答案，不要罗列推理过程。"""


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load raw person records (sample_id + conversation + qa)."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_narrative(person: dict) -> str:
    """Flatten a person's conversation sessions into chronological narrative text.

    Mirrors what the runner ingests: speaker + text (+ blip_caption) per message,
    sessions sorted by session number, messages kept in order.
    """
    conversation = person.get("conversation", {})
    if not isinstance(conversation, dict):
        return ""

    session_keys = sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", k)),
        key=lambda x: int(re.search(r"session_(\d+)", x).group(1)),
    )

    lines = []
    for key in session_keys:
        msgs = conversation.get(key)
        if not isinstance(msgs, list):
            continue
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            speaker = msg.get("speaker", "") or ""
            text = msg.get("text", "") or ""
            caption = msg.get("blip_caption", "") or ""
            if caption:
                text = f"{text}（配图描述：{caption}）" if text else f"（配图描述：{caption}）"
            if not text:
                continue
            lines.append(f"{speaker}: {text}" if speaker else text)

    return "\n".join(lines)


def load_qa_records(dataset_path: Path) -> list[dict]:
    """Load all QA records with their full narrative attached."""
    data = load_dataset(dataset_path)
    records = []
    for person in data:
        sample_id = person.get("sample_id", "")
        narrative = build_narrative(person)
        for qa in person.get("qa", []):
            records.append(
                {
                    "question_id": qa.get("question_id", ""),
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "category": str(qa.get("category", "")),
                    "question_type": qa.get("question_type", []),
                    "ask_time": qa.get("ask_time", ""),
                    "score_points": qa.get("score_points", []),
                    "person_id": sample_id,
                    "conversation_id": sample_id,
                    "narrative": narrative,
                }
            )
    return records


async def _answer_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    qa: dict,
    temperature: float,
) -> str:
    """Generate an answer for a single QA given its full narrative."""
    user_prompt = ANSWER_USER_TEMPLATE.format(
        narrative=qa["narrative"],
        question=qa["question"],
    )
    payload = {
        "model": ANSWER_MODEL,
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": ANSWER_MAX_TOKENS,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ANSWER_API_KEY}",
    }
    url = f"{ANSWER_BASE_URL}/chat/completions"

    async with semaphore:
        for attempt in range(1, 4):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    resp.raise_for_status()
                    data = await resp.json()

                if isinstance(data, dict) and "choices" in data:
                    return data["choices"][0]["message"]["content"] or ""
                return str(data)
            except Exception as exc:
                if attempt == 3:
                    return f"Error generating answer: {str(exc)[:100]}"
                await asyncio.sleep(2 ** attempt)


def _answer_result_to_dict(ar: AnswerResult) -> dict:
    return {
        "question_id": ar.question_id,
        "question": ar.question,
        "answer": ar.answer,
        "golden_answer": ar.golden_answer,
        "category": ar.category,
        "conversation_id": ar.conversation_id,
        "formatted_context": ar.formatted_context,
        "metadata": ar.metadata,
    }


def _dict_to_answer_result(d: dict) -> AnswerResult:
    return AnswerResult(
        question_id=d["question_id"],
        question=d["question"],
        answer=d["answer"],
        golden_answer=d["golden_answer"],
        category=d.get("category"),
        conversation_id=d.get("conversation_id", ""),
        formatted_context=d.get("formatted_context", ""),
        metadata=d.get("metadata", {}),
    )


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


async def run_answer(
    qa_records: list[dict],
    output_dir: Path,
    concurrency: int,
    temperature: float,
) -> None:
    """Generate answers for all QAs (resume-safe), saving incrementally."""
    answer_path = output_dir / "answer_results.json"
    existing: dict[str, dict] = {}
    if answer_path.exists():
        with open(answer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            existing = {item["question_id"]: item for item in data}

    remaining = [qa for qa in qa_records if qa["question_id"] not in existing]
    print(f"\n[ANSWER] total={len(qa_records)} already_answered={len(existing)} "
          f"remaining={len(remaining)}")

    if not remaining:
        print("  All answers already present, skipping answer stage.")
        return

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(remaining), desc="ANSWER")

        async def run_one(qa: dict) -> AnswerResult:
            answer = await _answer_one(session, semaphore, qa, temperature)
            pbar.update(1)
            return AnswerResult(
                question_id=qa["question_id"],
                question=qa["question"],
                answer=answer,
                golden_answer=qa["answer"],
                category=qa["category"],
                conversation_id=qa["conversation_id"],
                formatted_context=qa["narrative"],
                metadata={
                    "ask_time": qa["ask_time"],
                    "question_type": qa["question_type"],
                    "score_points": qa["score_points"],
                    "person_id": qa["person_id"],
                    "conversation_id": qa["conversation_id"],
                },
            )

        for coro in asyncio.as_completed([run_one(qa) for qa in remaining]):
            ar = await coro
            existing[ar.question_id] = _answer_result_to_dict(ar)
            _save_json(list(existing.values()), answer_path)

        pbar.close()


async def run_eval(output_dir: Path) -> None:
    """Evaluate answer_results.json using the cli.py LLM judge."""
    answer_path = output_dir / "answer_results.json"
    if not answer_path.exists():
        print(f"answer_results.json not found: {answer_path}")
        return

    with open(answer_path, "r", encoding="utf-8") as f:
        answer_results = [_dict_to_answer_result(d) for d in json.load(f)]

    dataset_config = load_yaml(str(DATASET_CONFIG_PATH))
    evaluator = create_evaluator(
        dataset_config["evaluation"]["type"], dataset_config["evaluation"]
    )
    print(f"\n[EVALUATE] {len(answer_results)} answers, evaluator={evaluator.get_name()}")

    eval_result = await evaluator.evaluate(answer_results)

    result = {
        "total_questions": eval_result.total_questions,
        "correct": eval_result.correct,
        "accuracy": eval_result.accuracy,
        "weighted_score": eval_result.weighted_score,
        "detailed_results": eval_result.detailed_results,
        "question_type_stats": {
            name: {
                "name": stats.name,
                "count": stats.count,
                "correct": stats.correct,
                "accuracy": stats.accuracy,
                "weighted_score": stats.weighted_score,
            }
            for name, stats in eval_result.question_type_stats.items()
        },
        "metadata": eval_result.metadata,
    }
    _save_json(result, output_dir / "eval_results.json")

    print(f"\n{'=' * 60}")
    print("Narrative-only baseline result")
    print(f"{'=' * 60}")
    print(f"Total Questions: {eval_result.total_questions}")
    print(f"Correct: {eval_result.correct}")
    print(f"Accuracy: {eval_result.accuracy:.4f} ({eval_result.accuracy * 100:.2f}%)")
    if eval_result.weighted_score is not None:
        print(f"Weighted Score: {eval_result.weighted_score:.4f} "
              f"({eval_result.weighted_score * 100:.2f}%)")
    if eval_result.question_type_stats:
        print("\nPer-type accuracy:")
        for name, stats in sorted(eval_result.question_type_stats.items()):
            print(f"  {name}: {stats.accuracy * 100:.2f}% ({stats.correct}/{stats.count})")
    print(f"\nSaved to: {output_dir / 'eval_results.json'}")


async def main():
    parser = argparse.ArgumentParser(description="Narrative-only answerability baseline for LifeBench")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Conversation-format dataset JSON")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N QAs (0 = all)")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent LLM answer calls")
    parser.add_argument("--temperature", type=float, default=0.0, help="Answer LLM temperature")
    parser.add_argument("--skip-answer", action="store_true", help="Skip answer generation, only evaluate")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, only generate answers")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_records = load_qa_records(Path(args.dataset))
    if args.limit > 0:
        qa_records = qa_records[: args.limit]

    print(f"Loaded {len(qa_records)} QA records from {args.dataset}")

    if not args.skip_answer:
        await run_answer(
            qa_records, output_dir,
            concurrency=args.concurrency, temperature=args.temperature,
        )

    if not args.skip_eval:
        await run_eval(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
