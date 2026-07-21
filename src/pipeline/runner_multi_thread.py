"""
Evaluation pipeline runner - multi-threaded batch processing mode.
"""
import asyncio
import json
import os
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed as thread_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

os.environ["PYTHONASYNCIODEBUG"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.evaluators.base import BaseEvaluator
from src.models import Dataset, SearchResult, AnswerResult
from src.models.message import Conversation, Message
from src.pipeline.checkpoint import CheckpointManager
from src.utils import get_deepseek_balance


class PipelineMultiThread:
    """
    Evaluation Pipeline - multi-threaded batch processing mode.

    Flow: For each date in sorted order:
        ADD: Ingest sessions from this date
        SEARCH: Retrieve memories for QAs from this date
        ANSWER: Generate answers for this date's QAs
    EVALUATE: Evaluate all answers
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        evaluator: BaseEvaluator,
        output_dir: Path,
        run_name: str = "default",
        use_checkpoint: bool = True,
        filter_categories: Optional[List[str]] = None,
        stats_collector=None,
        debug: bool = False,
        track_cost: bool = False,
    ):
        self.adapter = adapter
        self.evaluator = evaluator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.use_checkpoint = use_checkpoint
        self.filter_categories = filter_categories or []
        self._stats_collector = stats_collector
        self.debug = debug
        self.track_cost = track_cost

        self.checkpoint = (
            CheckpointManager(output_dir=self.output_dir, run_name=run_name)
            if use_checkpoint
            else None
        )
        self._results: Dict[str, Any] = {}

        self._debug_dir = self.output_dir / "debug" if debug else None
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        dataset: Dataset,
        stages: Optional[List[str]] = None,
        smoke_test: bool = False,
        smoke_messages: int = 10,
        smoke_questions: int = 3,
        from_conv: int = 0,
        to_conv: Optional[int] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        self._print_header(dataset)

        self._load_existing_results()

        if from_conv > 0 or to_conv is not None:
            dataset = self._apply_conversation_range(dataset, from_conv, to_conv)

        if smoke_test:
            dataset = self._apply_smoke_test(dataset, smoke_messages, smoke_questions)

        dataset = self._apply_category_filter(dataset)

        if len(dataset.conversations) == 0:
            print("[red] No conversations to process![/red]")
            return {"error": "No conversations selected"}

        dataset, date_info = self._extract_date_info(dataset)

        if stages is None:
            stages = ["add", "search", "answer", "evaluate"]

        await self._run_date_ordered(dataset, date_info, stages)

        elapsed = time.time() - start_time
        self._generate_report(elapsed)

        return self._results

    async def _run_date_ordered(
        self,
        dataset: Dataset,
        date_info: Dict[str, Any],
        stages: List[str],
    ) -> None:
        import time as time_module

        stage_timings: Dict[str, Dict[str, float]] = {}
        add_search_balance_before = None
        add_search_balance_after = None

        if self.track_cost and ("add" in stages or "search" in stages):
            try:
                add_search_balance_before = get_deepseek_balance(".env")
                for info in add_search_balance_before.get("balance_infos", []):
                    if info.get("currency") == "CNY":
                        add_search_balance_before = float(info.get("total_balance", "0"))
                        break
            except Exception:
                add_search_balance_before = None

        sorted_dates = date_info["sorted_dates"]
        sessions_by_date = date_info["sessions_by_date"]
        qas_by_date = date_info["qas_by_date"]
        ordering_info = date_info["ordering_info"]

        print(f"\n🚀 Date-ordered workflow: {len(sorted_dates)} days to process")
        print(f"   Dates: {[d.strftime('%Y-%m-%d') for d in sorted_dates[:5]]}"
              f"{'...' if len(sorted_dates) > 5 else ''}")

        all_qa_pairs: List[Any] = []
        all_search_results: List[SearchResult] = self._results.get("search_results", [])

        sample_ids = list({conv.conversation_id for conv in dataset.conversations})
        print(f"\n👥 {len(sample_ids)} samples to process")

        sample_data: Dict[str, Dict[str, Any]] = {}
        for sample_id in sample_ids:
            sample_sessions = []
            sample_qas = []

            for date in sorted_dates:
                date_str = date.strftime("%Y-%m-%d")
                session_ids = sessions_by_date.get(date_str, [])
                qa_pairs = qas_by_date.get(date_str, [])

                for sid in session_ids:
                    if ":" in sid:
                        conv_id, _ = sid.split(":", 1)
                        if conv_id == sample_id:
                            sample_sessions.append((date_str, sid))

                for qa in qa_pairs:
                    if qa.metadata.get("conversation_id") == sample_id:
                        sample_qas.append(qa)

            sample_data[sample_id] = {
                "sessions": sample_sessions,
                "qas": sample_qas,
            }

        concurrency = self.adapter.config.get("add", {}).get("num_workers", 10)
        thread_count = self.adapter.config.get("thread_count", 10)
        thread_batch_size = self.adapter.config.get("thread_batch_size", 1)

        if self.checkpoint and self.checkpoint.has_any_progress():
            progress = self.checkpoint.get_progress_summary()
            print(f"\n🔄 [ADD+SEARCH] Resuming from checkpoint (last updated: {progress['last_updated']})")

        async def process_sample_semaphore(sample_id: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
            """Process all stages for a single sample with semaphore control."""
            async with semaphore:
                result = {
                    "sample_id": sample_id,
                    "add_latency": [],
                    "search_latency": [],
                    "search_results": [],
                    "qas": sample_data[sample_id]["qas"],
                }

                sample_sessions = sample_data[sample_id]["sessions"]
                sample_qas = sample_data[sample_id]["qas"]

                if not sample_sessions:
                    return result

                session_dates = set(date_str for date_str, _ in sample_sessions)
                qa_dates = set(qa.metadata.get("ask_time", "")[:10] for qa in sample_qas if qa.metadata.get("ask_time"))
                all_dates = sorted(session_dates | qa_dates)

                sample_add_done = self.checkpoint and self.checkpoint.is_sample_add_complete(sample_id)
                sample_search_done = self.checkpoint and self.checkpoint.is_sample_search_complete(sample_id)

                if "add" in stages or "search" in stages:
                    pbar = tqdm(all_dates, desc=f"👤 {sample_id[:6]} | {len(sample_qas)} QAs", leave=True)
                    for date_str in pbar:
                        session_ids = [sid for d, sid in sample_sessions if d == date_str]
                        date_qas = [qa for qa in sample_qas if qa.metadata.get("ask_time", "").startswith(date_str)]

                        has_session = bool(session_ids)
                        has_qa = bool(date_qas)

                        date_add_done = self.checkpoint and self.checkpoint.is_date_add_complete(sample_id, date_str)
                        date_search_done = self.checkpoint and self.checkpoint.is_date_search_complete(sample_id, date_str)

                        if has_session and "add" in stages and date_add_done:
                            has_session = False
                        if has_qa and "search" in stages and date_search_done:
                            has_qa = False

                        if not has_session and not has_qa:
                            continue

                        ops = f"{date_str} | {'➕ ADD' if has_session else ''}{'🔍 SEARCH' if has_qa else ''}"
                        pbar.set_description(f"👤 {sample_id[:6]} | {ops}")

                        if has_session and "add" in stages and not date_add_done:
                            chunks = self._get_chunks_for_date(dataset, date_str, session_ids, ordering_info)

                            debug_file = None
                            if self._debug_dir:
                                debug_path = self._debug_dir / f"add_{sample_id}_{date_str}.txt"
                                debug_file = open(debug_path, "w", encoding="utf-8")
                                debug_file.write(f"{'=' * 80}\n")
                                debug_file.write(f"SAMPLE: {sample_id} | DATE: {date_str}\n")
                                debug_file.write(f"Sessions: {len(session_ids)}, Chunks: {len(chunks)}\n")
                                debug_file.write(f"{'=' * 80}\n\n")

                            try:
                                for session_id, session_chunks in self._group_chunks_by_session(chunks).items():
                                    if debug_file:
                                        debug_file.write(f"\n{'---' * 27}\n")
                                        debug_file.write(f"SESSION: {session_id}\n")
                                        debug_file.write(f"Chunks: {len(session_chunks)}\n")
                                        for chunk in session_chunks:
                                            debug_file.write(f"--- Chunk ({len(chunk.messages)} messages) ---\n")
                                            for msg in chunk.messages:
                                                debug_file.write(f"  {msg.speaker_name}: {msg.content}\n")
                                            debug_file.write("\n")

                                    balance_before = None
                                    balance_after = None
                                    if self.track_cost:
                                        balance_before = get_deepseek_balance(".env")
                                    start = time.perf_counter()
                                    r = await self.adapter.add_chunks(session_chunks)
                                    latency = time.perf_counter() - start
                                    if self.track_cost:
                                        balance_after = get_deepseek_balance(".env")
                                    total_messages = sum(len(c.messages) for c in session_chunks)

                                    def get_total(bal):
                                        if bal is None:
                                            return None
                                        for info in bal.get("balance_infos", []):
                                            if info.get("currency") == "CNY":
                                                return float(info.get("total_balance", "0"))
                                        return None
                                    cost = None
                                    if balance_before is not None and balance_after is not None:
                                        cost = get_total(balance_before) - get_total(balance_after)
                                    entry = {
                                        "date": date_str,
                                        "session_id": session_id,
                                        "num_chunks": len(session_chunks),
                                        "num_messages": total_messages,
                                        "latency_seconds": round(latency, 3),
                                        "added": r.get("added", 0),
                                        "failed": r.get("failed", 0),
                                    }
                                    if self.track_cost:
                                        entry["balance_before"] = get_total(balance_before)
                                        entry["balance_after"] = get_total(balance_after)
                                        entry["cost_cny"] = round(cost, 4)
                                    result["add_latency"].append(entry)
                            finally:
                                if debug_file:
                                    debug_file.write(f"\n{'=' * 80}\n")
                                    debug_file.write(f"SUMMARY: {len(chunks)} chunks, {len(date_qas)} searches\n")
                                    debug_file.write(f"{'=' * 80}\n\n")
                                    debug_file.close()

                        if has_qa and "search" in stages and not date_search_done:
                            for qa in date_qas:
                                start = time.perf_counter()
                                sr = await self.adapter.search(
                                    qa.question,
                                    sample_id,
                                    self._results.get("index"),
                                    question_id=qa.question_id,
                                )
                                latency = time.perf_counter() - start
                                result["search_latency"].append({
                                    "question_id": qa.question_id,
                                    "conversation_id": sample_id,
                                    "latency_seconds": round(latency, 3),
                                })
                                result["search_results"].append(sr)

                        if result["search_results"]:
                            self._append_search_results(result["search_results"])
                        if result["add_latency"]:
                            self._results.setdefault("add_latency", []).extend(result["add_latency"])
                            self._save_json(self._results["add_latency"], "add_latency.json")
                        if result["search_latency"]:
                            self._results.setdefault("search_latency", []).extend(result["search_latency"])
                            self._save_json(self._results["search_latency"], "search_latency.json")

                        if has_session and "add" in stages and not date_add_done:
                            if self.checkpoint:
                                self.checkpoint.mark_date_add_complete(sample_id, date_str)
                        if has_qa and "search" in stages and not date_search_done:
                            if self.checkpoint:
                                self.checkpoint.mark_date_search_complete(sample_id, date_str)
                            self._save_json(self._results["search_latency"], "search_latency.json")

                    if "add" in stages and not sample_add_done:
                        if self.checkpoint:
                            self.checkpoint.mark_sample_add_complete(sample_id)
                    if "search" in stages and not sample_search_done:
                        if self.checkpoint:
                            self.checkpoint.mark_sample_search_complete(sample_id)

                return result

        add_search_stage_start = time_module.time()

        def run_batch_in_thread(batch_sample_ids: List[str]) -> List[Dict[str, Any]]:
            """Run a batch of samples in a thread with its own event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                semaphore = asyncio.Semaphore(concurrency)
                tasks = [process_sample_semaphore(sid, semaphore) for sid in batch_sample_ids]
                results = loop.run_until_complete(asyncio.gather(*tasks))
                return results
            finally:
                loop.close()

        if sample_data:
            if thread_count > 1 and thread_batch_size > 1:
                batches = []
                for i in range(0, len(sample_ids), thread_batch_size):
                    batches.append(sample_ids[i:i + thread_batch_size])

                print(f"\n🧵 Multi-threaded: {thread_count} threads, {thread_batch_size} samples/batch, {len(batches)} batches")

                with ThreadPoolExecutor(max_workers=thread_count) as executor:
                    futures = {executor.submit(run_batch_in_thread, batch): batch for batch in batches}
                    for future in thread_completed(futures):
                        try:
                            batch_results = future.result()
                            for r in batch_results:
                                all_qa_pairs.extend(r["qas"])
                                new_srs = [sr for sr in r["search_results"] if sr.question_id not in {s.question_id for s in all_search_results}]
                                all_search_results.extend(new_srs)
                        except Exception as e:
                            print(f"  ⚠️ Batch failed: {e}")
            else:
                semaphore = asyncio.Semaphore(concurrency)
                tasks = [process_sample_semaphore(sid, semaphore) for sid in sample_ids]
                results = await asyncio.gather(*tasks)
                for r in results:
                    all_qa_pairs.extend(r["qas"])
                    new_srs = [sr for sr in r["search_results"] if sr.question_id not in {s.question_id for s in all_search_results}]
                    all_search_results.extend(new_srs)

        add_search_stage_elapsed = time_module.time() - add_search_stage_start
        stage_timings["add_search"] = {"elapsed": add_search_stage_elapsed}

        from src.models.search import SearchResult, RetrievedMemory
        results_dir = self.output_dir / "results"

        def dict_to_search_result_local(d: dict) -> SearchResult:
            results = [
                RetrievedMemory(content=r["content"], score=r["score"], metadata=r.get("metadata", {}))
                if isinstance(r, dict) else r
                for r in d.get("results", [])
            ]
            return SearchResult(
                question_id=d["question_id"],
                query=d["query"],
                conversation_id=d["conversation_id"],
                results=results,
                retrieval_metadata=d.get("retrieval_metadata", {}),
            )

        if results_dir.exists():
            loaded_count = 0
            for result_file in results_dir.glob("*.json"):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and data.get("question_id"):
                        sr = dict_to_search_result_local(data)
                        existing_idx = next((i for i, s in enumerate(all_search_results) if s.question_id == sr.question_id), None)
                        if existing_idx is not None:
                            all_search_results[existing_idx] = sr
                        else:
                            all_search_results.append(sr)
                        loaded_count += 1
                except Exception as e:
                    print(f"  ⚠️ Failed to load {result_file}: {e}")
            if loaded_count > 0:
                print(f"  📂 Loaded {loaded_count} search results from {results_dir}/")

        answer_stage_start = time_module.time()
        if "answer" in stages:
            print(f"\n{'=' * 60}")
            print("Phase 3: ANSWER")
            print(f"{'=' * 60}")

            if not all_qa_pairs:
                print("  ⚠️ No QA pairs to answer")
            else:
                answer_concurrency = self.adapter.config.get("answer", {}).get("num_workers", 5)
                answer_semaphore = asyncio.Semaphore(answer_concurrency)

                from src.formatters import format_context
                sr_map = {sr.question_id: sr for sr in all_search_results}

                async def process_qa(qa_pair):
                    async with answer_semaphore:
                        start = time.perf_counter()
                        sr = sr_map.get(qa_pair.question_id)
                        if sr is None:
                            return AnswerResult(
                                question_id=qa_pair.question_id,
                                question=qa_pair.question,
                                answer="",
                                golden_answer=getattr(qa_pair, "answer", ""),
                                category=getattr(qa_pair, "category", ""),
                                conversation_id=qa_pair.metadata.get("conversation_id", ""),
                                formatted_context="",
                                metadata=qa_pair.metadata,
                            )
                        context = format_context(sr)
                        answer = await self.adapter.answer(
                            query=qa_pair.question,
                            context=context,
                            conversation_id=sr.conversation_id,
                        )
                        latency = time.perf_counter() - start
                        return AnswerResult(
                            question_id=qa_pair.question_id,
                            question=qa_pair.question,
                            answer=answer,
                            golden_answer=getattr(qa_pair, "answer", ""),
                            category=getattr(qa_pair, "category", ""),
                            conversation_id=sr.conversation_id,
                            formatted_context=context,
                            metadata=qa_pair.metadata,
                        )

                pending = [process_qa(qa) for qa in all_qa_pairs]
                answer_results = []
                for fut in tqdm(asyncio.as_completed(pending), total=len(pending), desc="  💬 Answering"):
                    answer_results.append(await fut)

                self._results["answers"] = answer_results
                self._save_json([self._answer_result_to_dict(ar) for ar in answer_results], "answers.json")

        answer_stage_elapsed = time_module.time() - answer_stage_start
        stage_timings["answer"] = {"elapsed": answer_stage_elapsed}

        if "evaluate" in stages:
            print(f"\n{'=' * 60}")
            print("Phase 4: EVALUATE")
            print(f"{'=' * 60}")

            if not self._results.get("answers"):
                print("  ⚠️ No answers to evaluate")
            else:
                metrics = await self.evaluator.evaluate(self._results.get("answers", []))
                self._results["metrics"] = metrics
                print(f"\n  📊 Metrics: {metrics}")
                self._save_json(self._eval_result_to_dict(metrics), "metrics.json")

        print(f"\n{'=' * 60}")
        print("Pipeline complete!")
        print(f"{'=' * 60}")

    def _print_header(self, dataset: Dataset):
        print(f"\n{'=' * 60}")
        print("LifeBench Evaluation Pipeline")
        print(f"{'=' * 60}")
        print(f"  System:     {self.adapter.__class__.__name__}")
        print(f"  Dataset:    {dataset.dataset_name}")
        print(f"  Convs:      {len(dataset.conversations)}")
        print(f"  Questions:  {len(dataset.qa_pairs)}")
        print(f"  Stages:     add → search → answer → evaluate")
        print(f"{'=' * 60}\n")

    def _load_existing_results(self):
        index_file = self.output_dir / "index.json"
        if index_file.exists():
            try:
                import json
                with open(index_file, "r", encoding="utf-8") as f:
                    self._results["index"] = json.load(f)
                print(f"  📂 Loaded existing index: {index_file}")
            except Exception as e:
                print(f"  ⚠️ Failed to load index: {e}")

    def _apply_conversation_range(self, dataset: Dataset, from_conv: int, to_conv: Optional[int]) -> Dataset:
        if from_conv > 0:
            dataset.conversations = dataset.conversations[from_conv:]
        if to_conv is not None:
            dataset.conversations = dataset.conversations[:to_conv]
        return dataset

    def _apply_smoke_test(self, dataset: Dataset, messages: int, questions: int) -> Dataset:
        for conv in dataset.conversations:
            conv.messages = conv.messages[:messages]
        dataset.qa_pairs = dataset.qa_pairs[:questions]
        return dataset

    def _apply_category_filter(self, dataset: Dataset) -> Dataset:
        if not self.filter_categories:
            return dataset
        original = len(dataset.conversations)
        dataset.conversations = [
            c for c in dataset.conversations
            if c.metadata.get("category") in self.filter_categories
        ]
        print(f"  🔍 Filtered to {len(dataset.conversations)}/{original} conversations")
        return dataset

    def _extract_date_info(self, dataset: Dataset) -> tuple:
        sessions_by_date = defaultdict(list)
        qas_by_date = defaultdict(list)
        date_to_conversation = defaultdict(set)
        all_dates = set()
        ordering_info = {}

        for conv in dataset.conversations:
            raw_conv = conv.metadata.get("_raw_conversation", {})
            session_keys = sorted(
                [k for k in raw_conv.keys() if k.startswith("session_") and "_date_time" not in k],
                key=lambda x: int(re.search(r"session_(\d+)", x).group(1))
            )

            for session_key in session_keys:
                # Use session_N_date_time to get session date (same as runner.py)
                date_time_key = f"{session_key}_date_time"
                date_time_val = raw_conv.get(date_time_key, "")
                if date_time_val:
                    try:
                        dt = datetime.fromisoformat(date_time_val.replace("Z", "+00:00"))
                        date_str = dt.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        date_str = date_time_val[:10] if len(date_time_val) >= 10 else ""
                    if date_str:
                        all_dates.add(date_str)
                        sessions_by_date[date_str].append(f"{conv.conversation_id}:{session_key}")
                        date_to_conversation[date_str].add(conv.conversation_id)

            conv_qa_pairs = [qa for qa in dataset.qa_pairs if qa.metadata.get("conversation_id") == conv.conversation_id]
            for qa in conv_qa_pairs:
                if qa.metadata.get("ask_time"):
                    date_str = qa.metadata["ask_time"][:10]
                    all_dates.add(date_str)
                    qas_by_date[date_str].append(qa)
                    date_to_conversation[date_str].add(conv.conversation_id)

        sorted_dates = sorted(all_dates)

        for date_str in sorted_dates:
            conv_ids = date_to_conversation[date_str]
            date_objects = []
            for conv_id in conv_ids:
                for conv in dataset.conversations:
                    if conv.conversation_id == conv_id:
                        raw_conv = conv.metadata.get("_raw_conversation", {})
                        for session_key in raw_conv:
                            if session_key.startswith("session_") and "_date_time" not in session_key:
                                date_time_key = f"{session_key}_date_time"
                                date_time_val = raw_conv.get(date_time_key, "")
                                if date_time_val:
                                    try:
                                        dt = datetime.fromisoformat(date_time_val.replace("Z", "+00:00"))
                                        if dt.strftime("%Y-%m-%d") == date_str:
                                            date_objects.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
                                    except (ValueError, AttributeError):
                                        pass
                        break
            ordering_info[date_str] = sorted(date_objects) if date_objects else []

        return dataset, {
            "sorted_dates": [datetime.strptime(d, "%Y-%m-%d") for d in sorted_dates],
            "sessions_by_date": dict(sessions_by_date),
            "qas_by_date": dict(qas_by_date),
            "ordering_info": ordering_info,
        }

    def _get_chunks_for_date(
        self, dataset: Dataset, date_str: str, session_ids: List[str], ordering_info: Dict
    ) -> List[ChunkedMessage]:
        """Return one ChunkedMessage per session (all messages in that session)."""
        chunks = []

        sessions_by_conv: Dict[str, List[str]] = defaultdict(list)
        for sid in session_ids:
            if ":" in sid:
                conv_id, session_key = sid.split(":", 1)
                sessions_by_conv[conv_id].append(session_key)

        for conv in dataset.conversations:
            conv_id = conv.conversation_id
            raw_conv = conv.metadata.get("_raw_conversation", {})

            if conv_id not in sessions_by_conv:
                continue

            for session_key in sessions_by_conv[conv_id]:
                session_msgs = raw_conv.get(session_key, [])
                if not isinstance(session_msgs, list):
                    continue

                messages = []
                for msg_data in session_msgs:
                    if isinstance(msg_data, dict):
                        dia_id = msg_data.get("dia_id", "")
                        msg_timestamp = None
                        if dia_id:
                            date_part = dia_id.split("_")[0] if "_" in dia_id else ""
                            if date_part and len(date_part) == 10:
                                try:
                                    msg_timestamp = datetime.strptime(f"{date_part} 23:59:59", "%Y-%m-%d %H:%M:%S")
                                except ValueError:
                                    pass
                        msg = Message(
                            speaker_id=msg_data.get("speaker", ""),
                            speaker_name=msg_data.get("speaker", ""),
                            content=msg_data.get("text", ""),
                            timestamp=msg_timestamp,
                            metadata={"dia_id": dia_id} if dia_id else {},
                        )
                        messages.append(msg)

                if not messages:
                    continue

                unique_session_id = f"{conv_id}:{session_key}"
                chunks.append(ChunkedMessage(
                    messages=messages,
                    conversation_id=conv_id,
                    session_id=unique_session_id,
                    timestamp=int(datetime.strptime(date_str, "%Y-%m-%d").timestamp()),
                ))

        return chunks

    def _group_chunks_by_session(self, chunks: List[ChunkedMessage]) -> Dict[str, List[ChunkedMessage]]:
        sessions = defaultdict(list)
        for chunk in chunks:
            sessions[chunk.session_id].append(chunk)
        return dict(sessions)

    def _append_search_results(self, search_results: List[SearchResult]):
        if not search_results:
            return
        results_dir = self.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        for sr in search_results:
            result_file = results_dir / f"{sr.question_id}.json"
            try:
                existing = {}
                if result_file.exists():
                    with open(result_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                existing.update({
                    "question_id": sr.question_id,
                    "query": sr.query,
                    "conversation_id": sr.conversation_id,
                    "results": [
                        {"content": r.content, "score": r.score, "metadata": r.metadata}
                        for r in sr.results
                    ],
                    "retrieval_metadata": sr.retrieval_metadata,
                })
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  ⚠️ Failed to save search result {sr.question_id}: {e}")

    def _save_json(self, data: Any, filename: str):
        try:
            with open(self.output_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ⚠️ Failed to save {filename}: {e}")

    def _answer_result_to_dict(self, ar: AnswerResult) -> dict:
        return {
            "question_id": ar.question_id,
            "question": ar.question,
            "answer": ar.answer,
            "golden_answer": ar.golden_answer,
            "category": ar.category,
            "conversation_id": ar.conversation_id,
            "formatted_context": ar.formatted_context,
            "search_results": ar.search_results,
            "metadata": ar.metadata,
        }

    def _eval_result_to_dict(self, er) -> dict:
        return {
            "total_questions": er.total_questions,
            "correct": er.correct,
            "accuracy": er.accuracy,
            "weighted_score": er.weighted_score,
            "detailed_results": er.detailed_results,
            "question_type_stats": {
                k: {"name": v.name, "count": v.count, "correct": v.correct, "accuracy": v.accuracy, "weighted_score": v.weighted_score}
                for k, v in er.question_type_stats.items()
            },
            "metadata": er.metadata,
        }

    def _generate_report(self, elapsed: float):
        print(f"\n📊 Results saved to: {self.output_dir}")
        print(f"   ⏱️  Total time: {elapsed:.1f}s")
        if self._stats_collector:
            print(f"\n💰 Cost Report:")
            self._stats_collector.print_summary()
