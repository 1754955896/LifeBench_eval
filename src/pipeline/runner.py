"""
Evaluation pipeline runner - date-ordered mode only.
"""
import asyncio
import os
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Suppress asyncio slow task warnings (noise, not errors)
os.environ["PYTHONASYNCIODEBUG"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.evaluators.base import BaseEvaluator
from src.models import Dataset, SearchResult, AnswerResult
from src.models.message import Conversation, Message
from src.pipeline.checkpoint import CheckpointManager
from src.utils import get_deepseek_balance


class Pipeline:
    """
    Evaluation Pipeline - date-ordered mode only.

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

        # Debug logging
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

        # Load existing results from disk for resume
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

        # Track stage timings
        stage_timings: Dict[str, Dict[str, float]] = {}
        add_search_balance_before = None
        add_search_balance_after = None

        # Track balance before ADD+SEARCH if track_cost enabled
        if self.track_cost and ("add" in stages or "search" in stages):
            try:
                add_search_balance_before = get_deepseek_balance(".env")
                # Extract total CNY balance
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
        # Note: search_results are now incrementally appended to disk via _append_search_results
        # For answer stage, load from disk
        all_search_results: List[SearchResult] = self._results.get("search_results", [])

        # Group sessions and QAs by sample (conversation_id)
        sample_ids = list({conv.conversation_id for conv in dataset.conversations})
        print(f"\n👥 {len(sample_ids)} samples to process")

        # Collect all sample data
        sample_data: Dict[str, Dict[str, Any]] = {}
        for sample_id in sample_ids:
            sample_sessions = []  # list of (date_str, session_id)
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

        # Parallel processing of samples
        concurrency = self.adapter.config.get("add", {}).get("num_workers", 10)
        semaphore = asyncio.Semaphore(concurrency)

        # Check checkpoint for ADD+SEARCH phase
        if self.checkpoint and self.checkpoint.has_any_progress():
            progress = self.checkpoint.get_progress_summary()
            print(f"\n🔄 [ADD+SEARCH] Resuming from checkpoint (last updated: {progress['last_updated']})")

        async def process_sample(sample_id: str) -> Dict[str, Any]:
            """Process all stages for a single sample."""
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

                # Collect all dates (from sessions AND from QA ask_times)
                session_dates = set(date_str for date_str, _ in sample_sessions)
                qa_dates = set(qa.metadata.get("ask_time", "")[:10] for qa in sample_qas if qa.metadata.get("ask_time"))
                all_dates = sorted(session_dates | qa_dates)

                # Check completion status
                sample_add_done = self.checkpoint and self.checkpoint.is_sample_add_complete(sample_id)
                sample_search_done = self.checkpoint and self.checkpoint.is_sample_search_complete(sample_id)

                # Phase 1: ADD + SEARCH combined per date (ordered)
                if "add" in stages or "search" in stages:
                    pbar = tqdm(all_dates, desc=f"👤 {sample_id[:6]} | {len(sample_qas)} QAs", leave=True)
                    for date_str in pbar:
                        session_ids = [sid for d, sid in sample_sessions if d == date_str]
                        date_qas = [qa for qa in sample_qas if qa.metadata.get("ask_time", "").startswith(date_str)]

                        has_session = bool(session_ids)
                        has_qa = bool(date_qas)

                        # Check date-level completion
                        date_add_done = self.checkpoint and self.checkpoint.is_date_add_complete(sample_id, date_str)
                        date_search_done = self.checkpoint and self.checkpoint.is_date_search_complete(sample_id, date_str)

                        # Skip if both ADD and SEARCH for this date are done
                        if has_session and "add" in stages and date_add_done:
                            has_session = False
                        if has_qa and "search" in stages and date_search_done:
                            has_qa = False

                        if not has_session and not has_qa:
                            continue

                        ops = f"{date_str} | {'➕ ADD' if has_session else ''}{'🔍 SEARCH' if has_qa else ''}"
                        pbar.set_description(f"👤 {sample_id[:6]} | {ops}")

                        # ADD: if date has session and add stage requested and not done for this date
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

                                    # Pass all session chunks at once so the adapter can merge
                                    # them into a single memory-system call for efficient batch processing
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
                                    # Extract total_balance from balance_infos
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

                        # SEARCH: if date has QA and search stage requested and not done for this date
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

                        # Incremental append after each date (deduplicate by question_id)
                        if result["search_results"]:
                            self._append_search_results(result["search_results"])
                        if result["add_latency"]:
                            self._results.setdefault("add_latency", []).extend(result["add_latency"])
                            self._save_json(self._results["add_latency"], "add_latency.json")
                        if result["search_latency"]:
                            self._results.setdefault("search_latency", []).extend(result["search_latency"])
                            self._save_json(self._results["search_latency"], "search_latency.json")

                        # Mark date-level completion (after save to avoid losing data on crash)
                        if has_session and "add" in stages and not date_add_done:
                            if self.checkpoint:
                                self.checkpoint.mark_date_add_complete(sample_id, date_str)
                        if has_qa and "search" in stages and not date_search_done:
                            if self.checkpoint:
                                self.checkpoint.mark_date_search_complete(sample_id, date_str)
                            self._save_json(self._results["search_latency"], "search_latency.json")

                    # Mark sample-level completion after all dates processed
                    if "add" in stages and not sample_add_done:
                        if self.checkpoint:
                            self.checkpoint.mark_sample_add_complete(sample_id)
                    if "search" in stages and not sample_search_done:
                        if self.checkpoint:
                            self.checkpoint.mark_sample_search_complete(sample_id)

                return result

        # Process samples (parallel or serial based on config)
        add_search_stage_start = time_module.time()
        sample_parallel = self.adapter.config.get("sample_parallel", True)
        if sample_data:
            if sample_parallel:
                # Process and save incrementally as each sample completes
                pending = [process_sample(sid) for sid in sample_ids]
                for fut in asyncio.as_completed(pending):
                    r = await fut
                    all_qa_pairs.extend(r["qas"])
                    new_srs = [sr for sr in r["search_results"] if sr.question_id not in {s.question_id for s in all_search_results}]
                    all_search_results.extend(new_srs)
                    # Note: save is already done incrementally inside process_sample after each date
            else:
                # Serial processing
                for sid in sample_ids:
                    r = await process_sample(sid)
                    all_qa_pairs.extend(r["qas"])
                    new_srs = [sr for sr in r["search_results"] if sr.question_id not in {s.question_id for s in all_search_results}]
                    all_search_results.extend(new_srs)
                    # Note: save is already done incrementally inside process_sample after each date

        add_search_stage_elapsed = time_module.time() - add_search_stage_start
        stage_timings["add_search"] = {"elapsed": add_search_stage_elapsed}

        # Load individual result files from results/ directory BEFORE answer stage
        # This ensures buffered search results (saved by adapter's _save_search_result) are available
        import json
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
                        # Replace existing or add new (prefer the complete one from file)
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

        # Phase 3: ANSWER
        answer_stage_start = time_module.time()
        if "answer" in stages:
            print(f"\n{'=' * 60}")
            print(f"💬 [ANSWER] {len(all_qa_pairs)} QAs total")

            answered_ids = self.checkpoint.get_answered_qa_ids() if self.checkpoint else set()
            remaining_qas = [qa for qa in all_qa_pairs if qa.question_id not in answered_ids]
            print(f"   Already answered: {len(answered_ids)}, Remaining: {len(remaining_qas)}")

            if remaining_qas:
                sr_map = {sr.question_id: sr for sr in all_search_results}
                aligned_srs = [sr_map.get(qa.question_id) for qa in remaining_qas]
                valid_pairs = [(qa, sr) for qa, sr in zip(remaining_qas, aligned_srs) if sr is not None]

                if valid_pairs:
                    valid_qas, valid_srs = zip(*valid_pairs)
                    answer_results = await self._run_answer_for_qas_with_progress(list(valid_qas), list(valid_srs))
                    answered_ids.update([ar.question_id for ar in answer_results])
        answer_stage_elapsed = time_module.time() - answer_stage_start
        stage_timings["answer"] = {"elapsed": answer_stage_elapsed, "count": len(all_qa_pairs)}

        # Phase 4: EVALUATE
        evaluate_stage_start = time_module.time()
        if "evaluate" in stages:
            print(f"\n{'=' * 60}")
            print("⚖️  [EVALUATE] All answers")

            if self.checkpoint and self.checkpoint.is_evaluate_complete():
                print("   SKIP (already completed)")
            else:
                eval_result = await self.evaluator.evaluate(self._results.get("answer_results", []))
                self._results["eval_result"] = eval_result
                self._save_json(self._eval_result_to_dict(eval_result), "eval_results.json")

                if self.checkpoint:
                    self.checkpoint.mark_evaluate_complete()
        evaluate_stage_elapsed = time_module.time() - evaluate_stage_start
        stage_timings["evaluate"] = {"elapsed": evaluate_stage_elapsed, "count": len(self._results.get("answer_results", []))}

        # Track balance after ADD+SEARCH if track_cost enabled
        if self.track_cost and ("add" in stages or "search" in stages):
            try:
                add_search_balance_after = get_deepseek_balance(".env")
                for info in add_search_balance_after.get("balance_infos", []):
                    if info.get("currency") == "CNY":
                        add_search_balance_after = float(info.get("total_balance", "0"))
                        break
            except Exception:
                add_search_balance_after = None

        # Store stage timings and balance info for report
        self._results["stage_timings"] = stage_timings
        self._results["add_search_balance_before"] = add_search_balance_before
        self._results["add_search_balance_after"] = add_search_balance_after

        # Load individual result files from results/ directory (saved by adapter's _save_search_result)
        import json
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
            for result_file in results_dir.glob("*.json"):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and data.get("question_id"):
                        sr = dict_to_search_result_local(data)
                        # Replace existing or add new
                        existing_idx = next((i for i, s in enumerate(all_search_results) if s.question_id == sr.question_id), None)
                        if existing_idx is not None:
                            all_search_results[existing_idx] = sr
                        else:
                            all_search_results.append(sr)
                except Exception as e:
                    print(f"  ⚠️ Failed to load {result_file}: {e}")

        # Save all results
        self._save_json(
            [self._search_result_to_dict(sr) for sr in all_search_results],
            "search_results.json",
        )

        if self._results.get("add_latency"):
            self._save_json(self._results["add_latency"], "add_latency.json")

        if self._results.get("search_latency"):
            self._save_json(self._results["search_latency"], "search_latency.json")

        self._results["ordering_info"] = ordering_info

    def _get_chunks_for_date(
        self,
        dataset: Dataset,
        date_str: str,
        session_ids: List[str],
        ordering_info: Dict[str, Any],
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
                        # 从 dia_id 提取日期时间 (格式: "2025-01-01_fitness_health0")
                        dia_id = msg_data.get("dia_id", "")
                        msg_timestamp = None
                        if dia_id:
                            date_part = dia_id.split("_")[0] if "_" in dia_id else ""
                            if date_part and len(date_part) == 10:
                                msg_timestamp = datetime.strptime(f"{date_part} 23:59:59", "%Y-%m-%d %H:%M:%S")
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
        """Group chunks by session_id."""
        chunks_by_session = defaultdict(list)
        for chunk in chunks:
            chunks_by_session[chunk.session_id].append(chunk)
        return chunks_by_session

    async def _run_search_for_qas(
        self,
        qa_pairs: List[Any],
        ordering_info: Dict[str, Any],
    ) -> List[SearchResult]:
        concurrency = 10  # serial search to avoid overwhelming server
        semaphore = asyncio.Semaphore(concurrency)

        async def search_one(qa):
            async with semaphore:
                result = await self.adapter.search(
                    qa.question,
                    qa.metadata.get("conversation_id", ""),
                    self._results.get("index"),
                    question_id=qa.question_id,
                )
                interval = self.adapter.config.get("search", {}).get("search_interval", 0)
                if interval > 0:
                    await asyncio.sleep(interval)
                return result

        tasks = [search_one(qa) for qa in qa_pairs]
        return await asyncio.gather(*tasks)

    async def _run_answer_for_qas(
        self,
        qa_pairs: List[Any],
        search_results: List[SearchResult],
    ) -> List[AnswerResult]:
        from src.formatters import format_context

        concurrency = self.adapter.config.get("answer", {}).get("num_workers", 10)
        semaphore = asyncio.Semaphore(concurrency)

        async def answer_one(qa, sr):
            async with semaphore:
                context = format_context(sr)
                answer = await self.adapter.answer(
                    query=qa.question,
                    context=context,
                    conversation_id=sr.conversation_id,
                    search_result=sr,
                )
                return AnswerResult(
                    question_id=qa.question_id,
                    question=qa.question,
                    answer=answer,
                    golden_answer=qa.answer,
                    category=qa.category,
                    conversation_id=sr.conversation_id,
                    formatted_context=context,
                    metadata=qa.metadata,
                )

        tasks = [answer_one(qa, sr) for qa, sr in zip(qa_pairs, search_results)]
        return await asyncio.gather(*tasks)

    async def _run_answer_for_qas_with_progress(
        self,
        qa_pairs: List[Any],
        search_results: List[SearchResult],
    ) -> List[AnswerResult]:
        """Run answer with progress bar and incremental save."""
        pbar = tqdm(total=len(qa_pairs), desc="💬 ANSWER", leave=True)
        concurrency = self.adapter.config.get("answer", {}).get("num_workers", 10)
        semaphore = asyncio.Semaphore(concurrency)

        all_results: List[AnswerResult] = []
        existing_results = self._results.get("answer_results", [])
        answered_count = len(existing_results)

        async def answer_one(qa, sr):
            async with semaphore:
                from src.formatters import format_context
                context = format_context(sr)
                answer = await self.adapter.answer(
                    query=qa.question,
                    context=context,
                    conversation_id=sr.conversation_id,
                    search_result=sr,
                )
                pbar.update(1)
                return AnswerResult(
                    question_id=qa.question_id,
                    question=qa.question,
                    answer=answer,
                    golden_answer=qa.answer,
                    category=qa.category,
                    conversation_id=sr.conversation_id,
                    formatted_context=context,
                    metadata=qa.metadata,
                )

        # Process with incremental save
        tasks = [answer_one(qa, sr) for qa, sr in zip(qa_pairs, search_results)]
        pending = {asyncio.create_task(t): i for i, t in enumerate(tasks)}

        for fut in asyncio.as_completed(pending):
            result = await fut
            all_results.append(result)
            # Incremental save after each answer
            self._results.setdefault("answer_results", []).append(result)
            self._save_json(
                [self._answer_result_to_dict(ar) for ar in self._results["answer_results"]],
                "answer_results.json",
            )
            # Update checkpoint
            if self.checkpoint:
                self.checkpoint.mark_answer_complete([result.question_id])

        pbar.close()
        return all_results

    def _extract_date_info(self, dataset: Dataset) -> tuple:
        DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
        SESSION_DATE_PATTERN = re.compile(r"session_(\d+)_date_time")

        def parse_date(date_str: str) -> datetime:
            if not isinstance(date_str, str) or not date_str.strip():
                return datetime.max
            # Try ISO format first: 2023-05-08
            try:
                return datetime.strptime(date_str.strip(), "%Y-%m-%d")
            except ValueError:
                pass
            # Try locomo natural language: "1:56 pm on 8 May, 2023"
            try:
                return datetime.strptime(date_str.strip(), "%I:%M %p on %d %B, %Y")
            except ValueError:
                pass
            # Try with abbreviated month: "1:56 pm on 8 May 2023"
            try:
                return datetime.strptime(date_str.strip(), "%I:%M %p on %d %B %Y")
            except ValueError:
                pass
            return datetime.max

        def extract_qa_date(qa) -> datetime:
            match = DATE_PATTERN.search(qa.question)
            if match:
                return parse_date(match.group(1))
            ask_time = qa.metadata.get("ask_time", "")
            if ask_time:
                return parse_date(ask_time)
            return datetime.max

        all_dates: Set[datetime] = set()
        sessions_by_date: Dict[str, List[str]] = defaultdict(list)
        qas_by_date: Dict[str, List[Any]] = defaultdict(list)
        ordering_info: Dict[str, Dict[str, Any]] = {}

        for conv in dataset.conversations:
            conv_id = conv.conversation_id
            conv_dict = conv.metadata.get("_raw_conversation", {})

            session_dates = {}
            for key, value in conv_dict.items():
                match = SESSION_DATE_PATTERN.match(key)
                if match:
                    session_num = match.group(1)
                    unique_session_id = f"{conv_id}:session_{session_num}"
                    parsed = parse_date(value)
                    session_dates[unique_session_id] = parsed
                    if parsed != datetime.max:
                        date_str = parsed.strftime("%Y-%m-%d")
                        sessions_by_date[date_str].append(unique_session_id)
                        all_dates.add(parsed)

            sorted_session_ids = sorted(session_dates.keys(), key=lambda s: session_dates[s])

            conv_qas = [qa for qa in dataset.qa_pairs if qa.metadata.get("conversation_id") == conv_id]
            for qa in conv_qas:
                qa_date = extract_qa_date(qa)
                date_str = qa_date.strftime("%Y-%m-%d")
                if qa_date != datetime.max:
                    qas_by_date[date_str].append(qa)
                    all_dates.add(qa_date)

            ordering_info[conv_id] = {
                "session_order": sorted_session_ids,
                "qa_order": [qa.question_id for qa in conv_qas],
            }

        sorted_dates = sorted(all_dates)

        def get_first_session_date(conv):
            conv_dict = conv.metadata.get("_raw_conversation", {})
            for key, value in conv_dict.items():
                match = SESSION_DATE_PATTERN.match(key)
                if match:
                    return parse_date(value)
            return datetime.max

        sorted_convs = sorted(dataset.conversations, key=get_first_session_date)

        sorted_qa_pairs = []
        for conv in sorted_convs:
            conv_id = conv.conversation_id
            conv_qas = [qa for qa in dataset.qa_pairs if qa.metadata.get("conversation_id") == conv_id]
            sorted_qas = sorted(conv_qas, key=extract_qa_date)
            sorted_qa_pairs.extend(sorted_qas)

        updated_dataset = Dataset(
            dataset_name=dataset.dataset_name,
            conversations=sorted_convs,
            qa_pairs=sorted_qa_pairs,
            metadata={**dataset.metadata, "date_ordered": True},
        )

        sessions_by_date_str = {k.strftime("%Y-%m-%d") if isinstance(k, datetime) else k: v for k, v in sessions_by_date.items()}
        qas_by_date_str = {k.strftime("%Y-%m-%d") if isinstance(k, datetime) else k: v for k, v in qas_by_date.items()}

        date_info = {
            "sorted_dates": sorted_dates,
            "sessions_by_date": sessions_by_date_str,
            "qas_by_date": qas_by_date_str,
            "ordering_info": ordering_info,
        }

        return updated_dataset, date_info

    def _apply_conversation_range(self, dataset: Dataset, from_conv: int, to_conv: Optional[int]) -> Dataset:
        if not dataset.conversations:
            return dataset

        total_convs = len(dataset.conversations)
        end_idx = to_conv if to_conv is not None else total_convs

        if from_conv < 0:
            from_conv = 0
        if from_conv >= total_convs:
            return Dataset(dataset_name=dataset.dataset_name, conversations=[], qa_pairs=[], metadata=dataset.metadata)

        selected_convs = dataset.conversations[from_conv:end_idx]
        selected_ids = {c.conversation_id for c in selected_convs}
        selected_qa = [qa for qa in dataset.qa_pairs if qa.metadata.get("conversation_id") in selected_ids]

        return Dataset(
            dataset_name=dataset.dataset_name,
            conversations=selected_convs,
            qa_pairs=selected_qa,
            metadata={**dataset.metadata, "conversation_range": [from_conv, end_idx]},
        )

    def _apply_smoke_test(self, dataset: Dataset, num_messages: int, num_questions: int) -> Dataset:
        trimmed_convs = []
        trimmed_qa = []

        for conv in dataset.conversations:
            trimmed_convs.append(conv)
            conv_qa = [qa for qa in dataset.qa_pairs if qa.metadata.get("conversation_id") == conv.conversation_id]
            trimmed_qa.extend(conv_qa[:num_questions] if num_questions > 0 else conv_qa)

        return Dataset(
            dataset_name=dataset.dataset_name + "_smoke",
            conversations=trimmed_convs,
            qa_pairs=trimmed_qa,
            metadata={**dataset.metadata, "smoke_test": True},
        )

    def _apply_category_filter(self, dataset: Dataset) -> Dataset:
        if not self.filter_categories:
            return dataset

        filter_set = {str(c) for c in self.filter_categories}
        filtered_qa = [qa for qa in dataset.qa_pairs if qa.category not in filter_set]

        if len(filtered_qa) < len(dataset.qa_pairs):
            filtered_count = len(dataset.qa_pairs) - len(filtered_qa)
            print(f"Filtered out {filtered_count} questions from categories")

        return Dataset(
            dataset_name=dataset.dataset_name,
            conversations=dataset.conversations,
            qa_pairs=filtered_qa,
            metadata={**dataset.metadata, "filtered_categories": list(filter_set)},
        )

    def _print_header(self, dataset: Dataset) -> None:
        print(f"\n{'=' * 60}")
        print("📋 Evaluation Pipeline (date-ordered)")
        print(f"{'=' * 60}")
        print(f"📂 Dataset: {dataset.dataset_name}")
        print(f"⚙️  System: {self.adapter.get_system_info()['name']}")
        print(f"{'=' * 60}\n")

    def _load_existing_results(self) -> None:
        """Load existing results from output dir for resume."""
        import json
        from src.models.search import SearchResult, RetrievedMemory
        from src.models.answer import AnswerResult

        def dict_to_search_result(d: dict) -> SearchResult:
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

        def dict_to_answer_result(d: dict) -> AnswerResult:
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

        # Load answer_results.json if exists
        answer_path = self.output_dir / "answer_results.json"
        if answer_path.exists():
            try:
                with open(answer_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    ar_objects = [dict_to_answer_result(d) for d in data]
                    self._results["answer_results"] = ar_objects
                    print(f"  📂 Loaded {len(ar_objects)} existing answer results from {answer_path}")
            except Exception as e:
                print(f"  ⚠️  Failed to load answer_results.json: {e}")

        # Load search_results.json if exists
        search_path = self.output_dir / "search_results.json"
        if search_path.exists():
            try:
                with open(search_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    sr_objects = [dict_to_search_result(d) for d in data]
                    self._results["search_results"] = sr_objects
                    print(f"  📂 Loaded {len(sr_objects)} existing search results from {search_path}")
            except Exception as e:
                print(f"  ⚠️  Failed to load search_results.json: {e}")

        # Load add_latency.json if exists
        add_latency_path = self.output_dir / "add_latency.json"
        if add_latency_path.exists():
            try:
                with open(add_latency_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._results["add_latency"] = data
                    print(f"  📂 Loaded {len(data)} existing add_latency entries from {add_latency_path}")
            except Exception as e:
                print(f"  ⚠️  Failed to load add_latency.json: {e}")

        # Load search_latency.json if exists
        search_latency_path = self.output_dir / "search_latency.json"
        if search_latency_path.exists():
            try:
                with open(search_latency_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._results["search_latency"] = data
                    print(f"  📂 Loaded {len(data)} existing search_latency entries from {search_latency_path}")
            except Exception as e:
                print(f"  ⚠️  Failed to load search_latency.json: {e}")

    def _save_json(self, data: Any, filename: str) -> None:
        import json
        from filelock import FileLock
        filepath = self.output_dir / filename
        lock_path = self.output_dir / f"{filename}.lock"
        lock = FileLock(str(lock_path), timeout=60)
        with lock:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _append_search_results(self, new_results: List["SearchResult"]) -> None:
        """Incrementally append new search results to file (deduplicate by question_id)."""
        import json
        from filelock import FileLock

        filepath = self.output_dir / "search_results.json"
        lock_path = self.output_dir / "search_results.json.lock"

        # Load existing data
        existing_ids = set()
        existing_data = []
        if filepath.exists():
            lock = FileLock(str(lock_path), timeout=60)
            with lock:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_ids = {item["question_id"] for item in existing_data}
                except (json.JSONDecodeError, Exception):
                    existing_data = []

        # Deduplicate and append
        new_data = [self._search_result_to_dict(sr) for sr in new_results]
        for item in new_data:
            if item["question_id"] not in existing_ids:
                existing_data.append(item)
                existing_ids.add(item["question_id"])

        # Save
        lock = FileLock(str(lock_path), timeout=60)
        with lock:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False, default=str)

    def _generate_report(self, elapsed: float) -> None:
        if "eval_result" not in self._results:
            return

        eval_result = self._results["eval_result"]
        stage_timings = self._results.get("stage_timings", {})
        add_search_balance_before = self._results.get("add_search_balance_before")
        add_search_balance_after = self._results.get("add_search_balance_after")

        lines = [
            "=" * 60,
            "📊 Evaluation Report",
            "=" * 60,
            f"System: {self.adapter.get_system_info()['name']}",
            f"Total Time: {elapsed:.2f}s",
            "",
        ]

        # Stage timings
        if stage_timings:
            lines.append("📈 Stage Timings:")
            for stage, info in stage_timings.items():
                stage_name = stage.replace("_", " ").title()
                elapsed = info.get("elapsed", 0)
                count = info.get("count", "")
                count_str = f" ({count} items)" if count else ""
                lines.append(f"  {stage_name}: {elapsed:.2f}s{count_str}")
            lines.append("")

        # Balance change for add+search
        if add_search_balance_before is not None and add_search_balance_after is not None:
            balance_diff = add_search_balance_before - add_search_balance_after
            lines.append(f"💰 ADD+SEARCH Balance Change:")
            lines.append(f"  Before: {add_search_balance_before:.2f} CNY")
            lines.append(f"  After:  {add_search_balance_after:.2f} CNY")
            lines.append(f"  Cost:   {balance_diff:.4f} CNY")
            lines.append("")

        # Evaluation results
        lines.extend([
            f"Total Questions: {eval_result.total_questions}",
            f"Correct: {eval_result.correct}",
            f"Accuracy: {eval_result.accuracy:.2%}",
        ])

        if eval_result.weighted_score is not None:
            lines.append(f"Weighted Score: {eval_result.weighted_score:.2%}")

        report = "\n".join(lines)
        report_path = self.output_dir / "report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n{report}")

    def _search_result_to_dict(self, sr: SearchResult) -> dict:
        return {
            "question_id": sr.question_id,
            "query": sr.query,
            "conversation_id": sr.conversation_id,
            "results": [{"content": r.content, "score": r.score, "metadata": r.metadata} for r in sr.results],
            "retrieval_metadata": sr.retrieval_metadata,
        }

    def _answer_result_to_dict(self, ar: AnswerResult) -> dict:
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

    def _eval_result_to_dict(self, er) -> dict:
        return {
            "total_questions": er.total_questions,
            "correct": er.correct,
            "accuracy": er.accuracy,
            "detailed_results": er.detailed_results,
            "metadata": er.metadata,
        }