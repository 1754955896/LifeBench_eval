"""
EverMemOS Native Adapter for LifeBench_eval - Direct Python import version.

直接导入 EverMemOS 内部模块（stage1-4），不需要 HTTP API。
适配 LifeBench_eval 的数据集格式。

 buffering mode: 累积所有消息，最后批量处理索引和搜索
"""

import asyncio
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

# Add EverMemOS_bz to path for imports
import sys as _sys
import os as _os

# Get the path to EverMemOS_bz root and evaluation
_evermemos_root = Path(__file__).parent.parent.parent / "systems" / "EverMemOS_bz"
_evermemos_src = _evermemos_root / "src"
_evaluation_path = _evermemos_root / "evaluation"

# Add paths in correct order (src first for common_utils, core etc, then evaluation)
if str(_evermemos_src) not in _sys.path:
    _sys.path.insert(0, str(_evermemos_src))
if str(_evermemos_root) not in _sys.path:
    _sys.path.insert(0, str(_evermemos_root))
if str(_evaluation_path) not in _sys.path:
    _sys.path.insert(0, str(_evaluation_path))

# Now we can import from evaluation framework
from evaluation.src.adapters.evermemos import (
    stage1_memcells_extraction,
    stage2_index_building,
    stage3_memory_retrivel,
    stage4_response,
)
from evaluation.src.adapters.evermemos.config import ExperimentConfig
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.event_log_extractor import EventLogExtractor


def _to_iso_format(dt: datetime) -> str:
    """Convert datetime to ISO format string."""
    if dt is None:
        return ""
    return dt.isoformat()


@register_adapter("evermemos_native")
class EverMemOSNativeAdapter(BaseAdapter):
    """
    EverMemOS Native Adapter with buffering support.

    buffering mode:
    - 累积所有消息直到最后一天（2025-12-31）
    - 最后一天触发批量 MemCell 提取和索引构建
    - 搜索请求也会被缓存，等索引构建完成后批量执行

    配置项（在 system yaml 中）：
        buffer_mode: true  # 启用缓冲模式
        final_date: "2025-12-31"  # 最后日期

        llm:
            provider: "openai"
            model: "gpt-4o-mini"
            api_key: "..."
            base_url: "https://openrouter.ai/api/v1"
            temperature: 0.3
            max_tokens: 32768

        add:
            enable_semantic_extraction: false
            enable_clustering: true
            enable_profile_extraction: false

        search:
            mode: "agentic"  # agentic | lightweight
            lightweight_search_mode: "bm25_only"  # bm25_only | hybrid | emb_only
            use_hybrid_search: true
            use_reranker: true

        answer:
            temperature: 0
            response_top_k: 10
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else None
        self.stats_collector = stats_collector

        # Buffering configuration
        self.buffer_mode = config.get("buffer_mode", True)
        self.final_date = config.get("final_date", "2025-12-31")

        # Initialize LLM Provider
        llm_config = config.get("llm", {})
        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", "") or _os.getenv("LLM_API_KEY", ""),
            base_url=llm_config.get("base_url", "https://openrouter.ai/api/v1"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 32768),
        )

        # Initialize Event Log Extractor
        # use_eval_prompts=False: 走生产 prompt(受 MEMORY_LANGUAGE 控制),
        # 与 HTTP 版一致, 中文数据输出中文事件日志
        self.event_log_extractor = EventLogExtractor(
            llm_provider=self.llm_provider,
            use_eval_prompts=False,
        )

        # Ensure NLTK data is available
        stage2_index_building.ensure_nltk_data()

        # Per-conversation buffer storage
        # Structure: {
        #   "conv_id": {
        #       "messages": [...],
        #       "searches": [...],
        #       "index_built": False,
        #       "index_metadata": {}
        #   }
        # }
        self._conv_buffers: Dict[str, Dict[str, Any]] = {}
        # Track latest date seen per conversation (for auto-detecting final_date when dataset has no configured final_date)
        self._conv_latest_dates: Dict[str, str] = {}

        # Mapping from conv_id to numeric index for EverMemOS file naming
        # EverMemOS expects files named conv_0, conv_1, conv_2, etc.
        self._conv_id_to_index: Dict[str, int] = {}
        self._next_conv_index = 0

        # 续跑场景: 从 conv_index_map.json 恢复 conv_id → index 映射,
        # 保证与已落盘的 memcell_list_conv_N.json 等文件对应, 避免重跑提取
        if self.output_dir:
            map_file = self.output_dir / "conv_index_map.json"
            if map_file.exists():
                try:
                    saved = json.loads(map_file.read_text(encoding="utf-8"))
                    self._conv_id_to_index = {k: int(v) for k, v in saved.items()}
                    self._next_conv_index = max(self._conv_id_to_index.values()) + 1 if self._conv_id_to_index else 0
                    print(f"   📂 Loaded conv index map: {len(self._conv_id_to_index)} conversations")
                except Exception as e:
                    print(f"   ⚠️ Failed to load conv index map: {e}")

        # Per-conversation build lock: the runner searches all final-date QAs
        # concurrently, which would otherwise trigger N duplicate index builds
        # (each re-running the full Stage 1 extraction).
        self._conv_build_locks: Dict[str, asyncio.Lock] = {}

        # Loaded index cache (bm25/embedding pkl): avoids re-loading ~310MB per
        # search, which blows up memory under concurrent searches.
        self._index_cache: Dict[str, Any] = {}

        # Store config for later use
        self._config = config

        print(f"✅ EverMemOS Native Adapter initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")
        print(f"   Buffer Mode: {self.buffer_mode}, Final Date: {self.final_date}")

    def _get_numeric_index(self, conv_id: str) -> int:
        """Get or create a numeric index for a conversation ID.

        EverMemOS file naming expects conv_0, conv_1, conv_2, etc.
        """
        if conv_id not in self._conv_id_to_index:
            self._conv_id_to_index[conv_id] = self._next_conv_index
            self._next_conv_index += 1
            self._save_conv_index_map()
        return self._conv_id_to_index[conv_id]

    def _save_conv_index_map(self) -> None:
        """Persist conv_id → index mapping for resume-safe restarts."""
        if not self.output_dir:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            map_file = self.output_dir / "conv_index_map.json"
            map_file.write_text(
                json.dumps(self._conv_id_to_index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"   ⚠️ Failed to save conv index map: {e}")

    def _get_conv_numeric_index(self, conv_id: str) -> int:
        """Get the numeric index for an existing conversation ID."""
        return self._conv_id_to_index.get(conv_id, -1)

    def _load_index_file(self, path: str, cache_key: str) -> Any:
        """Load a pickle index file, caching it to avoid repeated 310MB loads."""
        if cache_key not in self._index_cache:
            with open(path, "rb") as f:
                self._index_cache[cache_key] = pickle.load(f)
        return self._index_cache[cache_key]

    def _convert_config_to_experiment_config(self) -> ExperimentConfig:
        """Convert YAML config to ExperimentConfig."""
        exp_config = ExperimentConfig()
        config = self._config

        # LLM config
        llm_cfg = config.get("llm", {})
        provider = llm_cfg.get("provider", "openai")

        exp_config.llm_service = provider
        exp_config.llm_config = {
            provider: {
                "llm_provider": provider,
                "model": llm_cfg.get("model", "gpt-4o-mini"),
                "api_key": llm_cfg.get("api_key") or _os.getenv("LLM_API_KEY", ""),
                "base_url": llm_cfg.get("base_url", "https://openrouter.ai/api/v1"),
                "temperature": llm_cfg.get("temperature", 0.3),
                "max_tokens": llm_cfg.get("max_tokens", 32768),
            }
        }

        # Add stage config
        add_config = config.get("add", {})
        if "enable_semantic_extraction" in add_config:
            exp_config.enable_semantic_extraction = add_config["enable_semantic_extraction"]
        if "enable_clustering" in add_config:
            exp_config.enable_clustering = add_config["enable_clustering"]
        if "enable_profile_extraction" in add_config:
            exp_config.enable_profile_extraction = add_config["enable_profile_extraction"]

        # Search stage config
        search_config = config.get("search", {})
        if "mode" in search_config:
            exp_config.retrieval_mode = search_config["mode"]
            exp_config.use_agentic_retrieval = exp_config.retrieval_mode == "agentic"
        if "lightweight_search_mode" in search_config:
            exp_config.lightweight_search_mode = search_config["lightweight_search_mode"]
        if "use_hybrid_search" in search_config:
            exp_config.use_hybrid_search = search_config["use_hybrid_search"]
        if "use_reranker" in search_config:
            exp_config.use_reranker = search_config["use_reranker"]
        if "hybrid_emb_candidates" in search_config:
            exp_config.hybrid_emb_candidates = search_config["hybrid_emb_candidates"]
        if "hybrid_bm25_candidates" in search_config:
            exp_config.hybrid_bm25_candidates = search_config["hybrid_bm25_candidates"]
        if "hybrid_rrf_k" in search_config:
            exp_config.hybrid_rrf_k = search_config["hybrid_rrf_k"]

        # Answer stage config
        answer_config = config.get("answer", {})
        if "response_top_k" in answer_config:
            exp_config.response_top_k = answer_config["response_top_k"]

        return exp_config

    def _convert_message_to_raw(self, msg: Any, idx: int) -> Dict[str, Any]:
        """Convert a single Message to EverMemOS raw data format."""
        # Handle timestamp
        if msg.timestamp is not None:
            timestamp_str = _to_iso_format(msg.timestamp)
        else:
            # Generate pseudo timestamp using message index
            base_time = datetime(2023, 1, 1, 0, 0, 0)
            pseudo_time = base_time + timedelta(seconds=idx * 30)
            timestamp_str = _to_iso_format(pseudo_time)

        speaker_id = getattr(msg, "speaker_id", None) or msg.speaker_name or "unknown"
        message_dict = {
            "speaker_id": speaker_id,
            "user_name": speaker_id,
            "speaker_name": msg.speaker_name or speaker_id,
            "content": msg.content,
            "timestamp": timestamp_str,
        }

        # Add optional fields from metadata
        for optional_field in ["img_url", "blip_caption", "query"]:
            if hasattr(msg, "metadata") and msg.metadata.get(optional_field):
                message_dict[optional_field] = msg.metadata[optional_field]

        return message_dict

    def _get_conv_buffer(self, conv_id: str) -> Dict[str, Any]:
        """Get or create a buffer for a specific conversation."""
        if conv_id not in self._conv_buffers:
            self._conv_buffers[conv_id] = {
                "messages": [],
                "searches": [],
                "index_built": False,
                "index_metadata": {},
            }
        return self._conv_buffers[conv_id]

    def _buffer_chunk(self, chunk: ChunkedMessage) -> None:
        """Buffer a chunk for later processing."""
        conv_id = chunk.conversation_id
        buf = self._get_conv_buffer(conv_id)

        for msg in chunk.messages:
            raw_msg = self._convert_message_to_raw(msg, len(buf["messages"]))
            buf["messages"].append(raw_msg)

    def _is_final_date(self, chunk: ChunkedMessage) -> bool:
        """Check if this chunk belongs to the final date.

        Checks dia_id date (e.g., "2025-12-31_fitness_health0") against final_date.

        Args:
            chunk: ChunkedMessage to check

        Returns:
            True if any message in chunk has dia_id containing final_date
        """
        final = self.final_date  # e.g., "2025-12-31"

        for msg in getattr(chunk, 'messages', []):
            # dia_id is stored in msg.metadata["dia_id"]
            dia_id = ""
            if hasattr(msg, 'metadata') and msg.metadata.get('dia_id'):
                dia_id = msg.metadata.get('dia_id', "")
            # dia_id format: "2025-12-31_fitness_health0"
            if dia_id.startswith(final):
                return True

        return False

    async def _build_index_locked(self, conv_id: str) -> Dict[str, Any]:
        """Build index under a per-conversation lock.

        The runner searches all final-date QAs concurrently; without the lock
        each search would trigger a duplicate full Stage 1 extraction.
        """
        lock = self._conv_build_locks.setdefault(conv_id, asyncio.Lock())
        async with lock:
            buf = self._get_conv_buffer(conv_id)
            if buf["index_built"]:
                return buf["index_metadata"]
            return await self._build_index_for_conv(conv_id)

    def _is_final_qa(self, ask_time: str) -> bool:
        """Check if this QA belongs to the final date.

        Args:
            ask_time: QA ask_time string (e.g., "2025-12-31")

        Returns:
            True if ask_time equals final_date
        """
        return ask_time == self.final_date

    async def _build_index_for_conv(self, conv_id: str) -> Dict[str, Any]:
        """Build the index for a specific conversation."""
        if not self.output_dir:
            raise ValueError("output_dir is required for EverMemOS Native Adapter")

        buf = self._get_conv_buffer(conv_id)
        if buf["index_built"]:
            return buf["index_metadata"]

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        memcells_dir = output_dir / "memcells"
        memcells_dir.mkdir(parents=True, exist_ok=True)
        bm25_index_dir = output_dir / "bm25_index"
        emb_index_dir = output_dir / "vectors"
        bm25_index_dir.mkdir(parents=True, exist_ok=True)
        emb_index_dir.mkdir(parents=True, exist_ok=True)

        raw_data = buf["messages"]

        exp_config = self._convert_config_to_experiment_config()

        # Get numeric index for this conversation
        # EverMemOS expects files named conv_0, conv_1, conv_2, etc.
        numeric_index = self._get_numeric_index(conv_id)

        memcell_file = memcells_dir / f"memcell_list_conv_{numeric_index}.json"
        if memcell_file.exists():
            # 续跑场景: memcells 已完整落盘, 直接复用; 缓冲区在续跑时可能不全,
            # 若重跑 Stage 1 会用不完整消息覆盖完整索引
            print(f"\n📂 MemCell list already exists ({memcell_file.name}), "
                  f"skipping Stage 1 extraction (reusing persisted memcells)")
            memcells = []
        elif not raw_data:
            return {}
        else:
            print(f"\n{'='*60}")
            print(f"Stage 1: MemCell Extraction (conv: {conv_id})")
            print(f"{'='*60}")
            print(f"  Processing conversation: {conv_id} (numeric_index: {numeric_index})")

            memcells = await stage1_memcells_extraction.process_single_conversation(
                conv_id=str(numeric_index),
                conversation=raw_data,
                save_dir=str(memcells_dir),
                llm_provider=self.llm_provider,
                event_log_extractor=self.event_log_extractor,
                progress_counter=None,
                progress=None,
                task_id=None,
                config=exp_config,
            )

        print(f"\n{'='*60}")
        print(f"Stage 2: Index Building (conv: {conv_id}, index: {numeric_index})")
        print(f"{'='*60}")

        # Build BM25 index - only for this conversation if not already built
        # Check if BM25 index already exists for this conversation
        existing_bm25 = bm25_index_dir / f"bm25_index_conv_{numeric_index}.pkl"
        if existing_bm25.exists():
            print(f"BM25 index already exists for conv {numeric_index}, skipping...")
        else:
            exp_config.num_conv = self._next_conv_index
            print(f"Building BM25 index (num_conv={exp_config.num_conv})...")
            stage2_index_building.build_bm25_index(
                config=exp_config,
                data_dir=memcells_dir,
                bm25_save_dir=bm25_index_dir,
            )

        # Build Embedding index - only for this conversation if not already built
        use_hybrid = exp_config.use_hybrid_search
        if use_hybrid:
            existing_emb = emb_index_dir / f"embedding_index_conv_{numeric_index}.pkl"
            if existing_emb.exists():
                print(f"Embedding index already exists for conv {numeric_index}, skipping...")
            else:
                print(f"Building Embedding index...")
                await stage2_index_building.build_emb_index(
                    config=exp_config,
                    data_dir=memcells_dir,
                    emb_save_dir=emb_index_dir,
                )

        # Store index metadata for this conversation
        buf["index_metadata"] = {
            "type": "lazy_load",
            "memcells_dir": str(memcells_dir),
            "bm25_index_dir": str(bm25_index_dir),
            "emb_index_dir": str(emb_index_dir),
            "conversation_ids": [conv_id],
            "use_hybrid_search": use_hybrid,
            "total_conversations": 1,
        }

        print(f"\n✅ Index built for conv {conv_id}")
        print(f"   MemCells: {len(memcells) if memcells else 'reused from file'}")

        buf["index_built"] = True
        return buf["index_metadata"]

    async def _execute_buffered_searches_for_conv(self, conv_id: str) -> List[SearchResult]:
        """Execute buffered searches for a specific conversation."""
        buf = self._get_conv_buffer(conv_id)
        if not buf["searches"]:
            return []

        print(f"\n{'='*60}")
        print(f"Executing {len(buf['searches'])} buffered searches for conv {conv_id}...")
        print(f"{'='*60}")

        results = []
        for search_req in buf["searches"]:
            result = await self._do_search(
                query=search_req["query"],
                conversation_id=search_req["conversation_id"],
                index=buf["index_metadata"],
                **search_req.get("kwargs", {})
            )
            # Save buffered search result directly
            self._save_search_result(result)
            results.append(result)

        # Clear buffered searches after execution
        buf["searches"] = []

        print(f"✅ Completed {len(results)} searches for conv {conv_id}")
        return results

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """
        Add stage: Buffer messages per conversation, build index when final date reached.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Index metadata dict
        """
        if not self.output_dir:
            raise ValueError("output_dir is required for EverMemOS Native Adapter")

        # Process each chunk - buffer messages per conversation
        for chunk in chunks:
            if not chunk.messages:
                continue

            conv_id = chunk.conversation_id
            buf = self._get_conv_buffer(conv_id)

            # Buffer the messages
            for msg in chunk.messages:
                raw_msg = self._convert_message_to_raw(msg, len(buf["messages"]))
                buf["messages"].append(raw_msg)
                # Track latest date for auto-detecting final_date fallback
                ts = raw_msg.get("timestamp", "")
                if ts and len(ts) >= 10:
                    msg_date = ts[:10]
                    if conv_id not in self._conv_latest_dates or msg_date > self._conv_latest_dates[conv_id]:
                        self._conv_latest_dates[conv_id] = msg_date

            # Check if this conversation reached final date
            if self.buffer_mode and self._is_final_date(chunk):
                print(f"\n🎯 Conversation {conv_id} reached final date ({self.final_date}), building index...")
                try:
                    await self._build_index_locked(conv_id)
                    # Execute buffered searches for this conversation
                    buf = self._get_conv_buffer(conv_id)
                    if buf["searches"]:
                        print(f"   Executing {len(buf['searches'])} buffered searches...")
                        await self._execute_buffered_searches_for_conv(conv_id)
                    else:
                        print(f"   No buffered searches to execute")
                except Exception as e:
                    print(f"   ERROR during index build or search execution: {e}")
                    import traceback
                    traceback.print_exc()

        # If buffering is disabled, build all indexes immediately
        if not self.buffer_mode:
            for conv_id in self._conv_buffers:
                await self._build_index_locked(conv_id)
            return {}

        # Check if this was the last call (force build all)
        if kwargs.get("_is_last_call", False):
            for conv_id in self._conv_buffers:
                buf = self._get_conv_buffer(conv_id)
                if not buf["index_built"]:
                    print(f"\n🔨 Last call - building index for conv {conv_id}...")
                    await self._build_index_locked(conv_id)
                    await self._execute_buffered_searches_for_conv(conv_id)
            return {}

        # In buffer mode, return pending status
        total_buffered = sum(len(buf["messages"]) for buf in self._conv_buffers.values())
        print(f"\n📦 Buffered {total_buffered} messages from {len(self._conv_buffers)} conversations")
        print(f"   Waiting for final date ({self.final_date}) per conversation...")

        return {
            "type": "evermemos_native",
            "status": "buffered",
            "buffered_messages": total_buffered,
            "buffered_conversations": len(self._conv_buffers),
            "waiting_for_final_date": self.final_date,
        }

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """
        Search stage: Buffer searches per conversation, build index when needed.

        Logic:
        1. If NOT final_date: buffer the search, return empty result
        2. If final_date:
           - If no index: build index
           - Execute ALL buffered searches + current search
           - Return current search result (buffered ones are saved via _save_search_result)

        Args:
            query: Query text
            conversation_id: Conversation ID
            index: Index metadata (from add stage)
            **kwargs: Extra parameters (e.g., top_k, ask_time)

        Returns:
            SearchResult with retrieved memories
        """
        buf = self._get_conv_buffer(conversation_id)

        # Try to extract ask_time from kwargs or parse from query text
        # Query format: "（提问时间：2025-12-31）..."
        ask_time = kwargs.get("ask_time", "")
        if not ask_time:
            import re
            match = re.search(r"（提问时间：(\d{4}-\d{2}-\d{2})）", query)
            if match:
                ask_time = match.group(1)

        is_final_date = self._is_final_qa(ask_time) if ask_time else False

        # Fallback: if configured final_date was never found in data,
        # auto-detect from latest buffered message date per conversation.
        if not is_final_date and self.buffer_mode and not buf["index_built"]:
            # Prefer ask_time-based detection (requires ask_time from metadata)
            latest = self._conv_latest_dates.get(conversation_id, "")
            if latest and ask_time and ask_time >= latest:
                print(f"\n🎯 Auto-detected final date ({latest}) for conv {conversation_id} "
                      f"(configured final_date={self.final_date} not found in data)")
                is_final_date = True
            # Fallback: if ask_time is unavailable (e.g. locomo has no zh timestamp prefix
            # and parser/kwargs don't provide it), trigger on first search if messages
            # are already buffered.  By the time search runs, all add_chunks calls for
            # this conversation have completed (runner processes add → search per date).
            elif latest and buf["messages"]:
                print(f"\n🎯 Auto-triggering index build for conv {conversation_id} "
                      f"(final_date={self.final_date} not in data, ask_time unavailable)")
                is_final_date = True

        # If index already built, execute search directly (don't buffer subsequent searches)
        if buf["index_built"]:
            is_final_date = True

        # Case 1: NOT final_date - buffer and return empty
        if not is_final_date:
            print(f"\n📦 Buffering search: {query[:50]}... (conv {conversation_id}, waiting for {self.final_date})")
            buf["searches"].append({
                "query": query,
                "conversation_id": conversation_id,
                "kwargs": kwargs,
            })
            # Return empty result - will be overwritten when final_date executes this search
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"status": "buffered", "waiting_for_final_date": self.final_date},
            )

        # Case 2: IS final_date - execute all searches
        print(f"\n🎯 Final date ({self.final_date}) reached for conv {conversation_id}")

        # Build index if needed
        if not buf["index_built"]:
            print(f"   Building index...")
            try:
                await self._build_index_locked(conversation_id)
            except Exception as e:
                print(f"   ERROR building index: {e}")
                import traceback
                traceback.print_exc()
                return SearchResult(
                    question_id=kwargs.get("question_id", ""),
                    query=query,
                    conversation_id=conversation_id,
                    results=[],
                    retrieval_metadata={"error": f"Index build failed: {str(e)}"},
                )
        else:
            print(f"   Index already built")

        # Execute buffered searches first (save each result)
        if buf["searches"]:
            print(f"   Executing {len(buf['searches'])} buffered searches...")
            for search_req in buf["searches"]:
                sr = await self._do_search(
                    query=search_req["query"],
                    conversation_id=search_req["conversation_id"],
                    index=buf["index_metadata"],
                    **search_req.get("kwargs", {})
                )
                # Save buffered search result directly
                self._save_search_result(sr)
            buf["searches"] = []
            print(f"   Buffered searches completed and saved")

        # Execute current search and return
        print(f"   Executing current search...")
        return await self._do_search(query, conversation_id, buf["index_metadata"], **kwargs)

    def _save_search_result(self, sr: SearchResult) -> None:
        """Save a single search result to its own file.

        Files are saved as individual JSON files named: results/{question_id}.json
        Pipeline will load these files to build the final search_results.json.
        """
        if not self.output_dir:
            return

        results_dir = Path(self.output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize question_id for use as filename
        safe_qid = sr.question_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_path = results_dir / f"{safe_qid}.json"

        entry = {
            "question_id": sr.question_id,
            "query": sr.query,
            "conversation_id": sr.conversation_id,
            "results": [
                {"content": r.content, "score": r.score, "metadata": r.metadata}
                for r in sr.results
            ],
            "retrieval_metadata": sr.retrieval_metadata,
        }

        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        print(f"   [DEBUG _save_search_result] Saved {sr.question_id} to {output_path.name}, results_count={len(sr.results)}")

    async def _do_search(
        self, query: str, conversation_id: str, index: Dict[str, Any], **kwargs
    ) -> SearchResult:
        """Perform the actual search operation."""
        if not index or index.get("type") != "lazy_load":
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": "Index not built or invalid"},
            )

        bm25_index_dir = Path(index["bm25_index_dir"])
        emb_index_dir = Path(index["emb_index_dir"])

        # Get numeric index for this conversation
        numeric_index = self._get_conv_numeric_index(conversation_id)
        if numeric_index < 0:
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": f"No numeric index for conv_id: {conversation_id}"},
            )

        # Load BM25 index using numeric index (cached: 每次搜索重新加载 310MB+
        # embedding pkl 在并发搜索时会内存爆炸)
        bm25_file = bm25_index_dir / f"bm25_index_conv_{numeric_index}.pkl"
        if not bm25_file.exists():
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": f"BM25 index not found: {bm25_file.name}"},
            )

        bm25_index_data = self._load_index_file(str(bm25_file), f"bm25_{numeric_index}")
        bm25 = bm25_index_data.get("bm25")
        docs = bm25_index_data.get("docs")

        # Load Embedding index using numeric index (cached)
        emb_index = None
        if index.get("use_hybrid_search"):
            emb_file = emb_index_dir / f"embedding_index_conv_{numeric_index}.pkl"
            if emb_file.exists():
                emb_index = self._load_index_file(str(emb_file), f"emb_{numeric_index}")

        # Get config
        exp_config = self._convert_config_to_experiment_config()
        llm_config = exp_config.llm_config.get(exp_config.llm_service, {})
        retrieval_mode = exp_config.retrieval_mode

        # Perform retrieval
        if retrieval_mode == "agentic":
            top_results, metadata = await stage3_memory_retrivel.agentic_retrieval(
                query=query,
                config=exp_config,
                llm_provider=self.llm_provider,
                llm_config=llm_config,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
            )
        elif retrieval_mode == "lightweight":
            lightweight_mode = getattr(exp_config, "lightweight_search_mode", "bm25_only")
            if lightweight_mode == "hybrid":
                top_results = await stage3_memory_retrivel.hybrid_search_with_rrf(
                    query=query,
                    emb_index=emb_index,
                    bm25=bm25,
                    docs=docs,
                    top_n=exp_config.lightweight_final_top_n,
                    emb_candidates=exp_config.lightweight_emb_top_n,
                    bm25_candidates=exp_config.lightweight_bm25_top_n,
                    rrf_k=exp_config.hybrid_rrf_k,
                )
                metadata = {}
            elif lightweight_mode == "emb_only":
                if emb_index is not None:
                    top_results = await stage3_memory_retrivel.embedding_search(
                        query=query,
                        emb_index=emb_index,
                        docs=docs,
                        top_n=exp_config.lightweight_final_top_n,
                    )
                else:
                    top_results = []
                metadata = {}
            else:  # bm25_only
                top_results = stage3_memory_retrivel.bm25_search(
                    query=query,
                    bm25=bm25,
                    docs=docs,
                    top_n=exp_config.lightweight_final_top_n,
                )
                metadata = {}
        else:
            # Default hybrid
            top_results = await stage3_memory_retrivel.hybrid_search_with_rrf(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=20,
                emb_candidates=exp_config.hybrid_emb_candidates,
                bm25_candidates=exp_config.hybrid_bm25_candidates,
                rrf_k=exp_config.hybrid_rrf_k,
            )
            metadata = {}

        # Convert to SearchResult format
        response_top_k = getattr(exp_config, "response_top_k", 10)
        retrieved = []
        for doc, score in top_results[:response_top_k]:
            content = doc.get("episode", "") or doc.get("summary", "") or doc.get("content", "")
            retrieved.append(RetrievedMemory(
                content=content,
                score=float(score),
                metadata={
                    "subject": doc.get("subject", ""),
                    "summary": doc.get("summary", ""),
                },
            ))

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=retrieved,
            retrieval_metadata={
                "adapter": "evermemos_native",
                "mode": retrieval_mode,
                "total_results": len(retrieved),
                "metadata": metadata,
            },
        )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """
        Answer stage: Generate answer using LLM.

        Args:
            query: Question text
            context: Formatted retrieved context
            conversation_id: Conversation ID
            **kwargs: Extra parameters

        Returns:
            Generated answer string
        """
        exp_config = self._convert_config_to_experiment_config()

        answer = await stage4_response.locomo_response(
            llm_provider=self.llm_provider,
            context=context,
            question=query,
            experiment_config=exp_config,
        )

        return answer

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "EverMemOS",
            "version": "native",
            "description": "EverMemOS memory system via direct Python import (buffering mode)",
        }

    async def close(self) -> None:
        """Cleanup resources."""
        pass
