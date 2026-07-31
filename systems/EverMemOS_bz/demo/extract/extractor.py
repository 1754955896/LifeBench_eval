"""记忆提取核心逻辑

使用 V3 API 进行记忆提取。
"""

from typing import Dict, Any, List
from pathlib import Path

from agentic_layer.memory_manager import MemoryManager
from api_specs.dtos.memory_command import MemorizeRequest
from api_specs.memory_types import RawDataType
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from common_utils.datetime_utils import from_iso_format

from demo.config import ExtractModeConfig, MongoDBConfig
from demo.utils import ensure_mongo_beanie_ready


class MemoryExtractor:
    """记忆提取器 - 使用 V3 API"""

    def __init__(self, config: ExtractModeConfig, mongo_config: MongoDBConfig):
        """初始化提取器

        Args:
            config: 提取配置
            mongo_config: MongoDB 配置
        """
        self.config = config
        self.mongo_config = mongo_config
        self.manager: MemoryManager | None = None

    async def initialize(self) -> None:
        """初始化 MongoDB 和 MemoryManager"""
        await ensure_mongo_beanie_ready(self.mongo_config)
        self.manager = MemoryManager()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_message(entry: Dict[str, Any]) -> Dict[str, Any] | None:
        """归一化消息格式

        Args:
            entry: 原始消息字典

        Returns:
            归一化后的消息字典，如果必填字段缺失则返回 None
        """
        # 提取时间戳
        timestamp = (
            entry.get("create_time")
            or entry.get("createTime")
            or entry.get("timestamp")
            or entry.get("created_at")
        )
        if not timestamp:
            return None

        if isinstance(timestamp, str):
            try:
                timestamp_dt = from_iso_format(timestamp)
            except Exception:
                return None
        else:
            return None

        # 提取发言人名称
        speaker_name = entry.get("sender_name") or entry.get("sender")
        if not speaker_name:
            origin = entry.get("origin")
            if isinstance(origin, dict):
                speaker_name = origin.get("fullName") or origin.get("full_name")
        if not speaker_name:
            return None

        # 提取发言人 ID
        raw_speaker_id = None
        origin = entry.get("origin")
        if isinstance(origin, dict):
            raw_speaker_id = origin.get("createBy") or origin.get("create_by")
        if not raw_speaker_id:
            raw_speaker_id = entry.get("sender_id") or entry.get("sender")

        return {
            "speaker_id": str(raw_speaker_id or speaker_name),
            "speaker_name": str(speaker_name),
            "content": str(entry.get("content", "")),
            "timestamp": timestamp_dt,
        }

    async def extract_from_events(self, events: List[Dict[str, Any]]) -> int:
        """从事件列表中提取记忆

        Args:
            events: 对话事件列表

        Returns:
            提取的 MemCell 数量
        """
        if not self.manager:
            raise RuntimeError("请先调用 initialize() 初始化提取器")

        print("=" * 80)
        print("使用 V3 API 提取记忆")
        print("=" * 80)
        print(f"\n✓ 场景类型: {self.config.scenario_type.value}")
        print(f"✓ 语言: {self.config.prompt_language}")
        print(f"✓ 群组 ID: {self.config.group_id}")
        print(f"✓ 语义提取: {self.config.enable_semantic_extraction}")
        print(f"\n开始处理 {len(events)} 条消息...\n")

        history: List[RawData] = []
        saved_count = 0

        for idx, entry in enumerate(events):
            # 归一化消息
            message_payload = self.normalize_message(entry)
            if not message_payload:
                continue

            # 提取消息 ID
            message_id = (
                entry.get("message_id")
                or entry.get("id")
                or entry.get("uuid")
                or entry.get("event_id")
                or f"msg_{idx}"
            )

            # 创建 RawData
            raw_item = RawData(
                content=message_payload,
                data_id=str(message_id),
                data_type=RawDataType.CONVERSATION,
            )

            # 初始化历史
            if not history:
                history.append(raw_item)
                continue

            # 构建请求
            request = MemorizeRequest(
                history_raw_data_list=list(history),
                new_raw_data_list=[raw_item],
                raw_data_type=RawDataType.CONVERSATION,
                user_id_list=["default"],
                group_id=self.config.group_id,
                group_name=self.config.group_name,
                enable_semantic_extraction=self.config.enable_semantic_extraction
                or False,
                enable_event_log_extraction=True,
            )

            # 调用 V3 API
            try:
                result = await self.manager.memorize(request)

                if result:
                    saved_count += 1
                    print(
                        f"  [{saved_count:3d}] ✅ 提取成功，返回 {len(result)} 个 Memory"
                    )
                    history = [raw_item]
                else:
                    history.append(raw_item)
                    if len(history) > self.config.history_window_size:
                        history = history[-self.config.history_window_size :]

            except Exception as e:
                print(f"\n⚠️ 提取失败: {e}")
                history.append(raw_item)
                if len(history) > self.config.history_window_size:
                    history = history[-self.config.history_window_size :]
                continue

        print(f"\n✅ 处理完成，共提取 {saved_count} 个 MemCell")
        return saved_count
