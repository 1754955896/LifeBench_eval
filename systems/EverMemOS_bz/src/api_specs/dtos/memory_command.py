from dataclasses import dataclass
import datetime
from typing import List, Optional

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from api_specs.memory_types import RawDataType
import re

from bson import ObjectId


iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'


@dataclass
class RawData:  # Memcell 这个其实偏向输入，这个在输入的上层；memcell表的那个是存储的结构，那个更加底层
    """Raw data structure for storing original content."""

    content: dict[str, Any]
    data_id: str
    data_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def _serialize_value(self, value: Any) -> Any:
        """
        递归序列化值，处理datetime、ObjectId等特殊类型

        Args:
            value: 待序列化的值

        Returns:
            可JSON序列化的值
        """
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, ObjectId):
            # 将 ObjectId 序列化为字符串
            return str(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif hasattr(value, '__dict__'):
            # 处理自定义对象，转换为字典
            return self._serialize_value(value.__dict__)
        else:
            return value

    def _deserialize_value(self, value: Any, field_name: str = "") -> Any:
        """
        递归反序列化值，基于字段名启发式判断是否需要恢复datetime类型

        Args:
            value: 待反序列化的值
            field_name: 字段名称，用于启发式判断

        Returns:
            反序列化后的值
        """
        if isinstance(value, str):
            # 基于字段名启发式判断是否为时间字段
            if self._is_datetime_field(field_name) and self._is_iso_datetime(value):
                try:
                    from common_utils.datetime_utils import from_iso_format

                    return from_iso_format(value)
                except (ValueError, ImportError):
                    return value
            return value
        elif isinstance(value, dict):
            return {k: self._deserialize_value(v, k) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._deserialize_value(item, field_name) for item in value]
        else:
            return value

    def _is_datetime_field(self, field_name: str) -> bool:
        """
        基于字段名启发式判断是否为时间字段

        Args:
            field_name: 字段名称

        Returns:
            bool: 是否为时间字段
        """
        if not isinstance(field_name, str):
            return False

        # 精确匹配的时间字段名（基于项目中实际使用的字段名）
        exact_datetime_fields = {
            'timestamp',
            'createTime',
            'updateTime',
            'create_time',
            'update_time',
            'sent_timestamp',
            'received_timestamp',
            'create_timestamp',
            'last_update_timestamp',
            'modify_timestamp',
            'readUpdateTime',
            'created_at',
            'updated_at',
            'joinTime',
            'leaveTime',
            'lastOnlineTime',
            'sync_time',
            'processed_at',
            'start_time',
            'end_time',
            'event_time',
            'build_timestamp',
            'datetime',
            'created',
            'updated',  # 添加常见的时间字段变体
        }

        field_lower = field_name.lower()

        # 精确匹配检查
        if field_name in exact_datetime_fields or field_lower in exact_datetime_fields:
            return True

        # 排除不应该被识别为时间字段的常见词汇
        exclusions = {
            'runtime',
            'timeout',
            'timeline',
            'timestamp_format',
            'time_zone',
            'time_limit',
            'timestamp_count',
            'timestamp_enabled',
            'time_sync',
            'playtime',
            'lifetime',
            'uptime',
            'downtime',
        }

        if field_name in exclusions or field_lower in exclusions:
            return False

        # 后缀匹配检查（更严格的规则）
        time_suffixes = ['_time', '_timestamp', '_at', '_date']
        for suffix in time_suffixes:
            if field_name.endswith(suffix) or field_lower.endswith(suffix):
                return True

        # 前缀匹配检查（更严格的规则）
        if field_name.endswith('Time') and not field_name.endswith('runtime'):
            # 匹配 xxxTime 模式，但排除 runtime
            return True

        if field_name.endswith('Timestamp'):
            # 匹配 xxxTimestamp 模式
            return True

        return False

    def _is_iso_datetime(self, value: str) -> bool:
        """
        检查字符串是否为ISO格式的datetime

        Args:
            value: 字符串值

        Returns:
            bool: 是否为ISO datetime格式
        """
        # 简单的ISO datetime格式检查
        if not isinstance(value, str) or len(value) < 19:
            return False

        # 检查基本的ISO格式模式：YYYY-MM-DDTHH:MM:SS
        return bool(re.match(iso_pattern, value))

    def to_json(self) -> str:
        """
        将RawData对象序列化为JSON字符串

        Returns:
            str: JSON字符串
        """
        try:
            data = {
                'content': self._serialize_value(self.content),
                'data_id': self.data_id,
                'data_type': self.data_type,
                'metadata': (
                    self._serialize_value(self.metadata) if self.metadata else None
                ),
            }
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            raise ValueError(f"无法序列化RawData为JSON: {e}") from e

    @classmethod
    def from_json_str(cls, json_str: str) -> 'RawData':
        """
        从JSON字符串反序列化为RawData对象

        Args:
            json_str: JSON字符串

        Returns:
            RawData: 反序列化后的RawData对象

        Raises:
            ValueError: JSON格式错误或缺少必需字段
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("JSON必须是一个对象")

        # 检查必需字段
        if 'content' not in data or 'data_id' not in data:
            raise ValueError("JSON缺少必需字段: content 和 data_id")

        # 创建实例并反序列化值
        instance = cls.__new__(cls)
        instance.content = instance._deserialize_value(data['content'], 'content')
        instance.data_id = data['data_id']
        instance.data_type = data.get('data_type')
        instance.metadata = (
            instance._deserialize_value(data.get('metadata'), 'metadata')
            if data.get('metadata')
            else None
        )

        return instance


@dataclass
class MemorizeRequest:
    history_raw_data_list: list[RawData]
    new_raw_data_list: list[RawData]
    raw_data_type: RawDataType
    # 整个group全量的user_id列表
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    current_time: Optional[datetime] = None
    # 可选的提取控制参数
    enable_semantic_extraction: bool = True  # 是否提取语义记忆
    enable_event_log_extraction: bool = True  # 是否提取事件日志


@dataclass
class MemorizeOfflineRequest:
    memorize_from: datetime
    memorize_to: datetime
