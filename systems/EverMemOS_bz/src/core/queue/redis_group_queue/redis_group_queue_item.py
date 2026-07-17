"""
Redis分组队列项接口

定义队列中存储项目的标准接口，支持JSON和BSON序列化和反序列化。
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict
from enum import Enum
import bson


class SerializationMode(Enum):
    """序列化模式枚举"""

    JSON = "json"  # JSON字符串序列化
    BSON = "bson"  # BSON字节序列化


class RedisGroupQueueItem(ABC):
    """Redis分组队列项接口"""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        将对象转换为字典

        Returns:
            Dict[str, Any]: 对象的字典表示
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json_str(cls, json_str: str) -> 'RedisGroupQueueItem':
        """
        从JSON字符串创建对象实例

        Args:
            json_str: JSON字符串

        Returns:
            RedisGroupQueueItem: 对象实例

        Raises:
            ValueError: JSON格式错误或数据无效
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_bson_bytes(cls, bson_bytes: bytes) -> 'RedisGroupQueueItem':
        """
        从BSON字节数据反序列化对象

        Args:
            bson_bytes: BSON字节数据

        Returns:
            RedisGroupQueueItem: 对象实例

        Raises:
            ValueError: BSON格式错误或数据无效
        """
        raise NotImplementedError

    def to_json_str(self) -> str:
        """
        将对象转换为JSON字符串

        Returns:
            str: JSON字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_bson_bytes(self) -> bytes:
        """
        将对象序列化为BSON字节数据

        Returns:
            bytes: BSON字节数据
        """
        return bson.encode(self.to_dict())


class SimpleQueueItem(RedisGroupQueueItem):
    """简单队列项实现示例"""

    def __init__(self, data: Any, item_type: str = "simple"):
        """
        初始化简单队列项

        Args:
            data: 数据内容
            item_type: 项目类型标识
        """
        self.data = data
        self.item_type = item_type

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {"data": self.data, "item_type": self.item_type}

    @classmethod
    def from_json_str(cls, json_str: str) -> 'SimpleQueueItem':
        """从JSON字符串创建实例"""
        try:
            json_dict = json.loads(json_str)
            return cls(
                data=json_dict["data"], item_type=json_dict.get("item_type", "simple")
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"无效的JSON数据: {e}") from e

    @classmethod
    def from_bson_bytes(cls, bson_bytes: bytes) -> 'SimpleQueueItem':
        """从BSON字节数据创建实例"""
        try:
            data = bson.decode(bson_bytes)
            return cls(data=data["data"], item_type=data.get("item_type", "simple"))
        except (Exception, KeyError) as e:
            raise ValueError(f"无效的BSON数据: {e}") from e

    def __repr__(self) -> str:
        return f"SimpleQueueItem(data={self.data}, item_type={self.item_type})"
