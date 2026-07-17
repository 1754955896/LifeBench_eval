"""
Kafka ConsumerRecord 队列项实现

提供 ConsumerRecord 与 RedisGroupQueueItem 之间的序列化/反序列化功能
使用 BSON 格式处理二进制数据，确保数据完整性
"""

import json
import base64
from typing import Optional, Sequence, Tuple, Any, Dict
from dataclasses import dataclass

import bson

from aiokafka import ConsumerRecord
from core.observation.logger import get_logger
from .redis_group_queue_item import RedisGroupQueueItem

logger = get_logger(__name__)


@dataclass
class KafkaConsumerRecordItem(RedisGroupQueueItem):
    """
    Kafka ConsumerRecord 队列项

    实现 RedisGroupQueueItem 接口，提供 ConsumerRecord 的序列化/反序列化功能
    """

    # ConsumerRecord 字段
    topic: str
    partition: int
    offset: int
    timestamp: int
    timestamp_type: int
    key: Optional[str]
    value: Optional[Any]
    checksum: Optional[int]
    serialized_key_size: int
    serialized_value_size: int
    headers: Sequence[Tuple[str, bytes]]

    def __init__(self, consumer_record: ConsumerRecord):
        """
        从 ConsumerRecord 初始化

        Args:
            consumer_record: aiokafka ConsumerRecord 对象
        """
        self.topic = consumer_record.topic
        self.partition = consumer_record.partition
        self.offset = consumer_record.offset
        self.timestamp = consumer_record.timestamp
        self.timestamp_type = consumer_record.timestamp_type
        # 处理 key：如果是 bytes，转换为 base64 字符串
        self.key = (
            self._encode_bytes_to_base64(consumer_record.key)
            if isinstance(consumer_record.key, bytes)
            else consumer_record.key
        )
        # 处理 value：保持原始格式，BSON 可以直接处理各种类型
        self.value = consumer_record.value
        self.checksum = consumer_record.checksum
        self.serialized_key_size = consumer_record.serialized_key_size
        self.serialized_value_size = consumer_record.serialized_value_size
        # 转换 headers 为可序列化格式，保持二进制数据
        self.headers = [
            (
                name,
                (
                    self._encode_bytes_to_base64(data)
                    if isinstance(data, bytes)
                    else str(data)
                ),
            )
            for name, data in consumer_record.headers
        ]

    def _encode_bytes_to_base64(self, data: bytes) -> str:
        """将 bytes 数据编码为 base64 字符串"""
        return base64.b64encode(data).decode('utf-8')

    def _decode_base64_to_bytes(self, data: str) -> bytes:
        """将 base64 字符串解码为 bytes 数据"""
        return base64.b64decode(data.encode('utf-8'))

    def to_dict(self) -> Dict[str, Any]:
        """
        将对象转换为可JSON序列化的字典

        Returns:
            Dict[str, Any]: 可序列化的字典
        """
        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "timestamp": self.timestamp,
            "timestamp_type": self.timestamp_type,
            "key": self.key,
            "value": self.value,
            "checksum": self.checksum,
            "serialized_key_size": self.serialized_key_size,
            "serialized_value_size": self.serialized_value_size,
            "headers": self.headers,
        }

    def to_json_str(self) -> str:
        """
        不支持 JSON 序列化，请使用 BSON 序列化
        """
        raise NotImplementedError(
            "KafkaConsumerRecordItem 不支持 JSON 序列化，请使用 to_bson_bytes() 方法"
        )

    def to_bson_bytes(self) -> bytes:
        """
        序列化为 BSON 字节数据

        Returns:
            bytes: BSON 字节数据
        """
        try:
            data = self.to_dict()  # 获取可序列化的字典
            return bson.encode(data)
        except Exception as e:
            logger.error("BSON 序列化 KafkaConsumerRecordItem 失败: %s", e)
            raise ValueError(f"BSON 序列化失败: {e}") from e

    @classmethod
    def from_json_str(cls, json_str: str) -> 'KafkaConsumerRecordItem':
        """
        不支持 JSON 反序列化，请使用 BSON 反序列化
        """
        raise NotImplementedError(
            "KafkaConsumerRecordItem 不支持 JSON 反序列化，请使用 from_bson_bytes() 方法"
        )

    @classmethod
    def from_bson_bytes(cls, bson_bytes: bytes) -> 'KafkaConsumerRecordItem':
        """
        从 BSON 字节数据反序列化

        Args:
            bson_bytes: BSON 字节数据

        Returns:
            KafkaConsumerRecordItem: 反序列化的对象
        """
        try:
            data = bson.decode(bson_bytes)

            # 创建实例
            item = cls.__new__(cls)  # 绕过 __init__
            item.topic = data["topic"]
            item.partition = data["partition"]
            item.offset = data["offset"]
            item.timestamp = data["timestamp"]
            item.timestamp_type = data["timestamp_type"]
            item.key = data["key"]
            item.value = data["value"]
            item.checksum = data["checksum"]
            item.serialized_key_size = data["serialized_key_size"]
            item.serialized_value_size = data["serialized_value_size"]
            item.headers = data["headers"]

            return item
        except Exception as e:
            logger.error("BSON 反序列化 KafkaConsumerRecordItem 失败: %s", e)
            raise ValueError(f"BSON 反序列化失败: {e}") from e

    def to_consumer_record(self) -> ConsumerRecord:
        """
        转换为 aiokafka ConsumerRecord 对象

        Returns:
            ConsumerRecord: aiokafka ConsumerRecord 对象
        """
        try:
            # 处理 key：如果是字符串，尝试从 base64 解码，否则保持原样
            key = self.key
            if isinstance(key, str):
                try:
                    key = self._decode_base64_to_bytes(key)
                except Exception:
                    # 如果解码失败，保持原字符串
                    pass

            # 处理 headers：将字符串数据从 base64 解码回 bytes
            headers_bytes = []
            for name, data in self.headers:
                if isinstance(data, str):
                    try:
                        # 尝试从 base64 解码
                        headers_bytes.append((name, self._decode_base64_to_bytes(data)))
                    except Exception:
                        # 如果解码失败，编码为 UTF-8 bytes
                        headers_bytes.append((name, data.encode('utf-8')))
                else:
                    headers_bytes.append((name, bytes(data)))

            return ConsumerRecord(
                topic=self.topic,
                partition=self.partition,
                offset=self.offset,
                timestamp=self.timestamp,
                timestamp_type=self.timestamp_type,
                key=key,
                value=self.value,
                checksum=self.checksum,
                serialized_key_size=self.serialized_key_size,
                serialized_value_size=self.serialized_value_size,
                headers=headers_bytes,
            )
        except Exception as e:
            logger.error("转换为 ConsumerRecord 失败: %s", e)
            raise ValueError(f"转换失败: {e}") from e

    def __repr__(self) -> str:
        return (
            f"KafkaConsumerRecordItem(topic={self.topic}, partition={self.partition}, "
            f"offset={self.offset}, timestamp={self.timestamp})"
        )


def consumer_record_to_queue_item(
    consumer_record: ConsumerRecord,
) -> KafkaConsumerRecordItem:
    """
    将 ConsumerRecord 转换为队列项

    Args:
        consumer_record: aiokafka ConsumerRecord 对象

    Returns:
        KafkaConsumerRecordItem: 队列项
    """
    return KafkaConsumerRecordItem(consumer_record)


def queue_item_to_consumer_record(
    queue_item: KafkaConsumerRecordItem,
) -> ConsumerRecord:
    """
    将队列项转换为 ConsumerRecord

    Args:
        queue_item: 队列项

    Returns:
        ConsumerRecord: aiokafka ConsumerRecord 对象
    """
    return queue_item.to_consumer_record()


def serialize_consumer_record_to_bson(consumer_record: ConsumerRecord) -> bytes:
    """
    将 ConsumerRecord 序列化为 BSON 字节数据

    Args:
        consumer_record: aiokafka ConsumerRecord 对象

    Returns:
        bytes: BSON 序列化的字节数据
    """
    queue_item = consumer_record_to_queue_item(consumer_record)
    return queue_item.to_bson_bytes()


def deserialize_bson_to_consumer_record(bson_bytes: bytes) -> ConsumerRecord:
    """
    从 BSON 字节数据反序列化为 ConsumerRecord

    Args:
        bson_bytes: BSON 字节数据

    Returns:
        ConsumerRecord: aiokafka ConsumerRecord 对象
    """
    queue_item = KafkaConsumerRecordItem.from_bson_bytes(bson_bytes)
    return queue_item_to_consumer_record(queue_item)
