"""
Redis 数据处理器

提供统一的数据序列化和反序列化功能，支持：
1. JSON 序列化（优先使用）
2. Pickle 序列化（JSON 失败时的降级处理）
3. 自动检测反序列化类型
"""

import json
import pickle
import uuid
from typing import Any, Union, Dict, List, Tuple
from core.observation.logger import get_logger

logger = get_logger(__name__)

# 配置常量
UUID_LENGTH = 8  # UUID 截取长度
PICKLE_MARKER = b"__PICKLE__"  # Pickle 数据标识符


class RedisDataProcessor:
    """Redis 数据处理器"""

    @staticmethod
    def serialize_data(data: Union[str, Dict, List, Any]) -> Union[str, bytes]:
        """
        序列化数据为字符串或二进制数据

        优先使用 JSON 序列化（返回字符串），失败时使用 Pickle 序列化（返回二进制数据）

        Args:
            data: 要序列化的数据

        Returns:
            Union[str, bytes]: JSON序列化的字符串或Pickle序列化的二进制数据

        Raises:
            ValueError: 序列化失败
        """
        # 如果已经是字符串，直接返回
        if isinstance(data, str):
            return data

        # 优先尝试 JSON 序列化
        try:
            return json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as json_error:
            logger.debug("JSON 序列化失败，尝试使用 Pickle: %s", str(json_error))

            # JSON 失败时使用 Pickle
            try:
                # 使用 Pickle 序列化，并添加标识符
                pickle_data = pickle.dumps(data)
                # 直接返回带标识符的二进制数据（Redis是二进制安全的）
                binary_data = PICKLE_MARKER + pickle_data
                logger.debug("使用 Pickle 序列化成功，数据长度: %d", len(binary_data))
                return binary_data
            except Exception as pickle_error:
                logger.error("Pickle 序列化也失败: %s", str(pickle_error))
                raise ValueError(
                    f"数据序列化失败: JSON错误={json_error}, Pickle错误={pickle_error}"
                ) from pickle_error

    @staticmethod
    def deserialize_data(data: Union[str, bytes]) -> Any:
        """
        反序列化数据

        自动检测是 JSON 还是 Pickle 数据并进行相应的反序列化

        Args:
            data: 序列化后的字符串或二进制数据

        Returns:
            Any: 反序列化后的数据
        """
        # 处理二进制数据（来自 decode_responses=False 的客户端）
        if isinstance(data, bytes):
            # 检查是否有 Pickle 标识符
            if data.startswith(PICKLE_MARKER):
                logger.debug("检测到 Pickle 二进制数据，进行反序列化")
                pickle_data = data[len(PICKLE_MARKER) :]
                try:
                    result = pickle.loads(pickle_data)
                    logger.debug("Pickle 反序列化成功")
                    return result
                except Exception as e:
                    logger.error("Pickle 反序列化失败: %s", str(e))
                    return data
            else:
                # 尝试解码为字符串进行 JSON 反序列化
                try:
                    data_str = data.decode('utf-8')
                    return json.loads(data_str)
                except UnicodeDecodeError:
                    logger.warning("二进制数据无法解码为UTF-8")
                    return data
                except json.JSONDecodeError:
                    # JSON 解析失败，但UTF-8解码成功，返回解码后的字符串
                    logger.debug("JSON解析失败，返回解码后的字符串: %s", data_str[:50])
                    return data_str

        # 处理字符串数据（来自 decode_responses=True 的客户端）
        if isinstance(data, str):
            # 尝试 JSON 反序列化
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError) as json_error:
                logger.debug("JSON 反序列化失败: %s", str(json_error))
                # JSON 失败时，返回原始字符串
                return data

        # 其他类型直接返回
        return data

    @staticmethod
    def create_unique_member(data: Union[str, bytes]) -> Union[str, bytes]:
        """
        创建唯一的成员标识符

        Args:
            data: 序列化后的数据（字符串或二进制）

        Returns:
            Union[str, bytes]: 唯一成员标识符
        """
        unique_id = str(uuid.uuid4())[:UUID_LENGTH]

        if isinstance(data, bytes):
            # 对于二进制数据，使用二进制分隔符
            unique_id_bytes = unique_id.encode('utf-8')
            separator = b":"
            return unique_id_bytes + separator + data
        else:
            # 对于字符串数据，使用字符串格式
            return f"{unique_id}:{data}"

    @staticmethod
    def parse_member_data(member: Union[str, bytes]) -> Tuple[str, Union[str, bytes]]:
        """
        解析成员数据，提取唯一ID和数据内容

        Args:
            member: 成员数据（字符串或二进制格式：unique_id:data）

        Returns:
            Tuple[str, Union[str, bytes]]: (unique_id, data)
        """
        if isinstance(member, bytes):
            # 处理二进制数据（来自 decode_responses=False 的客户端）
            separator = b":"
            if separator in member:
                unique_id_bytes, data = member.split(separator, 1)
                try:
                    unique_id = unique_id_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    unique_id = "unknown"
            else:
                # 兼容旧格式
                unique_id = "unknown"
                data = member
        else:
            # 处理字符串数据（来自 decode_responses=True 的客户端）
            if ':' in member:
                unique_id, data = member.split(':', 1)
            else:
                # 兼容旧格式
                unique_id = "unknown"
                data = member

        return unique_id, data

    @staticmethod
    def process_data_for_storage(
        data: Union[str, Dict, List, Any]
    ) -> Union[str, bytes]:
        """
        处理数据以供存储

        将数据序列化并创建唯一标识符

        Args:
            data: 要处理的数据

        Returns:
            Union[str, bytes]: 可存储的唯一成员数据
        """
        serialized_data = RedisDataProcessor.serialize_data(data)
        return RedisDataProcessor.create_unique_member(serialized_data)

    @staticmethod
    def process_data_from_storage(member: Union[str, bytes]) -> Dict[str, Any]:
        """
        处理从存储中读取的数据

        解析成员数据并反序列化

        Args:
            member: 从 Redis 读取的成员数据（字符串或二进制）

        Returns:
            Dict[str, Any]: 包含解析结果的字典，格式：
                {
                    "id": str,                    # 唯一标识符
                    "data": Any,                  # 反序列化后的原始数据
                    "raw_data": Union[str, bytes] # 序列化的数据
                }
        """
        # Redis 现在返回 bytes，需要先处理
        if isinstance(member, bytes):
            logger.debug("从Redis读取到二进制数据，长度: %d", len(member))

        unique_id, raw_data = RedisDataProcessor.parse_member_data(member)

        try:
            parsed_data = RedisDataProcessor.deserialize_data(raw_data)
        except Exception as e:
            logger.warning(
                "反序列化数据失败: member=%s, error=%s",
                (
                    str(member)[:100]
                    if isinstance(member, str)
                    else f"bytes({len(member)})"
                ),
                str(e),
            )
            # 反序列化失败时，返回原始数据
            parsed_data = raw_data

        return {"id": unique_id, "data": parsed_data, "raw_data": raw_data}


# 为了方便使用，提供模块级别的函数
def serialize_data(data: Union[str, Dict, List, Any]) -> Union[str, bytes]:
    """序列化数据（模块级别函数）"""
    return RedisDataProcessor.serialize_data(data)


def deserialize_data(data: Union[str, bytes]) -> Any:
    """反序列化数据（模块级别函数）"""
    return RedisDataProcessor.deserialize_data(data)


def create_unique_member(data: Union[str, bytes]) -> Union[str, bytes]:
    """创建唯一成员标识符（模块级别函数）"""
    return RedisDataProcessor.create_unique_member(data)


def parse_member_data(member: Union[str, bytes]) -> Tuple[str, Union[str, bytes]]:
    """解析成员数据（模块级别函数）"""
    return RedisDataProcessor.parse_member_data(member)


def process_data_for_storage(data: Union[str, Dict, List, Any]) -> Union[str, bytes]:
    """处理数据以供存储（模块级别函数）"""
    return RedisDataProcessor.process_data_for_storage(data)


def process_data_from_storage(member: Union[str, bytes]) -> Dict[str, Any]:
    """处理从存储中读取的数据（模块级别函数）"""
    return RedisDataProcessor.process_data_from_storage(member)
