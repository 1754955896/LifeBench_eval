"""
请求转换器模块

此模块包含各种外部请求格式到内部 Request 对象的转换函数。
"""

from __future__ import annotations

from typing import Any, Dict, List, Union, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from api_specs.memory_models import MemoryType
from api_specs.dtos.memory_query import RetrieveMemRequest, FetchMemRequest
from api_specs.dtos.memory_command import MemorizeRequest
from api_specs.memory_types import RawDataType
from api_specs.dtos.memory_command import RawData

from typing import Dict, Any, Optional
from common_utils.datetime_utils import from_iso_format
from zoneinfo import ZoneInfo
from core.observation.logger import get_logger

logger = get_logger(__name__)


class DataFields:
    """数据字段常量"""

    MESSAGES = "messages"
    RAW_DATA_TYPE = "raw_data_type"
    GROUP_ID = "group_id"


def convert_dict_to_fetch_mem_request(data: Dict[str, Any]) -> FetchMemRequest:
    """
    将字典转换为 FetchMemRequest 对象

    Args:
        data: 包含 FetchMemRequest 字段的字典

    Returns:
        FetchMemRequest 对象

    Raises:
        ValueError: 当必需字段缺失或类型不正确时
    """
    try:
        # 验证必需字段
        if "user_id" not in data:
            raise ValueError("user_id 是必需字段")

        # 转换 memory_type，如果未提供则使用默认值
        memory_type = MemoryType(data.get("memory_type", "multiple"))
        logger.debug(f"version_range: {data.get('version_range', None)}")

        # 转换 limit 和 offset 为整数类型（从 query_params 获取的都是字符串）
        limit = data.get("limit", 10)
        offset = data.get("offset", 0)
        if isinstance(limit, str):
            limit = int(limit)
        if isinstance(offset, str):
            offset = int(offset)

        # 构建 FetchMemRequest 对象
        return FetchMemRequest(
            user_id=data["user_id"],
            memory_type=memory_type,
            limit=limit,
            offset=offset,
            filters=data.get("filters", {}),
            sort_by=data.get("sort_by"),
            sort_order=data.get("sort_order", "desc"),
            version_range=data.get("version_range", None),
        )
    except Exception as e:
        raise ValueError(f"FetchMemRequest 转换失败: {e}")


def convert_dict_to_retrieve_mem_request(
    data: Dict[str, Any], query: Optional[str] = None
) -> RetrieveMemRequest:
    """
    将字典转换为 RetrieveMemRequest 对象

    Args:
        data: 包含 RetrieveMemRequest 字段的字典
        query: 查询文本（可选）

    Returns:
        RetrieveMemRequest 对象

    Raises:
        ValueError: 当必需字段缺失或类型不正确时
    """
    try:
        # 验证必需字段
        if "user_id" not in data:
            raise ValueError("user_id 是必需字段")

        # 处理 retrieve_method，如果未提供则使用默认值 keyword
        from api_specs.memory_models import RetrieveMethod

        retrieve_method_str = data.get("retrieve_method", "keyword")

        # 将字符串转换为 RetrieveMethod 枚举
        try:
            retrieve_method = RetrieveMethod(retrieve_method_str)
        except ValueError:
            logger.warning(
                f"无效的 retrieve_method: {retrieve_method_str}, 使用默认值 keyword"
            )
            retrieve_method = RetrieveMethod.KEYWORD

        # 转换 top_k 为整数类型（从 query_params 获取的都是字符串）
        top_k = data.get("top_k", 10)
        if isinstance(top_k, str):
            top_k = int(top_k)

        # 转换 include_metadata 为布尔类型
        include_metadata = data.get("include_metadata", True)
        if isinstance(include_metadata, str):
            include_metadata = include_metadata.lower() in ("true", "1", "yes")

        # 转换 radius 为浮点类型（如果存在）
        radius = data.get("radius", None)
        if radius is not None and isinstance(radius, str):
            radius = float(radius)

        return RetrieveMemRequest(
            retrieve_method=retrieve_method,
            user_id=data["user_id"],
            query=query or data.get("query", None),
            memory_types=data.get("memory_types", []),
            top_k=top_k,
            filters=data.get("filters", {}),
            include_metadata=include_metadata,
            start_time=data.get("start_time", None),
            end_time=data.get("end_time", None),
            radius=radius,  # COSINE 相似度阈值
        )
    except Exception as e:
        raise ValueError(f"RetrieveMemRequest 转换失败: {e}")


# =========================================


def _extract_current_time(data: Dict[str, Any]) -> Optional[datetime]:
    """
    从数据中提取 current_time 字段

    Args:
        data: 数据字典

    Returns:
        current_time 或 None
    """
    if "current_time" not in data:
        return None

    current_time_str = data["current_time"]
    if isinstance(current_time_str, str):
        try:
            return datetime.fromisoformat(current_time_str.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"无法解析 current_time: {current_time_str}")
            return None
    elif isinstance(current_time_str, datetime):
        # from anhua
        if current_time_str.tzinfo is None:
            return current_time_str.replace(tzinfo=ZoneInfo("UTC"))
        return current_time_str

    return None


def _create_memorize_request(
    history_data: List[RawData],
    new_data: List[RawData],
    data_type: RawDataType,
    participants: List[str],
    group_id: str = None,
    group_name: str = None,
    current_time: datetime = None,
) -> MemorizeRequest:
    """
    创建 MemorizeRequest 对象的公共函数

    Args:
        history_data: 历史数据列表
        new_data: 新数据列表
        data_type: 数据类型
        participants: 参与者列表
        group_id: 群组ID
        group_name: 群组名称
        current_time: 当前时间

    Returns:
        MemorizeRequest 对象
    """
    # 确保 participants 不为 None
    if participants is None:
        participants = []

    # 如果 current_time 为 None，尝试从 new_data[0] 的 timestamp 或 updateTime 来获取
    if current_time is None and new_data and new_data[0] is not None:
        first_data = new_data[0]
        if hasattr(first_data, 'content') and first_data.content:
            # 优先使用 updateTime
            if 'updateTime' in first_data.content and first_data.content['updateTime']:
                current_time = first_data.content['updateTime']
            elif 'timestamp' in first_data.content and first_data.content['timestamp']:
                current_time = first_data.content['timestamp']

    return MemorizeRequest(
        history_raw_data_list=history_data,
        new_raw_data_list=new_data,
        raw_data_type=data_type,
        user_id_list=participants,
        group_id=group_id,
        group_name=group_name,
        current_time=current_time,
    )


async def convert_single_message_to_raw_data(
    input_data: Dict[str, Any],
    data_id_field: str = "_id",
    group_name: Optional[str] = None,
) -> RawData:
    """
    将输入数据转换为 RawData 格式

    Args:
        input_data: 包含 _id, fullName, receiverId, roomId, userIdList,
                   referList, content, createTime, createBy, updateTime, orgId 的字典
        data_id_field: 用作 data_id 的字段名，默认为 "_id"
        group_name: 群组名称（从外部传入，会添加到消息的 content 中）

    Returns:
        RawData 对象
    """
    # 提取 data_id
    data_id = str(input_data.get(data_id_field, ""))

    room_id = input_data.get("roomId")

    # group_name 完全依赖外部传入
    # 如果外部没有传入，则为 None（不再查询数据库）
    if group_name:
        logger.debug("使用外部传入的 group_name: %s", group_name)
    else:
        logger.debug("未从外部传入 group_name，将使用 None")

    # 构建 content 字典，包含所有业务相关字段
    content = {
        "speaker_name": input_data.get("fullName"),
        "receiverId": input_data.get("receiverId"),
        "roomId": room_id,
        "groupName": group_name,  # 添加群组名称
        "userIdList": input_data.get("userIdList", []),
        "referList": input_data.get("referList", []),
        "content": input_data.get("content"),
        "timestamp": from_iso_format(
            input_data.get("createTime"), ZoneInfo("UTC")
        ),  # 使用转换后的UTC时间
        "createBy": input_data.get("createBy"),
        "updateTime": from_iso_format(
            input_data.get("updateTime"), ZoneInfo("UTC")
        ),  # 使用转换后的UTC时间
        "orgId": input_data.get("orgId"),
        "speaker_id": input_data.get("createBy"),
        "msgType": input_data.get("msgType"),
        "data_id": data_id,
    }

    # 如果input_data中包含这些字段，则添加到content中
    if "readStatus" in input_data:
        content["readStatus"] = input_data.get("readStatus")
    if "notifyType" in input_data:
        content["notifyType"] = input_data.get("notifyType")
    if "isReplySuggest" in input_data:
        content["isReplySuggest"] = input_data.get("isReplySuggest")
    if "readUpdateTime" in input_data:
        content["readUpdateTime"] = from_iso_format(
            input_data.get("readUpdateTime"), ZoneInfo("UTC")
        )

    # 构建 metadata，包含系统字段
    metadata = {
        "original_id": data_id,
        "createTime": from_iso_format(
            input_data.get("createTime"), ZoneInfo("UTC")
        ),  # 使用转换后的UTC时间
        "updateTime": from_iso_format(
            input_data.get("updateTime"), ZoneInfo("UTC")
        ),  # 使用转换后的UTC时间
        "createBy": input_data.get("createBy"),
        "orgId": input_data.get("orgId"),
    }

    return RawData(content=content, data_id=data_id, metadata=metadata)


async def convert_conversation_to_raw_data_list(
    input_data_list: list[Dict[str, Any]],
    data_id_field: str = "_id",
    group_name: Optional[str] = None,
) -> list[RawData]:
    """
    批量转换数据为 RawData 格式

    Args:
        input_data_list: 输入数据列表
        data_id_field: 用作 data_id 的字段名，默认为 "_id"
        group_name: 群组名称（从外部传入，会传递给每条消息转换）

    Returns:
        RawData 对象列表
    """
    return [
        await convert_single_message_to_raw_data(
            data, data_id_field=data_id_field, group_name=group_name
        )
        for data in input_data_list
    ]


async def handle_conversation_format(data: Dict[str, Any]) -> MemorizeRequest:
    """
    处理聊天消息格式数据

    Args:
        data: 包含 messages 字段的数据

    Returns:
        MemorizeRequest 对象
    """
    logger.debug("处理聊天消息格式数据")
    messages = data.get(DataFields.MESSAGES, [])
    if not messages:
        raise ValueError("messages 字段不能为空")

    # 提取群组级别信息
    group_name = data.get("group_name")

    # 转换为 RawData 格式，传递群组名称
    raw_data_list = await convert_conversation_to_raw_data_list(
        messages, group_name=group_name
    )

    # 提取 current_time
    current_time = _extract_current_time(data)

    # 提取参与者
    participants = []

    # 计算分割点（80%作为历史消息）
    split_ratio = data.get("split_ratio", 0.8)
    split_index = int(len(raw_data_list) * split_ratio)

    # 分割历史消息和新消息
    history_raw_data_list = raw_data_list[:split_index]
    new_raw_data_list = raw_data_list[split_index:]

    # 如果没有新消息，将最后一条消息作为新消息
    if not new_raw_data_list and history_raw_data_list:
        new_raw_data_list = [history_raw_data_list.pop()]

    return _create_memorize_request(
        history_data=history_raw_data_list,
        new_data=new_raw_data_list,
        data_type=RawDataType(data.get(DataFields.RAW_DATA_TYPE, "Conversation")),
        participants=participants,
        group_id=data.get(DataFields.GROUP_ID),
        group_name=data.get("group_name"),
        current_time=current_time,
    )
