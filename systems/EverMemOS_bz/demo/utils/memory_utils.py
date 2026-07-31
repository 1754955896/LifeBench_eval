"""共享工具函数模块 - 用于记忆提取和对话系统

本模块提供公共的工具函数，供 extract_memory.py 和 chat_with_memory.py 共同使用。

主要功能：
- MongoDB 连接和初始化
- MemCell 查询
- 时间序列化工具
- Prompt 语言设置

V4 更新：
- 删除了自定义检索策略（使用 src 中的 API）
- 保留基础工具函数
- 新增 Prompt 语言设置功能
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# 导入项目中的文档模型
from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell as DocMemCell,
)
from demo.config import MongoDBConfig


# ============================================================================
# Prompt 语言设置
# ============================================================================


def set_prompt_language(language: str) -> None:
    """设置记忆提取的 Prompt 语言
    
    通过设置环境变量 MEMORY_LANGUAGE 来控制 src/memory_layer/prompts 使用的语言。
    这会影响所有记忆提取器（MemCell、Profile、Episode、Semantic）使用的 Prompt。
    
    Args:
        language: 语言代码，"zh" 或 "en"
        
    注意：
        - 必须在导入 memory_layer 相关模块之前调用
        - 建议在程序启动时立即调用
    """
    if language not in ["zh", "en"]:
        print(f"[Warning] 不支持的语言 '{language}'，将使用默认语言 'en'")
        language = "en"
    
    os.environ["MEMORY_LANGUAGE"] = language
    print(f"[Prompt Language] 已设置为: {language} (影响所有记忆提取 Prompt)")


def get_prompt_language() -> str:
    """获取当前的 Prompt 语言设置
    
    Returns:
        当前的 MEMORY_LANGUAGE 环境变量值，默认为 "en"
    """
    return os.getenv("MEMORY_LANGUAGE", "en")


# ============================================================================
# MongoDB 相关工具
# ============================================================================


async def ensure_mongo_beanie_ready(mongo_config: MongoDBConfig) -> None:
    """初始化 MongoDB 和 Beanie 连接

    Args:
        mongo_config: MongoDB 配置对象

    Raises:
        Exception: 如果连接失败
    """
    # 设置环境变量供 Beanie 使用
    os.environ["MONGODB_URI"] = mongo_config.uri

    # 创建 MongoDB 客户端并测试连接
    client = AsyncIOMotorClient(mongo_config.uri)
    try:
        await client.admin.command('ping')
        print(f"[MongoDB] ✅ 连接成功: {mongo_config.database}")
    except Exception as e:
        print(f"[MongoDB] ❌ 连接失败: {e}")
        raise

    # 初始化 Beanie 文档模型
    await init_beanie(
        database=client[mongo_config.database], document_models=[DocMemCell]
    )


async def query_all_groups_from_mongodb() -> List[Dict[str, Any]]:
    """查询所有群组 ID 及其记忆数量

    使用聚合管道统计每个群组的 MemCell 数量。

    Returns:
        群组列表，格式：[{"group_id": "xxx", "memcell_count": 76}, ...]
    """
    # 使用聚合管道统计每个群组的记忆数量
    pipeline = [
        {"$match": {"group_id": {"$ne": None}}},  # 过滤掉没有 group_id 的记录
        {"$group": {"_id": "$group_id", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},  # 按 group_id 排序
    ]

    # 获取 PyMongo/Motor 集合进行聚合查询
    # get_pymongo_collection() 在 Beanie 中返回 Motor 集合（异步）
    collection = DocMemCell.get_pymongo_collection()
    cursor = collection.aggregate(pipeline)
    results = await cursor.to_list(length=None)

    groups = []
    for result in results:
        groups.append({"group_id": result["_id"], "memcell_count": result["count"]})

    return groups


async def query_memcells_by_group_and_time(
    group_id: str, start_date: datetime, end_date: datetime
) -> List[DocMemCell]:
    """按群组和时间范围查询 MemCell

    Args:
        group_id: 群组 ID
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        MemCell 文档对象列表
    """
    memcells = (
        await DocMemCell.find(
            {"group_id": group_id, "timestamp": {"$gte": start_date, "$lt": end_date}}
        )
        .sort("timestamp")
        .to_list()
    )

    return memcells


# ============================================================================
# 时间序列化工具
# ============================================================================


def serialize_datetime(obj: Any) -> Any:
    """递归序列化 datetime 对象为 ISO 格式字符串

    Args:
        obj: 要序列化的对象（可以是任意类型）

    Returns:
        序列化后的对象
    """
    # 如果已经是字符串，直接返回（避免处理已序列化的时间戳）
    if isinstance(obj, str):
        return obj
    # datetime 对象转为 ISO 字符串
    elif isinstance(obj, datetime):
        return obj.isoformat()
    # 递归处理字典
    elif isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    # 递归处理列表
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    # 处理对象（转换 __dict__）
    elif hasattr(obj, '__dict__'):
        return serialize_datetime(obj.__dict__)
    # 其他类型直接返回
    else:
        return obj

