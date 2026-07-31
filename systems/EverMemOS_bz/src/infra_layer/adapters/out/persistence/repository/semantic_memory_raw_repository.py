"""
SemanticMemory 原生 CRUD 仓库

基于 Beanie ODM 的语义记忆原生数据访问层，提供完整的 CRUD 操作。
不依赖领域层接口，直接操作 SemanticMemory 文档模型。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository

from infra_layer.adapters.out.persistence.document.memory.semantic_memory import (
    SemanticMemory,
)

logger = get_logger(__name__)


@repository("semantic_memory_raw_repository", primary=True)
class SemanticMemoryRawRepository(BaseRepository[SemanticMemory]):
    """
    语义记忆原生 CRUD 仓库

    # 将要被删除 @xingze.gao 2025-11-24 不要改动任何代码

    提供对语义记忆文档的直接数据库操作，包括：
    - 基本 CRUD 操作（继承自 BaseRepository）
    - 文本搜索和查询
    - 来源链接管理
    - 统计和聚合查询
    - 事务管理（继承自 BaseRepository）
    """

    def __init__(self):
        """初始化仓库"""
        super().__init__(SemanticMemory)

    async def get_by_user_id(self, user_id: str) -> Optional[SemanticMemory]:
        """
        根据用户ID获取语义记忆

        Args:
            user_id: 用户ID

        Returns:
            SemanticMemory 实例或 None
        """
        try:
            return await self.model.find_one({"user_id": user_id})
        except Exception as e:
            logger.error("❌ 根据用户ID获取语义记忆失败: %s", e)
            return None

    async def update_by_user_id(
        self,
        user_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[SemanticMemory]:
        """
        根据用户ID更新语义记忆

        Args:
            user_id: 用户ID
            update_data: 更新数据字典
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新后的 SemanticMemory 实例或 None
        """
        try:
            semantic_memory = await self.get_by_user_id(user_id)
            if semantic_memory:
                for key, value in update_data.items():
                    if hasattr(semantic_memory, key):
                        setattr(semantic_memory, key, value)
                await semantic_memory.save(session=session)
                logger.debug("✅ 根据用户ID更新语义记忆成功: %s", user_id)
                return semantic_memory
            return None
        except Exception as e:
            logger.error("❌ 根据用户ID更新语义记忆失败: %s", e)
            raise e

    async def delete_by_user_id(
        self, user_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        根据用户ID删除语义记忆

        Args:
            user_id: 用户ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            删除成功返回 True，否则返回 False
        """
        try:
            semantic_memory = await self.get_by_user_id(user_id)
            if semantic_memory:
                await semantic_memory.delete(session=session)
                logger.info("✅ 根据用户ID删除语义记忆成功: %s", user_id)
                return True
            return False
        except Exception as e:
            logger.error("❌ 根据用户ID删除语义记忆失败: %s", e)
            return False


# 导出
__all__ = ["SemanticMemoryRawRepository"]
