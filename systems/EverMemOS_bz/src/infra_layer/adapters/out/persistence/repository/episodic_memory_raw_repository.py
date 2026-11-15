from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from agentic_layer.vectorize_service import get_vectorize_service
import time

logger = get_logger(__name__)


@repository("episodic_memory_raw_repository", primary=True)
class EpisodicMemoryRawRepository(BaseRepository[EpisodicMemory]):
    """
    情景记忆原始数据仓库
    会生成向量化的文本内容，并保存到数据库中
    提供情景记忆的 CRUD 操作和基础查询功能。
    """

    def __init__(self):
        super().__init__(EpisodicMemory)
        self.vectorize_service = get_vectorize_service()

    # ==================== 基础 CRUD 方法 ====================

    async def get_by_event_id(
        self,
        event_id: str,
        user_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[EpisodicMemory]:
        """
        根据事件ID和用户ID获取情景记忆

        Args:
            event_id: 事件ID
            user_id: 用户ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            EpisodicMemory 或 None
        """
        try:
            # 将字符串 event_id 转换为 ObjectId
            object_id = ObjectId(event_id)
            result = await self.model.find_one(
                {"_id": object_id, "user_id": user_id}, session=session
            )
            if result:
                logger.debug("✅ 根据事件ID和用户ID获取情景记忆成功: %s", event_id)
            else:
                logger.debug(
                    "ℹ️  未找到情景记忆: event_id=%s, user_id=%s", event_id, user_id
                )
            return result
        except Exception as e:
            logger.error("❌ 根据事件ID和用户ID获取情景记忆失败: %s", e)
            return None

    async def get_by_event_ids(
        self,
        event_ids: List[str],
        user_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Dict[str, EpisodicMemory]:
        """
        根据事件ID列表和用户ID批量获取情景记忆

        Args:
            event_ids: 事件ID列表
            user_id: 用户ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            Dict[str, EpisodicMemory]: 以 event_id 为 key 的字典，方便快速查找
        """
        if not event_ids:
            return {}

        try:
            # 将字符串 event_id 列表转换为 ObjectId 列表
            object_ids = []
            for event_id in event_ids:
                try:
                    object_ids.append(ObjectId(event_id))
                except Exception as e:
                    logger.warning(f"⚠️  无效的 event_id: {event_id}, 错误: {e}")
                    continue

            if not object_ids:
                return {}

            # 批量查询
            query = {"_id": {"$in": object_ids}}
            if user_id:
                query["user_id"] = user_id
            
            results = await self.model.find(
                query, session=session
            ).to_list()

            # 转换为字典，方便后续使用
            result_dict = {str(doc.id): doc for doc in results}

            logger.debug(
                "✅ 批量获取情景记忆成功: user_id=%s, 请求 %d 个, 找到 %d 个",
                user_id,
                len(event_ids),
                len(result_dict),
            )
            return result_dict
        except Exception as e:
            logger.error("❌ 批量获取情景记忆失败: %s", e)
            return {}

    async def get_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[EpisodicMemory]:
        """
        根据用户ID获取情景记忆列表

        Args:
            user_id: 用户ID
            limit: 限制返回数量
            skip: 跳过数量
            sort_desc: 是否按时间降序排序
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            EpisodicMemory 列表
        """
        try:
            query = self.model.find({"user_id": user_id})

            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据用户ID获取情景记忆成功: %s, 找到 %d 条记录",
                user_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户ID获取情景记忆失败: %s", e)
            return []

    async def append_episodic_memory(
        self,
        episodic_memory: EpisodicMemory,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[EpisodicMemory]:
        """
        追加新的情景记忆

        Args:
            episodic_memory: 情景记忆对象
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            追加的 EpisodicMemory 或 None
        """

        # 同步向量
        if episodic_memory.episode and not episodic_memory.vector:
            try:
                vector = await self.vectorize_service.get_embedding(
                    episodic_memory.episode
                )
                episodic_memory.vector = vector.tolist()
                # 设置向量化模型信息
                episodic_memory.vector_model = self.vectorize_service.get_model_name()
            except Exception as e:
                logger.error("❌ 同步向量失败: %s", e)
        try:
            await episodic_memory.insert(session=session)
            logger.info(
                "✅ 追加情景记忆成功: event_id=%s, user_id=%s",
                episodic_memory.event_id,
                episodic_memory.user_id,
            )
            return episodic_memory
        except Exception as e:
            logger.error("❌ 追加情景记忆失败: %s", e)
            return None

    async def delete_by_event_id(
        self,
        event_id: str,
        user_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> bool:
        """
        根据事件ID和用户ID删除情景记忆

        Args:
            event_id: 事件ID
            user_id: 用户ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否删除成功
        """
        try:
            # 将字符串 event_id 转换为 ObjectId
            object_id = ObjectId(event_id)
            # 直接删除并检查删除数量
            result = await self.model.find(
                {"_id": object_id, "user_id": user_id}, session=session
            ).delete()

            deleted_count = (
                result.deleted_count if hasattr(result, 'deleted_count') else 0
            )
            success = deleted_count > 0

            if success:
                logger.info("✅ 根据事件ID和用户ID删除情景记忆成功: %s", event_id)
                return True
            else:
                logger.warning(
                    "⚠️  未找到要删除的情景记忆: event_id=%s, user_id=%s",
                    event_id,
                    user_id,
                )
                return False
        except Exception as e:
            logger.error("❌ 根据事件ID和用户ID删除情景记忆失败: %s", e)
            return False

    async def delete_by_user_id(
        self, user_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """
        根据用户ID删除所有情景记忆

        Args:
            user_id: 用户ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            删除的记录数量
        """
        try:
            result = await self.model.find({"user_id": user_id}).delete(session=session)
            count = result.deleted_count if result else 0
            logger.info(
                "✅ 根据用户ID删除情景记忆成功: %s, 删除 %d 条记录", user_id, count
            )
            return count
        except Exception as e:
            logger.error("❌ 根据用户ID删除情景记忆失败: %s", e)
            return 0

    async def find_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = False,
    ) -> List[EpisodicMemory]:
        """
        根据时间范围查询 EpisodicMemory

        Args:
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制返回数量
            skip: 跳过数量
            sort_desc: 是否按时间降序排序，默认False（升序）

        Returns:
            EpisodicMemory 列表
        """
        try:
            query = self.model.find(
                {"timestamp": {"$gte": start_time, "$lt": end_time}}
            )

            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据时间范围查询 EpisodicMemory 成功: 时间范围: %s - %s, 找到 %d 条记录",
                start_time,
                end_time,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据时间范围查询 EpisodicMemory 失败: %s", e)
            return []


# 导出
__all__ = ["EpisodicMemoryRawRepository"]
