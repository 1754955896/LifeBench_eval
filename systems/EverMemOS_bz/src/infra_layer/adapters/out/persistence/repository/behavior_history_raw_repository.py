from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClientSession
from beanie import PydanticObjectId
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.behavior_history import (
    BehaviorHistory,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("behavior_history_raw_repository", primary=True)
class BehaviorHistoryRawRepository(BaseRepository[BehaviorHistory]):
    """
    行为历史原始数据仓库

    提供用户行为历史数据的CRUD操作和查询功能。
    """

    def __init__(self):
        super().__init__(BehaviorHistory)

    # ==================== 基础CRUD操作 ====================

    async def get_by_user_id(
        self,
        user_id: str,
        limit: int = 100,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[BehaviorHistory]:
        """根据用户ID获取行为历史列表"""
        try:
            results = (
                await self.model.find({"user_id": user_id}, session=session)
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ 根据用户ID获取行为历史成功: %s, count=%d", user_id, len(results)
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户ID获取行为历史失败: %s", e)
            return []

    async def get_by_time_range(
        self,
        start_timestamp: int,
        end_timestamp: int,
        user_id: Optional[str] = None,
        limit: int = 100,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[BehaviorHistory]:
        """根据时间范围获取行为历史列表"""
        try:
            query = {"timestamp": {"$gte": start_timestamp, "$lte": end_timestamp}}
            if user_id:
                query["user_id"] = user_id

            results = (
                await self.model.find(query, session=session)
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ 根据时间范围获取行为历史成功: start=%d, end=%d, count=%d",
                start_timestamp,
                end_timestamp,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据时间范围获取行为历史失败: %s", e)
            return []

    async def append_behavior(
        self,
        behavior_history: BehaviorHistory,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[BehaviorHistory]:
        """追加新的行为历史"""
        try:
            await behavior_history.insert(session=session)
            logger.debug(
                "✅ 追加行为历史成功: user_id=%s, timestamp=%s",
                behavior_history.user_id,
                behavior_history.timestamp,
            )
            return behavior_history
        except Exception as e:
            logger.error("❌ 追加行为历史失败: %s", e)
            return None

    async def delete_by_user_and_timestamp(
        self,
        user_id: str,
        timestamp: int,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> bool:
        """根据用户ID和时间戳删除行为历史"""
        try:
            result = await self.model.find_one(
                {"user_id": user_id, "timestamp": timestamp}, session=session
            )
            if not result:
                logger.warning(
                    "⚠️  未找到要删除的行为历史: user_id=%s, timestamp=%d",
                    user_id,
                    timestamp,
                )
                return False

            await result.delete(session=session)
            logger.info(
                "✅ 根据用户ID和时间戳删除行为历史成功: %s, %d", user_id, timestamp
            )
            return True
        except Exception as e:
            logger.error("❌ 根据用户ID和时间戳删除行为历史失败: %s", e)
            return False

    # ==================== 批量操作 ====================

    async def get_recent_behaviors(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 100,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[BehaviorHistory]:
        """获取用户最近N小时的行为历史"""
        try:
            current_timestamp = int(datetime.now().timestamp())
            start_timestamp = current_timestamp - (hours * 3600)

            results = (
                await self.model.find(
                    {"user_id": user_id, "timestamp": {"$gte": start_timestamp}},
                    session=session,
                )
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ 获取用户最近行为历史成功: user_id=%s, hours=%d, count=%d",
                user_id,
                hours,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 获取用户最近行为历史失败: %s", e)
            return []

    # ==================== 统计方法 ====================

    async def count_by_user(
        self, user_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """统计用户的行为历史数量"""
        try:
            count = await self.model.find({"user_id": user_id}, session=session).count()
            logger.debug(
                "✅ 统计用户行为历史数量成功: user_id=%s, count=%d", user_id, count
            )
            return count
        except Exception as e:
            logger.error("❌ 统计用户行为历史数量失败: %s", e)
            return 0

    async def count_by_time_range(
        self,
        start_timestamp: int,
        end_timestamp: int,
        user_id: Optional[str] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> int:
        """统计时间范围内的行为历史数量"""
        try:
            query = {"timestamp": {"$gte": start_timestamp, "$lte": end_timestamp}}
            if user_id:
                query["user_id"] = user_id

            count = await self.model.find(query, session=session).count()
            logger.debug(
                "✅ 统计时间范围内行为历史数量成功: start=%d, end=%d, count=%d",
                start_timestamp,
                end_timestamp,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ 统计时间范围内行为历史数量失败: %s", e)
            return 0
