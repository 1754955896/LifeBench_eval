from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClientSession
from beanie import PydanticObjectId
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.entity import Entity
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("entity_raw_repository", primary=True)
class EntityRawRepository(BaseRepository[Entity]):
    """
    实体库原始数据仓库

    提供实体数据的CRUD操作和查询功能。
    """

    def __init__(self):
        super().__init__(Entity)

    # ==================== 基础CRUD操作 ====================

    async def get_by_entity_id(
        self, entity_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> Optional[Entity]:
        """根据实体ID获取实体"""
        try:
            result = await self.model.find_one(
                {"entity_id": entity_id}, session=session
            )
            if result:
                logger.debug("✅ 根据实体ID获取实体成功: %s", entity_id)
            else:
                logger.debug("⚠️  未找到实体: entity_id=%s", entity_id)
            return result
        except Exception as e:
            logger.error("❌ 根据实体ID获取实体失败: %s", e)
            return None

    async def get_by_alias(
        self, alias: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> List[Entity]:
        """根据别名获取实体列表"""
        try:
            results = await self.model.find(
                {"aliases": {"$in": [alias]}}, session=session
            ).to_list()
            logger.debug(
                "✅ 根据别名获取实体成功: alias=%s, count=%d", alias, len(results)
            )
            return results
        except Exception as e:
            logger.error("❌ 根据别名获取实体失败: %s", e)
            return []

    async def update_by_entity_id(
        self,
        entity_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[Entity]:
        """根据实体ID更新实体"""
        try:
            existing_doc = await self.model.find_one(
                {"entity_id": entity_id}, session=session
            )
            if not existing_doc:
                logger.warning("⚠️  未找到要更新的实体: entity_id=%s", entity_id)
                return None

            for key, value in update_data.items():
                setattr(existing_doc, key, value)
            await existing_doc.save(session=session)
            logger.debug("✅ 根据实体ID更新实体成功: %s", entity_id)
            return existing_doc
        except Exception as e:
            logger.error("❌ 根据实体ID更新实体失败: %s", e)
            return None

    async def delete_by_entity_id(
        self, entity_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """根据实体ID删除实体"""
        try:
            result = await self.model.find_one(
                {"entity_id": entity_id}, session=session
            )
            if not result:
                logger.warning("⚠️  未找到要删除的实体: entity_id=%s", entity_id)
                return False

            await result.delete(session=session)
            logger.debug("✅ 根据实体ID删除实体成功: %s", entity_id)
            return True
        except Exception as e:
            logger.error("❌ 根据实体ID删除实体失败: %s", e)
            return False

    # ==================== 批量操作 ====================

    async def get_all_entities(
        self, limit: int = 100, session: Optional[AsyncIOMotorClientSession] = None
    ) -> List[Entity]:
        """获取所有实体"""
        try:
            results = await self.model.find({}, session=session).limit(limit).to_list()
            logger.debug("✅ 获取所有实体成功: count=%d", len(results))
            return results
        except Exception as e:
            logger.error("❌ 获取所有实体失败: %s", e)
            return []

    async def get_entities_by_ids(
        self, entity_ids: List[str], session: Optional[AsyncIOMotorClientSession] = None
    ) -> List[Entity]:
        """根据实体ID列表批量获取实体"""
        try:
            results = await self.model.find(
                {"entity_id": {"$in": entity_ids}}, session=session
            ).to_list()
            logger.debug("✅ 根据ID列表批量获取实体成功: count=%d", len(results))
            return results
        except Exception as e:
            logger.error("❌ 根据ID列表批量获取实体失败: %s", e)
            return []

    # ==================== 统计方法 ====================

    async def count_by_type(
        self, entity_type: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """统计指定类型的实体数量"""
        try:
            count = await self.model.find(
                {"type": entity_type}, session=session
            ).count()
            logger.debug("✅ 统计实体数量成功: type=%s, count=%d", entity_type, count)
            return count
        except Exception as e:
            logger.error("❌ 统计实体数量失败: %s", e)
            return 0

    async def count_all(
        self, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """统计所有实体数量"""
        try:
            count = await self.model.count()
            logger.debug("✅ 统计所有实体数量成功: count=%d", count)
            return count
        except Exception as e:
            logger.error("❌ 统计所有实体数量失败: %s", e)
            return 0
