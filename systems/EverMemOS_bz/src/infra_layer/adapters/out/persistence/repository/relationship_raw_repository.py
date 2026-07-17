from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClientSession
from beanie import PydanticObjectId
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.relationship import (
    Relationship,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("relationship_raw_repository", primary=True)
class RelationshipRawRepository(BaseRepository[Relationship]):
    """
    关系库原始数据仓库

    提供实体关系数据的CRUD操作和查询功能。
    """

    def __init__(self):
        super().__init__(Relationship)

    # ==================== 基础CRUD操作 ====================

    async def get_by_entity_ids(
        self,
        source_entity_id: str,
        target_entity_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[Relationship]:
        """根据源实体ID和目标实体ID获取关系"""
        try:
            result = await self.model.find_one(
                {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                },
                session=session,
            )
            if result:
                logger.debug(
                    "✅ 根据实体ID获取关系成功: %s -> %s",
                    source_entity_id,
                    target_entity_id,
                )
            else:
                logger.debug(
                    "⚠️  未找到关系: source=%s, target=%s",
                    source_entity_id,
                    target_entity_id,
                )
            return result
        except Exception as e:
            logger.error("❌ 根据实体ID获取关系失败: %s", e)
            return None

    async def get_by_source_entity(
        self,
        source_entity_id: str,
        limit: int = 100,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[Relationship]:
        """根据源实体ID获取所有关系"""
        try:
            results = (
                await self.model.find(
                    {"source_entity_id": source_entity_id}, session=session
                )
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ 根据源实体ID获取关系成功: %s, count=%d",
                source_entity_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据源实体ID获取关系失败: %s", e)
            return []

    async def get_by_target_entity(
        self,
        target_entity_id: str,
        limit: int = 100,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[Relationship]:
        """根据目标实体ID获取所有关系"""
        try:
            results = (
                await self.model.find(
                    {"target_entity_id": target_entity_id}, session=session
                )
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ 根据目标实体ID获取关系成功: %s, count=%d",
                target_entity_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据目标实体ID获取关系失败: %s", e)
            return []

    async def update_by_entity_ids(
        self,
        source_entity_id: str,
        target_entity_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[Relationship]:
        """根据实体ID更新关系"""
        try:
            existing_doc = await self.model.find_one(
                {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                },
                session=session,
            )
            if not existing_doc:
                logger.warning(
                    "⚠️  未找到要更新的关系: source=%s, target=%s",
                    source_entity_id,
                    target_entity_id,
                )
                return None

            for key, value in update_data.items():
                setattr(existing_doc, key, value)
            await existing_doc.save(session=session)
            logger.debug(
                "✅ 根据实体ID更新关系成功: %s -> %s",
                source_entity_id,
                target_entity_id,
            )
            return existing_doc
        except Exception as e:
            logger.error("❌ 根据实体ID更新关系失败: %s", e)
            return None

    async def delete_by_entity_ids(
        self,
        source_entity_id: str,
        target_entity_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> bool:
        """根据实体ID删除关系"""
        try:
            result = await self.model.find_one(
                {
                    "source_entity_id": source_entity_id,
                    "target_entity_id": target_entity_id,
                },
                session=session,
            )
            if not result:
                logger.warning(
                    "⚠️  未找到要删除的关系: source=%s, target=%s",
                    source_entity_id,
                    target_entity_id,
                )
                return False

            await result.delete(session=session)
            logger.info(
                "✅ 根据实体ID删除关系成功: %s -> %s",
                source_entity_id,
                target_entity_id,
            )
            return True
        except Exception as e:
            logger.error("❌ 根据实体ID删除关系失败: %s", e)
            return False

    # ==================== 批量操作 ====================

    async def get_relationships_by_entity(
        self,
        entity_id: str,
        limit: int = 100,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[Relationship]:
        """获取与指定实体相关的所有关系（作为源或目标）"""
        try:
            results = (
                await self.model.find(
                    {
                        "$or": [
                            {"source_entity_id": entity_id},
                            {"target_entity_id": entity_id},
                        ]
                    },
                    session=session,
                )
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ 获取实体相关关系成功: entity=%s, count=%d", entity_id, len(results)
            )
            return results
        except Exception as e:
            logger.error("❌ 获取实体相关关系失败: %s", e)
            return []

    # ==================== 统计方法 ====================

    async def count_by_entity(
        self, entity_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """统计与指定实体相关的关系数量"""
        try:
            count = await self.model.find(
                {
                    "$or": [
                        {"source_entity_id": entity_id},
                        {"target_entity_id": entity_id},
                    ]
                },
                session=session,
            ).count()
            logger.debug(
                "✅ 统计实体关系数量成功: entity=%s, count=%d", entity_id, count
            )
            return count
        except Exception as e:
            logger.error("❌ 统计实体关系数量失败: %s", e)
            return 0

    async def count_all(
        self, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """统计所有关系数量"""
        try:
            count = await self.model.count()
            logger.debug("✅ 统计所有关系数量成功: count=%d", count)
            return count
        except Exception as e:
            logger.error("❌ 统计所有关系数量失败: %s", e)
            return 0
