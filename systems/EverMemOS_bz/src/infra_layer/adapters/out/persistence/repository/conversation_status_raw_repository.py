from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClientSession
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
    ConversationStatus,
)
from core.observation.logger import get_logger
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("conversation_status_raw_repository", primary=True)
class ConversationStatusRawRepository(BaseRepository[ConversationStatus]):
    """
    对话状态原始数据仓库

    提供对话状态数据的CRUD操作和查询功能。
    """

    def __init__(self):
        super().__init__(ConversationStatus)

    # ==================== 基础CRUD操作 ====================

    async def get_by_group_id(
        self, group_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> Optional[ConversationStatus]:
        """根据群组ID获取对话状态"""
        try:
            result = await self.model.find_one({"group_id": group_id}, session=session)
            if result:
                logger.debug("✅ 根据群组ID获取对话状态成功: %s", group_id)
            else:
                logger.debug("⚠️  未找到对话状态: group_id=%s", group_id)
            return result
        except Exception as e:
            logger.error("❌ 根据群组ID获取对话状态失败: %s", e)
            return None

    async def delete_by_group_id(
        self, group_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """根据群组ID删除对话状态"""
        try:
            result = await self.model.find_one({"group_id": group_id}, session=session)
            if not result:
                logger.warning("⚠️  未找到要删除的对话状态: group_id=%s", group_id)
                return False

            await result.delete(session=session)
            logger.info("✅ 根据群组ID删除对话状态成功: %s", group_id)
            return True
        except Exception as e:
            logger.error("❌ 根据群组ID删除对话状态失败: %s", e)
            return False

    async def upsert_by_group_id(
        self,
        group_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[ConversationStatus]:
        """根据群组ID更新或插入对话状态

        使用MongoDB原子upsert操作来避免并发竞态条件。
        如果找到对应的记录则更新，否则创建新记录。
        由于group_id是唯一的，conversation_id会自动使用group_id作为值。

        Args:
            group_id: 群组ID（同时也会用作conversation_id）
            update_data: 要更新的数据
            session: MongoDB会话

        Returns:
            更新或创建的对话状态记录
        """
        try:
            # 1. 首先尝试查找现有记录
            existing_doc = await self.model.find_one(
                {"group_id": group_id}, session=session
            )

            if existing_doc:
                # 找到记录，直接更新
                for key, value in update_data.items():
                    setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug("✅ 更新现有对话状态成功: group_id=%s", group_id)
                print(
                    f"[ConversationStatusRawRepository] 更新现有对话状态成功: {existing_doc}"
                )
                return existing_doc

            # 2. 没找到记录，尝试创建新记录
            try:
                new_doc = ConversationStatus(group_id=group_id, **update_data)
                await new_doc.create(session=session)
                logger.info("✅ 创建新对话状态成功: group_id=%s", group_id)
                print(
                    f"[ConversationStatusRawRepository] 创建新对话状态成功: {new_doc}"
                )
                return new_doc

            except Exception as create_error:
                # 3. 创建失败，检查是否是重复键错误（并发情况）
                error_str = str(create_error)
                if "E11000" in error_str and "duplicate key" in error_str:
                    logger.warning(
                        "⚠️  并发创建冲突，重新查找并更新: group_id=%s", group_id
                    )

                    # 重复键错误说明其他线程已经创建了记录，重新查找并更新
                    retry_doc = await self.model.find_one(
                        {"group_id": group_id}, session=session
                    )

                    if retry_doc:
                        # 找到了被其他线程创建的记录，更新它
                        for key, value in update_data.items():
                            setattr(retry_doc, key, value)
                        await retry_doc.save(session=session)
                        logger.debug("✅ 并发冲突后更新成功: group_id=%s", group_id)
                        print(
                            f"[ConversationStatusRawRepository] 并发冲突后更新成功: {retry_doc}"
                        )
                        return retry_doc
                    else:
                        logger.error(
                            "❌ 并发冲突后仍无法找到记录: group_id=%s", group_id
                        )
                        return None
                else:
                    # 其他类型的创建错误，直接抛出
                    raise create_error

        except Exception as e:
            logger.error("❌ 更新或创建对话状态失败: %s", e)
            return None

    # ==================== 统计方法 ====================

    async def count_by_group_id(
        self, group_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """统计指定群组的对话状态数量"""
        try:
            count = await self.model.find(
                {"group_id": group_id}, session=session
            ).count()
            logger.debug(
                "✅ 统计对话状态数量成功: group_id=%s, count=%d", group_id, count
            )
            return count
        except Exception as e:
            logger.error("❌ 统计对话状态数量失败: %s", e)
            return 0
