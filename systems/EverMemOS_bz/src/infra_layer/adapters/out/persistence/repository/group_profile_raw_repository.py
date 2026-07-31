from typing import List, Optional, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.group_profile import (
    GroupProfile,
)

logger = get_logger(__name__)


@repository("group_profile_raw_repository", primary=True)
class GroupProfileRawRepository(BaseRepository[GroupProfile]):
    """
    群组记忆原始数据仓库

    提供群组记忆的 CRUD 操作和查询功能。
    支持群组信息、角色定义、用户标签和近期话题的管理。
    """

    def __init__(self):
        super().__init__(GroupProfile)

    # ==================== 版本管理方法 ====================

    async def ensure_latest(
        self, group_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        确保指定群组的最新版本标记正确

        根据group_id找到最新的version，将其is_latest设为True，其他版本设为False。
        这是一个幂等操作，可以安全地重复调用。

        Args:
            group_id: 群组ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否成功更新
        """
        try:
            # 只查询最新的一条记录（优化性能）
            latest_version = await self.model.find_one(
                {"group_id": group_id}, sort=[("version", -1)], session=session
            )

            if not latest_version:
                logger.debug("ℹ️  未找到需要更新的群组记忆: group_id=%s", group_id)
                return True

            # 批量更新：将所有旧版本的is_latest设为False
            await self.model.find(
                {"group_id": group_id, "version": {"$ne": latest_version.version}},
                session=session,
            ).update_many({"$set": {"is_latest": False}})

            # 更新最新版本的is_latest为True
            if latest_version.is_latest != True:
                latest_version.is_latest = True
                await latest_version.save(session=session)
                logger.debug(
                    "✅ 设置最新版本标记: group_id=%s, version=%s",
                    group_id,
                    latest_version.version,
                )

            return True
        except Exception as e:
            logger.error("❌ 确保最新版本标记失败: group_id=%s, error=%s", group_id, e)
            return False

    # ==================== 基础 CRUD 方法 ====================

    async def get_by_group_id(
        self,
        group_id: str,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[GroupProfile]:
        """
        根据群组ID获取群组记忆

        Args:
            group_id: 群组ID
            version_range: 版本范围 (start, end)，左闭右闭区间 [start, end]。
                          如果不传或为None，则获取最新版本（按version倒序）
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            GroupProfile 或 None
        """
        try:
            query_filter = {"group_id": group_id}

            # 处理版本范围查询
            if version_range:
                start_version, end_version = version_range
                version_filter = {}
                if start_version is not None:
                    version_filter["$gte"] = start_version
                if end_version is not None:
                    version_filter["$lte"] = end_version
                if version_filter:
                    query_filter["version"] = version_filter

            # 按版本倒序，获取最新版本
            result = await self.model.find_one(
                query_filter, sort=[("version", -1)], session=session
            )

            if result:
                logger.debug(
                    "✅ 根据群组ID获取群组记忆成功: %s, version=%s",
                    group_id,
                    result.version,
                )
            else:
                logger.debug("ℹ️  未找到群组记忆: group_id=%s", group_id)
            return result
        except Exception as e:
            logger.error("❌ 根据群组ID获取群组记忆失败: %s", e)
            return None

    async def update_by_group_id(
        self,
        group_id: str,
        update_data: Dict[str, Any],
        version: Optional[str] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[GroupProfile]:
        """
        根据群组ID更新群组记忆

        Args:
            group_id: 群组ID
            update_data: 更新数据
            version: 可选的版本号，如果指定则更新特定版本，否则更新最新版本
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新后的 GroupProfile 或 None
        """
        try:
            # 查找要更新的文档
            if version is not None:
                # 更新特定版本
                existing_doc = await self.model.find_one(
                    {"group_id": group_id, "version": version}, session=session
                )
            else:
                # 更新最新版本
                existing_doc = await self.model.find_one(
                    {"group_id": group_id}, sort=[("version", -1)], session=session
                )

            if not existing_doc:
                logger.warning(
                    "⚠️  未找到要更新的群组记忆: group_id=%s, version=%s",
                    group_id,
                    version,
                )
                return None

            # 更新文档
            for key, value in update_data.items():
                if hasattr(existing_doc, key):
                    setattr(existing_doc, key, value)

            # 保存更新后的文档
            await existing_doc.save(session=session)
            logger.debug(
                "✅ 根据群组ID更新群组记忆成功: group_id=%s, version=%s",
                group_id,
                existing_doc.version,
            )

            return existing_doc
        except Exception as e:
            logger.error("❌ 根据群组ID更新群组记忆失败: %s", e)
            return None

    async def delete_by_group_id(
        self,
        group_id: str,
        version: Optional[str] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> bool:
        """
        根据群组ID删除群组记忆

        Args:
            group_id: 群组ID
            version: 可选的版本号，如果指定则只删除特定版本，否则删除所有版本
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否删除成功
        """
        try:
            query_filter = {"group_id": group_id}
            if version is not None:
                query_filter["version"] = version

            if version is not None:
                # 删除特定版本
                result = await self.model.find_one(query_filter, session=session)
                if not result:
                    logger.warning(
                        "⚠️  未找到要删除的群组记忆: group_id=%s, version=%s",
                        group_id,
                        version,
                    )
                    return False

                await result.delete(session=session)
                logger.debug(
                    "✅ 根据群组ID和版本删除群组记忆成功: group_id=%s, version=%s",
                    group_id,
                    version,
                )

                # 删除后确保最新版本标记正确
                await self.ensure_latest(group_id, session)
                return True
            else:
                # 删除所有版本
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ 根据群组ID删除所有群组记忆成功: group_id=%s, 删除 %d 条",
                        group_id,
                        deleted_count,
                    )
                else:
                    logger.warning("⚠️  未找到要删除的群组记忆: group_id=%s", group_id)

                return success
        except Exception as e:
            logger.error("❌ 根据群组ID删除群组记忆失败: %s", e)
            return False

    async def upsert_by_group_id(
        self,
        group_id: str,
        update_data: Dict[str, Any],
        timestamp: Optional[int] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[GroupProfile]:
        """
        根据群组ID更新或插入群组记忆

        如果update_data中包含version字段：
        - 如果该version已存在，则更新该版本
        - 如果该version不存在，则创建新版本（必须提供version）
        如果update_data中不包含version字段：
        - 获取最新版本并更新，如果不存在则报错（创建时必须提供version）

        Args:
            group_id: 群组ID
            update_data: 要更新的数据（创建新版本时必须包含version字段）
            timestamp: 时间戳，创建新记录时必需
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新或创建的群组记忆记录
        """
        try:
            version = update_data.get("version")

            if version is not None:
                # 如果指定了版本，查找特定版本
                existing_doc = await self.model.find_one(
                    {"group_id": group_id, "version": version}, session=session
                )
            else:
                # 如果未指定版本，查找最新版本
                existing_doc = await self.model.find_one(
                    {"group_id": group_id}, sort=[("version", -1)], session=session
                )

            if existing_doc:
                # 更新现有记录
                for key, value in update_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "✅ 更新现有群组记忆成功: group_id=%s, version=%s",
                    group_id,
                    existing_doc.version,
                )

                # 如果更新了版本，需要确保最新标记正确
                if version is not None:
                    await self.ensure_latest(group_id, session)

                return existing_doc
            else:
                # 创建新记录时必须提供version
                if version is None:
                    logger.error(
                        "❌ 创建新群组记忆时必须提供version字段: group_id=%s", group_id
                    )
                    raise ValueError(
                        f"创建新群组记忆时必须提供version字段: group_id={group_id}"
                    )

                # 创建新记录，需要提供 timestamp
                if timestamp is None:
                    from time import time

                    timestamp = int(time() * 1000)  # 毫秒级时间戳

                new_doc = GroupProfile(
                    group_id=group_id, timestamp=timestamp, **update_data
                )
                await new_doc.create(session=session)
                logger.info(
                    "✅ 创建新群组记忆成功: group_id=%s, version=%s",
                    group_id,
                    new_doc.version,
                )

                # 创建后确保最新版本标记正确
                await self.ensure_latest(group_id, session)

                return new_doc
        except ValueError:
            # 重新抛出ValueError，不要被Exception捕获
            raise
        except Exception as e:
            logger.error("❌ 更新或创建群组记忆失败: %s", e)
            return None

    # ==================== 查询方法 ====================

    async def find_by_group_ids(
        self,
        group_ids: List[str],
        only_latest: bool = True,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[GroupProfile]:
        """
        根据群组ID列表批量获取群组记忆

        Args:
            group_ids: 群组ID列表
            only_latest: 是否只获取最新版本，默认为True。批量查询时使用is_latest字段过滤
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            GroupProfile 列表
        """
        try:
            if not group_ids:
                return []

            query_filter = {"group_id": {"$in": group_ids}}

            # 批量查询时，使用is_latest字段过滤最新版本
            if only_latest:
                query_filter["is_latest"] = True

            query = self.model.find(query_filter, session=session)

            results = await query.to_list()
            logger.debug(
                "✅ 根据群组ID列表获取群组记忆成功: %d 个群组ID, only_latest=%s, 找到 %d 条记录",
                len(group_ids),
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据群组ID列表获取群组记忆失败: %s", e)
            return []


__all__ = ["GroupProfileRawRepository"]
