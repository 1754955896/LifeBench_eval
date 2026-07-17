from typing import List, Optional, Dict, Any, Tuple, Union
from motor.motor_asyncio import AsyncIOMotorClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.core_memory import CoreMemory

logger = get_logger(__name__)


@repository("core_memory_raw_repository", primary=True)
class CoreMemoryRawRepository(BaseRepository[CoreMemory]):
    """
    核心记忆原始数据仓库

    提供核心记忆的 CRUD 操作和查询功能。
    单个文档包含 BaseMemory 和 Profile 两种记忆类型的数据。
    （Preference 相关字段已合并到 Profile 中）
    """

    def __init__(self):
        super().__init__(CoreMemory)

    # ==================== 版本管理方法 ====================

    async def ensure_latest(
        self, user_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        确保指定用户的最新版本标记正确

        根据user_id找到最新的version，将其is_latest设为True，其他版本设为False。
        这是一个幂等操作，可以安全地重复调用。

        Args:
            user_id: 用户ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否成功更新
        """
        try:
            # 只查询最新的一条记录（优化性能）
            latest_version = await self.model.find_one(
                {"user_id": user_id}, sort=[("version", -1)], session=session
            )

            if not latest_version:
                logger.debug("ℹ️  未找到需要更新的核心记忆: user_id=%s", user_id)
                return True

            # 批量更新：将所有旧版本的is_latest设为False
            await self.model.find(
                {"user_id": user_id, "version": {"$ne": latest_version.version}},
                session=session,
            ).update_many({"$set": {"is_latest": False}})

            # 更新最新版本的is_latest为True
            if latest_version.is_latest != True:
                latest_version.is_latest = True
                await latest_version.save(session=session)
                logger.debug(
                    "✅ 设置最新版本标记: user_id=%s, version=%s",
                    user_id,
                    latest_version.version,
                )

            return True
        except Exception as e:
            logger.error("❌ 确保最新版本标记失败: user_id=%s, error=%s", user_id, e)
            return False

    # ==================== 基础 CRUD 方法 ====================

    async def get_by_user_id(
        self,
        user_id: str,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Union[Optional[CoreMemory], List[CoreMemory]]:
        """
        根据用户ID获取核心记忆

        Args:
            user_id: 用户ID
            version_range: 版本范围 (start, end)，左闭右闭区间 [start, end]。
                          如果不传或为None，则获取最新版本（按version倒序）
                          如果传入，则返回该范围内的所有版本
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            如果version_range为None，返回单个CoreMemory或None
            如果version_range不为None，返回List[CoreMemory]
        """
        try:
            query_filter = {"user_id": user_id}

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

            # 如果没有指定版本范围，获取最新版本（单个结果）
            if version_range is None:
                result = await self.model.find_one(
                    query_filter,
                    sort=[("version", -1)],  # 按版本倒序，获取最新的
                    session=session,
                )
                if result:
                    logger.debug(
                        "✅ 根据用户ID获取核心记忆成功: %s, version=%s",
                        user_id,
                        result.version,
                    )
                else:
                    logger.debug("ℹ️  未找到核心记忆: user_id=%s", user_id)
                return result
            else:
                # 如果指定了版本范围，获取所有匹配的版本
                results = await self.model.find(
                    query_filter, sort=[("version", -1)], session=session  # 按版本倒序
                ).to_list()
                logger.debug(
                    "✅ 根据用户ID获取核心记忆版本成功: %s, version_range=%s, 找到 %d 条记录",
                    user_id,
                    version_range,
                    len(results),
                )
                return results
        except Exception as e:
            logger.error("❌ 根据用户ID获取核心记忆失败: %s", e)
            return None if version_range is None else []

    async def update_by_user_id(
        self,
        user_id: str,
        update_data: Dict[str, Any],
        version: Optional[str] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[CoreMemory]:
        """
        根据用户ID更新核心记忆

        Args:
            user_id: 用户ID
            update_data: 更新数据
            version: 可选的版本号，如果指定则更新特定版本，否则更新最新版本
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新后的 CoreMemory 或 None
        """
        try:
            # 查找要更新的文档
            if version is not None:
                # 更新特定版本
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "version": version}, session=session
                )
            else:
                # 更新最新版本
                existing_doc = await self.model.find_one(
                    {"user_id": user_id}, sort=[("version", -1)], session=session
                )

            if not existing_doc:
                logger.warning(
                    "⚠️  未找到要更新的核心记忆: user_id=%s, version=%s",
                    user_id,
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
                "✅ 根据用户ID更新核心记忆成功: user_id=%s, version=%s",
                user_id,
                existing_doc.version,
            )

            return existing_doc
        except Exception as e:
            logger.error("❌ 根据用户ID更新核心记忆失败: %s", e)
            return None

    async def delete_by_user_id(
        self,
        user_id: str,
        version: Optional[str] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> bool:
        """
        根据用户ID删除核心记忆

        Args:
            user_id: 用户ID
            version: 可选的版本号，如果指定则只删除特定版本，否则删除所有版本
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否删除成功
        """
        try:
            query_filter = {"user_id": user_id}
            if version is not None:
                query_filter["version"] = version

            if version is not None:
                # 删除特定版本 - 直接删除并检查删除数量
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ 根据用户ID和版本删除核心记忆成功: user_id=%s, version=%s",
                        user_id,
                        version,
                    )
                    # 删除后确保最新版本标记正确
                    await self.ensure_latest(user_id, session)
                else:
                    logger.warning(
                        "⚠️  未找到要删除的核心记忆: user_id=%s, version=%s",
                        user_id,
                        version,
                    )
            else:
                # 删除所有版本
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ 根据用户ID删除所有核心记忆成功: user_id=%s, 删除 %d 条",
                        user_id,
                        deleted_count,
                    )
                else:
                    logger.warning("⚠️  未找到要删除的核心记忆: user_id=%s", user_id)

            return success
        except Exception as e:
            logger.error("❌ 根据用户ID删除核心记忆失败: %s", e)
            return False

    async def upsert_by_user_id(
        self,
        user_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[CoreMemory]:
        """
        根据用户ID更新或插入核心记忆

        如果update_data中包含version字段：
        - 如果该version已存在，则更新该版本
        - 如果该version不存在，则创建新版本（必须提供version）
        如果update_data中不包含version字段：
        - 获取最新版本并更新，如果不存在则报错（创建时必须提供version）

        Args:
            user_id: 用户ID
            update_data: 要更新的数据（创建新版本时必须包含version字段）
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新或创建的核心记忆记录
        """
        try:
            version = update_data.get("version")

            if version is not None:
                # 如果指定了版本，查找特定版本
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "version": version}, session=session
                )
            else:
                # 如果未指定版本，查找最新版本
                existing_doc = await self.model.find_one(
                    {"user_id": user_id}, sort=[("version", -1)], session=session
                )

            if existing_doc:
                # 更新现有记录
                for key, value in update_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "✅ 更新现有核心记忆成功: user_id=%s, version=%s",
                    user_id,
                    existing_doc.version,
                )

                # 如果更新了版本，需要确保最新标记正确
                if version is not None:
                    await self.ensure_latest(user_id, session)

                return existing_doc
            else:
                # 创建新记录时必须提供version
                if version is None:
                    logger.error(
                        "❌ 创建新核心记忆时必须提供version字段: user_id=%s", user_id
                    )
                    raise ValueError(
                        f"创建新核心记忆时必须提供version字段: user_id={user_id}"
                    )

                # 创建新记录
                new_doc = CoreMemory(user_id=user_id, **update_data)
                await new_doc.create(session=session)
                logger.info(
                    "✅ 创建新核心记忆成功: user_id=%s, version=%s",
                    user_id,
                    new_doc.version,
                )

                # 创建后确保最新版本标记正确
                await self.ensure_latest(user_id, session)

                return new_doc
        except ValueError:
            # 重新抛出ValueError，不要被Exception捕获
            raise
        except Exception as e:
            logger.error("❌ 更新或创建核心记忆失败: %s", e)
            return None

    # ==================== 字段提取方法 ====================

    def get_base(self, memory: CoreMemory) -> Dict[str, Any]:
        """
        获取基础信息

        Args:
            memory: CoreMemory 实例

        Returns:
            基础信息字典
        """
        return {
            "user_name": memory.user_name,
            "gender": memory.gender,
            "position": memory.position,
            "supervisor_user_id": memory.supervisor_user_id,
            "team_members": memory.team_members,
            "okr": memory.okr,
            "base_location": memory.base_location,
            "hiredate": memory.hiredate,
            "age": memory.age,
            "department": memory.department,
        }

    def get_profile(self, memory: CoreMemory) -> Dict[str, Any]:
        """
        获取个人档案

        Args:
            memory: CoreMemory 实例

        Returns:
            个人档案字典
        """
        return {
            "hard_skills": memory.hard_skills,
            "soft_skills": memory.soft_skills,
            "personality": memory.personality,
            "projects_participated": memory.projects_participated,
            "user_goal": memory.user_goal,
            "work_responsibility": memory.work_responsibility,
            "working_habit_preference": memory.working_habit_preference,
            "interests": getattr(memory, "interests", None),
            "tendency": memory.tendency,
        }

    async def find_by_user_ids(
        self,
        user_ids: List[str],
        only_latest: bool = True,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[CoreMemory]:
        """
        根据用户ID列表批量获取核心记忆

        Args:
            user_ids: 用户ID列表
            only_latest: 是否只获取最新版本，默认为True。批量查询时使用is_latest字段过滤
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            CoreMemory 列表
        """
        try:
            if not user_ids:
                return []

            query_filter = {"user_id": {"$in": user_ids}}

            # 批量查询时，使用is_latest字段过滤最新版本
            if only_latest:
                query_filter["is_latest"] = True

            query = self.model.find(query_filter, session=session)

            results = await query.to_list()
            logger.debug(
                "✅ 根据用户ID列表获取核心记忆成功: %d 个用户ID, only_latest=%s, 找到 %d 条记录",
                len(user_ids),
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户ID列表获取核心记忆失败: %s", e)
            return []


# 导出
__all__ = ["CoreMemoryRawRepository"]
