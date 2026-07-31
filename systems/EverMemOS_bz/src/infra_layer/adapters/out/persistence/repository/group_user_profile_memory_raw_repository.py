"""
GroupUserProfileMemory 原生 CRUD 仓库

基于 Beanie ODM 的 GroupUserProfileMemory 原生数据访问层，提供完整的 CRUD 操作。
不依赖领域层接口，直接操作 GroupUserProfileMemory 文档模型。
支持基于 user_id 和 group_id 的联合查询和操作。
"""

from typing import List, Optional, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository

from infra_layer.adapters.out.persistence.document.memory.group_user_profile_memory import (
    GroupUserProfileMemory,
)

logger = get_logger(__name__)


@repository("group_user_profile_memory_raw_repository", primary=True)
class GroupUserProfileMemoryRawRepository(BaseRepository[GroupUserProfileMemory]):
    """
    GroupUserProfileMemory 原生 CRUD 仓库

    提供对 GroupUserProfileMemory 文档的直接数据库操作，包括：
    - 基本 CRUD 操作（继承自 BaseRepository）
    - 基于 user_id 和 group_id 的联合查询
    - 基于 user_id 或 group_id 的单独查询
    - 批量查询和操作
    - 档案相关的专门方法
    - 事务管理（继承自 BaseRepository）
    """

    def __init__(self):
        """初始化仓库"""
        super().__init__(GroupUserProfileMemory)

    # ==================== 版本管理方法 ====================

    async def ensure_latest(
        self,
        user_id: str,
        group_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> bool:
        """
        确保指定用户在指定群组中的最新版本标记正确

        根据user_id和group_id找到最新的version，将其is_latest设为True，其他版本设为False。
        这是一个幂等操作，可以安全地重复调用。

        Args:
            user_id: 用户ID
            group_id: 群组ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否成功更新
        """
        try:
            # 只查询最新的一条记录（优化性能）
            latest_version = await self.model.find_one(
                {"user_id": user_id, "group_id": group_id},
                sort=[("version", -1)],
                session=session,
            )

            if not latest_version:
                logger.debug(
                    "ℹ️  未找到需要更新的群组用户档案: user_id=%s, group_id=%s",
                    user_id,
                    group_id,
                )
                return True

            # 批量更新：将所有旧版本的is_latest设为False
            await self.model.find(
                {
                    "user_id": user_id,
                    "group_id": group_id,
                    "version": {"$ne": latest_version.version},
                },
                session=session,
            ).update_many({"$set": {"is_latest": False}})

            # 更新最新版本的is_latest为True
            if latest_version.is_latest != True:
                latest_version.is_latest = True
                await latest_version.save(session=session)
                logger.debug(
                    "✅ 设置最新版本标记: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    latest_version.version,
                )

            return True
        except Exception as e:
            logger.error(
                "❌ 确保最新版本标记失败: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            return False

    # ==================== 基于联合键的 CRUD 方法 ====================

    async def get_by_user_group(
        self,
        user_id: str,
        group_id: str,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[GroupUserProfileMemory]:
        """
        根据用户ID和群组ID获取群组用户档案记忆

        Args:
            user_id: 用户ID
            group_id: 群组ID
            version_range: 版本范围 (start, end)，左闭右闭区间 [start, end]。
                          如果不传或为None，则获取最新版本（按version倒序）
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            GroupUserProfileMemory 实例或 None
        """
        try:
            query_filter = {"user_id": user_id, "group_id": group_id}

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
                    "✅ 根据用户ID和群组ID获取群组用户档案成功: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    result.version,
                )
            else:
                logger.debug(
                    "ℹ️ 未找到群组用户档案: user_id=%s, group_id=%s", user_id, group_id
                )
            return result
        except Exception as e:
            logger.error("❌ 根据用户ID和群组ID获取群组用户档案失败: %s", e)
            return None

    async def batch_get_by_user_groups(
        self,
        user_group_pairs: List[Tuple[str, str]],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Dict[Tuple[str, str], Optional[GroupUserProfileMemory]]:
        """
        批量根据用户ID和群组ID获取群组用户档案记忆

        Args:
            user_group_pairs: (user_id, group_id) 元组列表
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            Dict[(user_id, group_id), GroupUserProfileMemory]: 映射字典
        """
        try:
            if not user_group_pairs:
                return {}

            # 去重
            unique_pairs = list(set(user_group_pairs))
            logger.debug(
                "批量获取群组用户档案: 总数 %d (去重前: %d)",
                len(unique_pairs),
                len(user_group_pairs),
            )

            # 构造查询条件：查询所有 (user_id, group_id) 对的最新版本
            # 使用聚合管道来实现批量查询最新版本
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"user_id": user_id, "group_id": group_id}
                            for user_id, group_id in unique_pairs
                        ]
                    }
                },
                # 按 user_id, group_id, version 分组，获取每组的最新版本
                {"$sort": {"user_id": 1, "group_id": 1, "version": -1}},
                {
                    "$group": {
                        "_id": {"user_id": "$user_id", "group_id": "$group_id"},
                        "doc": {"$first": "$$ROOT"},
                    }
                },
                {"$replaceRoot": {"newRoot": "$doc"}},
            ]

            # 执行聚合查询
            results = (
                await self.model.get_pymongo_collection()
                .aggregate(pipeline, session=session)
                .to_list(length=None)
            )

            # 构建结果字典
            result_dict = {}
            for doc in results:
                if not doc:
                    continue
                memory = GroupUserProfileMemory.model_validate(doc)
                key = (memory.user_id, memory.group_id)
                result_dict[key] = memory

            # 填充未找到的记录为 None
            for pair in unique_pairs:
                if pair not in result_dict:
                    result_dict[pair] = None

            logger.debug(
                "✅ 批量获取群组用户档案完成: 成功获取 %d 个",
                len([v for v in result_dict.values() if v is not None]),
            )

            return result_dict
        except Exception as e:
            logger.error("❌ 批量获取群组用户档案失败: %s", e)
            return {}

    async def update_by_user_group(
        self,
        user_id: str,
        group_id: str,
        update_data: Dict[str, Any],
        version: Optional[str] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[GroupUserProfileMemory]:
        """
        根据用户ID和群组ID更新群组用户档案记忆

        Args:
            user_id: 用户ID
            group_id: 群组ID
            update_data: 更新数据
            version: 可选的版本号，如果指定则更新特定版本，否则更新最新版本
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新后的 GroupUserProfileMemory 或 None
        """
        try:
            # 查找要更新的文档
            if version is not None:
                # 更新特定版本
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "group_id": group_id, "version": version},
                    session=session,
                )
            else:
                # 更新最新版本
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "group_id": group_id},
                    sort=[("version", -1)],
                    session=session,
                )

            if not existing_doc:
                logger.warning(
                    "⚠️ 未找到要更新的群组用户档案: user_id=%s, group_id=%s, version=%s",
                    user_id,
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
                "✅ 根据用户ID和群组ID更新群组用户档案成功: user_id=%s, group_id=%s, version=%s",
                user_id,
                group_id,
                existing_doc.version,
            )

            return existing_doc
        except Exception as e:
            logger.error("❌ 根据用户ID和群组ID更新群组用户档案失败: %s", e)
            return None

    async def delete_by_user_group(
        self,
        user_id: str,
        group_id: str,
        version: Optional[str] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> bool:
        """
        根据用户ID和群组ID删除群组用户档案记忆

        Args:
            user_id: 用户ID
            group_id: 群组ID
            version: 可选的版本号，如果指定则只删除特定版本，否则删除所有版本
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否删除成功
        """
        try:
            query_filter = {"user_id": user_id, "group_id": group_id}
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
                        "✅ 根据用户ID、群组ID和版本删除群组用户档案成功: user_id=%s, group_id=%s, version=%s",
                        user_id,
                        group_id,
                        version,
                    )
                    # 删除后确保最新版本标记正确
                    await self.ensure_latest(user_id, group_id, session)
                else:
                    logger.warning(
                        "⚠️ 未找到要删除的群组用户档案: user_id=%s, group_id=%s, version=%s",
                        user_id,
                        group_id,
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
                        "✅ 根据用户ID和群组ID删除所有群组用户档案成功: user_id=%s, group_id=%s, 删除 %d 条",
                        user_id,
                        group_id,
                        deleted_count,
                    )
                else:
                    logger.warning(
                        "⚠️ 未找到要删除的群组用户档案: user_id=%s, group_id=%s",
                        user_id,
                        group_id,
                    )

            return success
        except Exception as e:
            logger.error("❌ 根据用户ID和群组ID删除群组用户档案失败: %s", e)
            return False

    async def upsert_by_user_group(
        self,
        user_id: str,
        group_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[GroupUserProfileMemory]:
        """
        根据用户ID和群组ID更新或插入群组用户档案记忆

        如果update_data中包含version字段：
        - 如果该version已存在，则更新该版本
        - 如果该version不存在，则创建新版本（必须提供version）
        如果update_data中不包含version字段：
        - 获取最新版本并更新，如果不存在则报错（创建时必须提供version）

        Args:
            user_id: 用户ID
            group_id: 群组ID
            update_data: 要更新的数据（创建新版本时必须包含version字段）
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新或创建的群组用户档案记录
        """
        try:
            version = update_data.get("version")

            if version is not None:
                # 如果指定了版本，查找特定版本
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "group_id": group_id, "version": version},
                    session=session,
                )
            else:
                # 如果未指定版本，查找最新版本
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "group_id": group_id},
                    sort=[("version", -1)],
                    session=session,
                )

            if existing_doc:
                # 更新现有记录
                for key, value in update_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "✅ 更新现有群组用户档案成功: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    existing_doc.version,
                )

                # 如果更新了版本，需要确保最新标记正确
                if version is not None:
                    await self.ensure_latest(user_id, group_id, session)

                return existing_doc
            else:
                # 创建新记录时必须提供version
                if version is None:
                    logger.error(
                        "❌ 创建新群组用户档案时必须提供version字段: user_id=%s, group_id=%s",
                        user_id,
                        group_id,
                    )
                    raise ValueError(
                        f"创建新群组用户档案时必须提供version字段: user_id={user_id}, group_id={group_id}"
                    )

                # 创建新记录
                new_doc = GroupUserProfileMemory(
                    user_id=user_id, group_id=group_id, **update_data
                )
                await new_doc.create(session=session)
                logger.info(
                    "✅ 创建新群组用户档案成功: user_id=%s, group_id=%s, version=%s",
                    user_id,
                    group_id,
                    new_doc.version,
                )

                # 创建后确保最新版本标记正确
                await self.ensure_latest(user_id, group_id, session)

                return new_doc
        except ValueError:
            # 重新抛出ValueError，不要被Exception捕获
            raise
        except Exception as e:
            logger.error("❌ 更新或创建群组用户档案失败: %s", e)
            return None

    # ==================== 基于单个键的查询方法 ====================

    async def get_by_user_id(
        self,
        user_id: str,
        only_latest: bool = True,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[GroupUserProfileMemory]:
        """
        根据用户ID获取该用户在所有群组中的档案记忆

        Args:
            user_id: 用户ID
            only_latest: 是否只获取最新版本，默认为True。批量查询时使用is_latest字段过滤
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            GroupUserProfileMemory 列表
        """
        try:
            query_filter = {"user_id": user_id}

            # 批量查询时，使用is_latest字段过滤最新版本
            if only_latest:
                query_filter["is_latest"] = True

            results = await self.model.find(query_filter, session=session).to_list()
            logger.debug(
                "✅ 根据用户ID获取群组用户档案成功: user_id=%s, only_latest=%s, 找到 %d 条记录",
                user_id,
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户ID获取群组用户档案失败: %s", e)
            return []

    async def get_by_group_id(
        self,
        group_id: str,
        only_latest: bool = True,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[GroupUserProfileMemory]:
        """
        根据群组ID获取该群组中所有用户的档案记忆

        Args:
            group_id: 群组ID
            only_latest: 是否只获取最新版本，默认为True。批量查询时使用is_latest字段过滤
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            GroupUserProfileMemory 列表
        """
        try:
            query_filter = {"group_id": group_id}

            # 批量查询时，使用is_latest字段过滤最新版本
            if only_latest:
                query_filter["is_latest"] = True

            results = await self.model.find(query_filter, session=session).to_list()
            logger.debug(
                "✅ 根据群组ID获取群组用户档案成功: group_id=%s, only_latest=%s, 找到 %d 条记录",
                group_id,
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据群组ID获取群组用户档案失败: %s", e)
            return []

    # ==================== 批量查询方法 ====================

    async def get_by_user_ids(
        self,
        user_ids: List[str],
        group_id: Optional[str] = None,
        only_latest: bool = True,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[GroupUserProfileMemory]:
        """
        根据用户ID列表批量获取群组用户档案记忆

        Args:
            user_ids: 用户ID列表
            group_id: 可选的群组ID，如果提供则只查询该群组中的用户档案
            only_latest: 是否只获取最新版本，默认为True。批量查询时使用is_latest字段过滤
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            GroupUserProfileMemory 列表
        """
        try:
            if not user_ids:
                return []

            # 构建查询条件
            query_filter = {"user_id": {"$in": user_ids}}
            if group_id:
                query_filter["group_id"] = group_id

            # 批量查询时，使用is_latest字段过滤最新版本
            if only_latest:
                query_filter["is_latest"] = True

            results = await self.model.find(query_filter, session=session).to_list()

            logger.debug(
                "✅ 根据用户ID列表获取群组用户档案成功: %d 个用户ID, group_id=%s, only_latest=%s, 找到 %d 条记录",
                len(user_ids),
                group_id,
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户ID列表获取群组用户档案失败: %s", e)
            return []

    async def get_by_group_ids(
        self,
        group_ids: List[str],
        user_id: Optional[str] = None,
        only_latest: bool = True,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[GroupUserProfileMemory]:
        """
        根据群组ID列表批量获取群组用户档案记忆

        Args:
            group_ids: 群组ID列表
            user_id: 可选的用户ID，如果提供则只查询该用户在这些群组中的档案
            only_latest: 是否只获取最新版本，默认为True。批量查询时使用is_latest字段过滤
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            GroupUserProfileMemory 列表
        """
        try:
            if not group_ids:
                return []

            # 构建查询条件
            query_filter = {"group_id": {"$in": group_ids}}
            if user_id:
                query_filter["user_id"] = user_id

            # 批量查询时，使用is_latest字段过滤最新版本
            if only_latest:
                query_filter["is_latest"] = True

            results = await self.model.find(query_filter, session=session).to_list()

            logger.debug(
                "✅ 根据群组ID列表获取群组用户档案成功: %d 个群组ID, user_id=%s, only_latest=%s, 找到 %d 条记录",
                len(group_ids),
                user_id,
                only_latest,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据群组ID列表获取群组用户档案失败: %s", e)
            return []

    # ==================== 档案相关的专门方法 ====================

    def get_profile(self, memory: GroupUserProfileMemory) -> Dict[str, Any]:
        """
        获取个人档案

        Args:
            memory: GroupUserProfileMemory 实例

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
            "interests": memory.interests,
            "tendency": memory.tendency,
        }

    # ==================== 删除相关方法 ====================

    async def delete_by_user_id(
        self, user_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """
        删除用户在所有群组中的档案记忆

        Args:
            user_id: 用户ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            删除的记录数量
        """
        try:
            result = await self.model.find(
                {"user_id": user_id}, session=session
            ).delete()
            deleted_count = (
                result.deleted_count if hasattr(result, 'deleted_count') else 0
            )
            logger.debug(
                "✅ 根据用户ID删除群组用户档案成功: user_id=%s, 删除 %d 条记录",
                user_id,
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            logger.error("❌ 根据用户ID删除群组用户档案失败: %s", e)
            return 0

    async def delete_by_group_id(
        self, group_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """
        删除群组中所有用户的档案记忆

        Args:
            group_id: 群组ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            删除的记录数量
        """
        try:
            result = await self.model.find(
                {"group_id": group_id}, session=session
            ).delete()
            deleted_count = (
                result.deleted_count if hasattr(result, 'deleted_count') else 0
            )
            logger.debug(
                "✅ 根据群组ID删除群组用户档案成功: group_id=%s, 删除 %d 条记录",
                group_id,
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            logger.error("❌ 根据群组ID删除群组用户档案失败: %s", e)
            return 0


# 导出
__all__ = ["GroupUserProfileMemoryRawRepository"]
