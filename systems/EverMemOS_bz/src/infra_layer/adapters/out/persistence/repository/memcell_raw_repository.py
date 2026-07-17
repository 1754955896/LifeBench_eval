"""
MemCell 原生 CRUD 仓库

基于 Beanie ODM 的 MemCell 原生数据访问层，提供完整的 CRUD 操作。
不依赖领域层接口，直接操作 MemCell 文档模型。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Type
from bson import ObjectId
from pydantic import BaseModel
from beanie.operators import And, GTE, LT, Eq, RegEx, Or
from motor.motor_asyncio import AsyncIOMotorClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository

from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell,
    DataTypeEnum,
)

logger = get_logger(__name__)


@repository("memcell_raw_repository", primary=True)
class MemCellRawRepository(BaseRepository[MemCell]):
    """
    MemCell 原生 CRUD 仓库

    提供对 MemCell 文档的直接数据库操作，包括：
    - 基本 CRUD 操作（继承自 BaseRepository）
    - 复合查询和筛选
    - 批量操作
    - 统计和聚合查询
    - 事务管理（继承自 BaseRepository）
    """

    def __init__(self):
        """初始化仓库"""
        super().__init__(MemCell)

    async def get_by_event_id(self, event_id: str) -> Optional[MemCell]:
        """
        根据 event_id 获取 MemCell

        Args:
            event_id: 事件 ID

        Returns:
            MemCell 实例或 None
        """
        try:
            result = await self.model.find_one({"_id": ObjectId(event_id)})
            if result:
                logger.debug("✅ 根据 event_id 获取 MemCell 成功: %s", event_id)
            else:
                logger.debug("⚠️  未找到 MemCell: event_id=%s", event_id)
            return result
        except Exception as e:
            logger.error("❌ 根据 event_id 获取 MemCell 失败: %s", e)
            return None

    async def get_by_event_ids(
        self, event_ids: List[str], projection_model: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        根据 event_id 列表批量获取 MemCell

        Args:
            event_ids: 事件 ID 列表
            projection_model: Pydantic 投影模型类，用于指定返回的字段
                             例如：传入一个只包含部分字段的 Pydantic 模型
                             None 表示返回完整的 MemCell 对象

        Returns:
            Dict[event_id, MemCell | ProjectionModel]：event_id 到 MemCell（或投影模型）的映射字典
            未找到的 event_id 不会出现在字典中
        """
        try:
            if not event_ids:
                logger.debug("⚠️  event_ids 列表为空，返回空字典")
                return {}

            # 将 event_id 列表转换为 ObjectId 列表
            object_ids = []
            valid_event_ids = []  # 保存有效的原始 event_id 字符串
            for event_id in event_ids:
                try:
                    object_ids.append(ObjectId(event_id))
                    valid_event_ids.append(event_id)
                except Exception as e:
                    logger.warning("⚠️  无效的 event_id: %s, 错误: %s", event_id, e)

            if not object_ids:
                logger.debug("⚠️  没有有效的 event_id，返回空字典")
                return {}

            # 构建查询
            query = self.model.find({"_id": {"$in": object_ids}})

            # 应用字段投影
            # 使用 Beanie 的 .project() 方法，传入 projection_model 参数
            if projection_model:
                query = query.project(projection_model=projection_model)

            # 批量查询
            results = await query.to_list()

            # 创建 event_id 到 MemCell（或投影模型）的映射字典
            result_dict = {str(result.id): result for result in results}

            logger.debug(
                "✅ 根据 event_ids 批量获取 MemCell 成功: 请求 %d 个, 找到 %d 个, 投影: %s",
                len(event_ids),
                len(result_dict),
                "有" if projection_model else "无",
            )

            return result_dict

        except Exception as e:
            logger.error("❌ 根据 event_ids 批量获取 MemCell 失败: %s", e)
            return {}

    async def append_memcell(
        self, memcell: MemCell, session: Optional[AsyncIOMotorClientSession] = None
    ) -> Optional[MemCell]:
        """
        追加 MemCell
        """
        try:
            await memcell.insert(session=session)
            print(f"✅ 追加 MemCell 成功: {memcell.event_id}")
            return memcell
        except Exception as e:
            logger.error("❌ 追加 MemCell 失败: %s", e)
            return None

    async def update_by_event_id(
        self,
        event_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[MemCell]:
        """
        根据 event_id 更新 MemCell

        Args:
            event_id: 事件 ID
            update_data: 更新数据字典
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新后的 MemCell 实例或 None
        """
        try:
            memcell = await self.get_by_event_id(event_id)
            if memcell:
                for key, value in update_data.items():
                    if hasattr(memcell, key):
                        setattr(memcell, key, value)
                await memcell.save(session=session)
                logger.debug("✅ 根据 event_id 更新 MemCell 成功: %s", event_id)
                return memcell
            return None
        except Exception as e:
            logger.error("❌ 根据 event_id 更新 MemCell 失败: %s", e)
            raise e

    async def delete_by_event_id(
        self, event_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        根据 event_id 删除 MemCell

        Args:
            event_id: 事件 ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            删除成功返回 True，否则返回 False
        """
        try:
            memcell = await self.get_by_event_id(event_id)
            if memcell:
                await memcell.delete(session=session)
                logger.debug("✅ 根据 event_id 删除 MemCell 成功: %s", event_id)
                return True
            return False
        except Exception as e:
            logger.error("❌ 根据 event_id 删除 MemCell 失败: %s", e)
            return False

    # ==================== 查询方法 ====================

    async def find_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
    ) -> List[MemCell]:
        """
        根据用户 ID 查询 MemCell

        Args:
            user_id: 用户 ID
            limit: 限制返回数量
            skip: 跳过数量
            sort_desc: 是否按时间降序排序

        Returns:
            MemCell 列表
        """
        try:
            query = self.model.find({"user_id": user_id})

            # 排序
            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            # 分页
            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据用户 ID 查询 MemCell 成功: %s, 找到 %d 条记录",
                user_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户 ID 查询 MemCell 失败: %s", e)
            return []

    async def find_by_user_and_time_range(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[MemCell]:
        """
        根据用户 ID 和时间范围查询 MemCell

        同时检查 user_id 字段和 participants 数组，只要 user_id 在其中之一即可

        Args:
            user_id: 用户 ID
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制返回数量
            skip: 跳过数量

        Returns:
            MemCell 列表
        """
        try:
            # 同时检查 user_id 字段和 participants 数组
            # 使用 OR 逻辑：user_id 匹配 或者 user_id 在 participants 中
            # 注意：MongoDB 对数组字段使用 Eq 会自动检查数组是否包含该值
            query = self.model.find(
                And(
                    Or(
                        Eq(MemCell.user_id, user_id),
                        Eq(
                            MemCell.participants, user_id
                        ),  # MongoDB 会检查数组中是否包含该值
                    ),
                    GTE(MemCell.timestamp, start_time),
                    LT(MemCell.timestamp, end_time),
                )
            ).sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据用户和时间范围查询 MemCell 成功: %s, 时间范围: %s - %s, 找到 %d 条记录",
                user_id,
                start_time,
                end_time,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户和时间范围查询 MemCell 失败: %s", e)
            return []

    async def find_by_group_id(
        self,
        group_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
    ) -> List[MemCell]:
        """
        根据群组 ID 查询 MemCell

        Args:
            group_id: 群组 ID
            limit: 限制返回数量
            skip: 跳过数量
            sort_desc: 是否按时间降序排序

        Returns:
            MemCell 列表
        """
        try:
            query = self.model.find({"group_id": group_id})

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
                "✅ 根据群组 ID 查询 MemCell 成功: %s, 找到 %d 条记录",
                group_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据群组 ID 查询 MemCell 失败: %s", e)
            return []

    async def find_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = False,
    ) -> List[MemCell]:
        """
        根据时间范围查询 MemCell

        Args:
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制返回数量
            skip: 跳过数量
            sort_desc: 是否按时间降序排序，默认False（升序）

        Returns:
            MemCell 列表
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
                "✅ 根据时间范围查询 MemCell 成功: 时间范围: %s - %s, 找到 %d 条记录",
                start_time,
                end_time,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据时间范围查询 MemCell 失败: %s", e)
            import traceback

            logger.error("详细错误信息: %s", traceback.format_exc())
            return []

    async def find_by_participants(
        self,
        participants: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[MemCell]:
        """
        根据参与者查询 MemCell

        Args:
            participants: 参与者列表
            match_all: 是否匹配所有参与者（True）或匹配任一参与者（False）
            limit: 限制返回数量
            skip: 跳过数量

        Returns:
            MemCell 列表
        """
        try:
            if match_all:
                # 匹配所有参与者
                query = self.model.find({"participants": {"$all": participants}})
            else:
                # 匹配任一参与者
                query = self.model.find({"participants": {"$in": participants}})

            query = query.sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据参与者查询 MemCell 成功: %s, 匹配模式: %s, 找到 %d 条记录",
                participants,
                '全部' if match_all else '任一',
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据参与者查询 MemCell 失败: %s", e)
            return []

    async def search_by_keywords(
        self,
        keywords: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[MemCell]:
        """
        根据关键词查询 MemCell

        Args:
            keywords: 关键词列表
            match_all: 是否匹配所有关键词（True）或匹配任一关键词（False）
            limit: 限制返回数量
            skip: 跳过数量

        Returns:
            MemCell 列表
        """
        try:
            if match_all:
                query = self.model.find({"keywords": {"$all": keywords}})
            else:
                query = self.model.find({"keywords": {"$in": keywords}})

            query = query.sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据关键词查询 MemCell 成功: %s, 匹配模式: %s, 找到 %d 条记录",
                keywords,
                '全部' if match_all else '任一',
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 根据关键词查询 MemCell 失败: %s", e)
            return []

    # ==================== 批量操作 ====================

    async def delete_by_user_id(
        self, user_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> int:
        """
        删除用户的所有 MemCell

        Args:
            user_id: 用户 ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            删除的记录数量
        """
        try:
            result = await self.model.find({"user_id": user_id}).delete(session=session)
            count = result.deleted_count if result else 0
            logger.info(
                "✅ 删除用户所有 MemCell 成功: %s, 删除 %d 条记录", user_id, count
            )
            return count
        except Exception as e:
            logger.error("❌ 删除用户所有 MemCell 失败: %s", e)
            return 0

    async def delete_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> int:
        """
        删除时间范围内的 MemCell

        Args:
            start_time: 开始时间
            end_time: 结束时间
            user_id: 可选的用户 ID 筛选
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            删除的记录数量
        """
        try:
            conditions = [
                GTE(MemCell.timestamp, start_time),
                LT(MemCell.timestamp, end_time),
            ]

            if user_id:
                conditions.append(Eq(MemCell.user_id, user_id))

            result = await self.model.find(And(*conditions)).delete(session=session)
            count = result.deleted_count if result else 0
            logger.info(
                "✅ 删除时间范围内 MemCell 成功: %s - %s, 用户: %s, 删除 %d 条记录",
                start_time,
                end_time,
                user_id or '全部',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ 删除时间范围内 MemCell 失败: %s", e)
            return 0

    # ==================== 统计和聚合查询 ====================

    async def count_by_user_id(self, user_id: str) -> int:
        """
        统计用户的 MemCell 数量

        Args:
            user_id: 用户 ID

        Returns:
            记录数量
        """
        try:
            count = await self.model.find({"user_id": user_id}).count()
            logger.debug(
                "✅ 统计用户 MemCell 数量成功: %s, 共 %d 条记录", user_id, count
            )
            return count
        except Exception as e:
            logger.error("❌ 统计用户 MemCell 数量失败: %s", e)
            return 0

    async def count_by_time_range(
        self, start_time: datetime, end_time: datetime, user_id: Optional[str] = None
    ) -> int:
        """
        统计时间范围内的 MemCell 数量

        Args:
            start_time: 开始时间
            end_time: 结束时间
            user_id: 可选的用户 ID 筛选

        Returns:
            记录数量
        """
        try:
            conditions = [
                GTE(MemCell.timestamp, start_time),
                LT(MemCell.timestamp, end_time),
            ]

            if user_id:
                conditions.append(Eq(MemCell.user_id, user_id))

            count = await self.model.find(And(*conditions)).count()
            logger.debug(
                "✅ 统计时间范围内 MemCell 数量成功: %s - %s, 用户: %s, 共 %d 条记录",
                start_time,
                end_time,
                user_id or '全部',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ 统计时间范围内 MemCell 数量失败: %s", e)
            return 0

    async def get_latest_by_user(self, user_id: str, limit: int = 10) -> List[MemCell]:
        """
        获取用户最新的 MemCell 记录

        Args:
            user_id: 用户 ID
            limit: 返回数量限制

        Returns:
            MemCell 列表
        """
        try:
            results = (
                await self.model.find({"user_id": user_id})
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ 获取用户最新 MemCell 成功: %s, 返回 %d 条记录",
                user_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ 获取用户最新 MemCell 失败: %s", e)
            return []

    async def get_user_activity_summary(
        self, user_id: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """
        获取用户活动摘要统计

        Args:
            user_id: 用户 ID
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            活动摘要字典
        """
        try:
            # 基础查询条件
            base_query = And(
                Eq(MemCell.user_id, user_id),
                GTE(MemCell.timestamp, start_time),
                LT(MemCell.timestamp, end_time),
            )

            # 总数量
            total_count = await self.model.find(base_query).count()

            # 按类型统计
            type_stats = {}
            for data_type in DataTypeEnum:
                type_query = And(base_query, Eq(MemCell.type, data_type))
                count = await self.model.find(type_query).count()
                if count > 0:
                    type_stats[data_type.value] = count

            # 获取最新和最早的记录
            latest = (
                await self.model.find(base_query).sort("-timestamp").limit(1).to_list()
            )
            earliest = (
                await self.model.find(base_query).sort("timestamp").limit(1).to_list()
            )

            summary = {
                "user_id": user_id,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "total_count": total_count,
                "type_distribution": type_stats,
                "latest_activity": latest[0].timestamp.isoformat() if latest else None,
                "earliest_activity": (
                    earliest[0].timestamp.isoformat() if earliest else None
                ),
            }

            logger.debug(
                "✅ 获取用户活动摘要成功: %s, 总计 %d 条记录", user_id, total_count
            )
            return summary
        except Exception as e:
            logger.error("❌ 获取用户活动摘要失败: %s", e)
            return {}


# 导出
__all__ = ["MemCellRawRepository"]
