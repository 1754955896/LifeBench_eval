"""
Milvus 基础仓库类

提供通用的基础操作，所有 Milvus 仓库都应该继承这个类以获得统一的操作支持。
"""

from abc import ABC
from typing import Optional, TypeVar, Generic, Type, List, Any
from core.oxm.milvus.milvus_collection_base import MilvusCollectionBase
from core.oxm.milvus.async_collection import AsyncCollection
from core.observation.logger import get_logger
from core.di.utils import get_bean

logger = get_logger(__name__)

# 泛型类型变量
T = TypeVar('T', bound=MilvusCollectionBase)


class BaseMilvusRepository(ABC, Generic[T]):
    """
    Milvus 基础仓库类

    提供通用的基础操作，所有 Milvus 仓库都应该继承这个类。

    特性：
    - 异步 Milvus 集合管理
    - 基础 CRUD 操作模板
    - 统一的错误处理和日志记录
    - 集合管理
    """

    def __init__(self, model: Type[T]):
        """
        初始化基础仓库

        Args:
            model: Milvus 集合模型类
        """
        self.model = model
        self.model_name = model.__name__
        self.collection: Optional[AsyncCollection] = model.async_collection()
        self.schema = model._SCHEMA
        self.all_output_fields = [field.name for field in self.schema.fields]

    # ==================== 基础 CRUD 操作 ====================

    async def insert(self, entity: T, flush: bool = False) -> str:
        """
        插入新实体

        Args:
            entity: 实体实例
            flush: 是否立即刷新

        Returns:
            str: 插入的实体ID
        """
        try:
            entity_id = await self.collection.insert(entity)
            if flush:
                await self.collection.flush()
            logger.debug("✅ 插入实体成功 [%s]: %s", self.model_name, entity_id)
            return entity_id
        except Exception as e:
            logger.error("❌ 插入实体失败 [%s]: %s", self.model_name, e)
            raise

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        根据ID获取实体

        Args:
            entity_id: 实体ID

        Returns:
            实体实例或 None
        """
        try:
            # 获取集合的所有字段
            # 使用query查询
            results = await self.collection.query(
                expr=f'id == "{entity_id}"',
                output_fields=self.all_output_fields,
                limit=1,
            )
            return results[0] if results else None
        except Exception as e:
            logger.error("❌ 根据ID获取实体失败 [%s]: %s", self.model_name, e)
            return None

    async def upsert(self, entity: T, flush: bool = False) -> str:
        """
        更新或插入实体

        Args:
            entity: 实体实例
            flush: 是否立即刷新

        Returns:
            str: 实体ID
        """
        try:
            entity_id = await self.collection.upsert(entity)
            if flush:
                await self.collection.flush()
            logger.debug("✅ 更新/插入实体成功 [%s]: %s", self.model_name, entity_id)
            return entity_id
        except Exception as e:
            logger.error("❌ 更新/插入实体失败 [%s]: %s", self.model_name, e)
            raise

    async def delete_by_id(self, entity_id: str, flush: bool = False) -> bool:
        """
        根据ID删除实体

        Args:
            entity_id: 实体ID
            flush: 是否立即刷新

        Returns:
            bool: 删除成功返回 True
        """
        try:
            result = await self.collection.delete(expr=f'id == "{entity_id}"')
            success = result.delete_count > 0

            if flush and success:
                await self.collection.flush()
            if success:
                logger.debug("✅ 删除实体成功 [%s]: %s", self.model_name, entity_id)
            return success
        except Exception as e:
            logger.error("❌ 删除实体失败 [%s]: %s", self.model_name, e)
            return False

    # ==================== 批量操作 ====================

    async def insert_batch(self, entities: List[T], flush: bool = False) -> List[str]:
        """
        批量插入实体

        Args:
            entities: 实体列表
            flush: 是否立即刷新

        Returns:
            List[str]: 插入的实体ID列表
        """
        try:
            entity_ids = await self.collection.insert_batch(entities)
            if flush:
                await self.collection.flush()
            logger.debug(
                "✅ 批量插入实体成功 [%s]: %d 条记录", self.model_name, len(entities)
            )
            return entity_ids
        except Exception as e:
            logger.error("❌ 批量插入实体失败 [%s]: %s", self.model_name, e)
            raise

    # ==================== 集合操作 ====================

    async def flush(self) -> bool:
        """
        刷新集合

        Returns:
            bool: 刷新成功返回 True
        """
        try:
            await self.collection.flush()
            logger.debug("✅ 刷新集合成功 [%s]", self.model_name)
            return True
        except Exception as e:
            logger.error("❌ 刷新集合失败 [%s]: %s", self.model_name, e)
            return False

    async def load(self) -> bool:
        """
        加载集合到内存

        Returns:
            bool: 加载成功返回 True
        """
        try:
            await self.collection.load()
            logger.debug("✅ 加载集合成功 [%s]", self.model_name)
            return True
        except Exception as e:
            logger.error("❌ 加载集合失败 [%s]: %s", self.model_name, e)
            return False

    # ==================== 辅助方法 ====================

    def get_model_name(self) -> str:
        """
        获取模型名称

        Returns:
            str: 模型类名
        """
        return self.model_name
