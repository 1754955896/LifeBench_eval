"""
Milvus 集合转换器基类

提供任意数据源到 Milvus 集合的转换基础功能。
所有 Milvus 集合转换器都应该继承这个基类以获得统一的转换接口。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Any, get_args, get_origin
from core.oxm.milvus.milvus_collection_base import MilvusCollectionBase
from core.observation.logger import get_logger

logger = get_logger(__name__)

# 泛型类型变量 - 只限制 Milvus 集合类型
MilvusCollectionType = TypeVar('MilvusCollectionType', bound=MilvusCollectionBase)


class BaseMilvusConverter(ABC, Generic[MilvusCollectionType]):
    """
    Milvus 集合转换器基类

    提供任意数据源到 Milvus 集合的转换基础功能。
    所有 Milvus 集合转换器都应该继承这个类。

    特性：
    - 统一的转换接口（类方法）
    - 类型安全的 Milvus 集合泛型支持
    - 自动从泛型获取 Milvus 集合类型
    - 灵活的数据源支持
    """

    @classmethod
    def get_milvus_model(cls) -> Type[MilvusCollectionType]:
        """
        从泛型信息中获取 Milvus 集合模型类型

        Returns:
            Type[MilvusCollectionType]: Milvus 集合模型类
        """
        # 获取类的泛型基类
        if hasattr(cls, '__orig_bases__'):
            for base in cls.__orig_bases__:
                if get_origin(base) is BaseMilvusConverter:
                    args = get_args(base)
                    if args:
                        return args[0]

        raise ValueError(f"无法从 {cls.__name__} 的泛型信息中获取 Milvus 集合类型")

    @classmethod
    @abstractmethod
    def from_mongo(cls, source_doc: Any) -> MilvusCollectionType:
        """
        从数据源转换为 Milvus 集合实体

        这是核心转换方法，子类必须实现具体的转换逻辑。

        Args:
            source_doc: 源数据（可以是任意类型）

        Returns:
            MilvusCollectionType: Milvus 集合实体实例

        Raises:
            Exception: 当转换过程中发生错误时抛出异常
        """
        raise NotImplementedError("子类必须实现 from_mongo 方法")
