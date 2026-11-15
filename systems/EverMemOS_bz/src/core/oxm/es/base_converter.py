"""
Elasticsearch 文档转换器基类

提供任意数据源到Elasticsearch文档的转换基础功能。
所有ES文档转换器都应该继承这个基类以获得统一的转换接口。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Any, get_args, get_origin
from core.oxm.es.doc_base import DocBase
from core.observation.logger import get_logger

logger = get_logger(__name__)

# 泛型类型变量 - 只限制ES文档类型
EsDocType = TypeVar('EsDocType', bound=DocBase)


class BaseEsConverter(ABC, Generic[EsDocType]):
    """
    Elasticsearch 文档转换器基类

    提供任意数据源到Elasticsearch文档的转换基础功能。
    所有ES文档转换器都应该继承这个类。

    特性：
    - 统一的转换接口（类方法）
    - 类型安全的ES文档泛型支持
    - 自动从泛型获取ES文档类型
    - 灵活的数据源支持
    """

    @classmethod
    def get_es_model(cls) -> Type[EsDocType]:
        """
        从泛型信息中获取ES文档模型类型

        Returns:
            Type[EsDocType]: ES文档模型类
        """
        # 获取类的泛型基类
        if hasattr(cls, '__orig_bases__'):
            for base in cls.__orig_bases__:
                if get_origin(base) is BaseEsConverter:
                    args = get_args(base)
                    if args:
                        return args[0]

        raise ValueError(f"无法从 {cls.__name__} 的泛型信息中获取ES文档类型")

    @classmethod
    @abstractmethod
    def from_mongo(cls, source_doc: Any) -> EsDocType:
        """
        从数据源转换为Elasticsearch文档

        这是核心转换方法，子类必须实现具体的转换逻辑。

        Args:
            source_doc: 源数据（可以是任意类型）

        Returns:
            EsDocType: Elasticsearch文档实例

        Raises:
            Exception: 当转换过程中发生错误时抛出异常
        """
        raise NotImplementedError("子类必须实现from_mongo方法")
