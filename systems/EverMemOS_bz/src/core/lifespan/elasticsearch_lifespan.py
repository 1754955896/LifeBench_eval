"""
Elasticsearch 生命周期提供者实现
"""

from fastapi import FastAPI
from typing import Any

from core.observation.logger import get_logger
from core.di.utils import get_bean, get_all_subclasses, get_bean_by_type
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider
from core.oxm.es.doc_base import DocBase
from component.elasticsearch_client_factory import ElasticsearchClientFactory

logger = get_logger(__name__)


@component(name="elasticsearch_lifespan_provider")
class ElasticsearchLifespanProvider(LifespanProvider):
    """Elasticsearch 生命周期提供者"""

    def __init__(self, name: str = "elasticsearch", order: int = 20):
        """
        初始化 Elasticsearch 生命周期提供者

        Args:
            name (str): 提供者名称
            order (int): 执行顺序，Elasticsearch 在数据库连接之后启动
        """
        super().__init__(name, order)
        self._es_factory = None
        self._es_client = None

    async def startup(self, app: FastAPI) -> Any:
        """
        启动 Elasticsearch 连接和初始化

        Args:
            app (FastAPI): FastAPI应用实例

        Returns:
            Any: Elasticsearch 客户端信息
        """
        logger.info("正在初始化 Elasticsearch 连接...")

        try:
            # 获取 Elasticsearch 客户端工厂
            self._es_factory: ElasticsearchClientFactory = get_bean(
                "elasticsearch_client_factory"
            )

            # 获取默认客户端
            self._es_client = await self._es_factory.get_default_client()

            # 获取所有 DocBase 的子类 动态生成的类好像找不到不知道为什么
            all_doc_classes = get_all_subclasses(DocBase)

            # 过滤出有效的文档类
            document_classes = []
            for doc_class in all_doc_classes:
                if hasattr(doc_class, '_index') and hasattr(doc_class._index, '_name'):
                    index_name = doc_class._index._name
                    # 检查索引名称是否有效
                    if index_name and index_name.strip():
                        document_classes.append(doc_class)
                        logger.info(
                            "发现文档类: %s -> %s", doc_class.__name__, index_name
                        )
                    else:
                        logger.warning(
                            "跳过文档类 %s，索引名称为空", doc_class.__name__
                        )
                else:
                    logger.debug("跳过文档类 %s，缺少索引配置", doc_class.__name__)

            # 初始化索引
            if document_classes:
                await self._es_client.initialize_indices(document_classes)
            else:
                logger.info("没有发现需要初始化的文档类")

            # 初始化 elasticsearch_dsl 连接
            self._es_factory.get_default_connection()

            logger.info("✅ Elasticsearch 连接初始化完成")

            return {
                "client": self._es_client,
                "factory": self._es_factory,
                "document_classes": [cls.__name__ for cls in document_classes],
            }

        except Exception as e:
            logger.error("❌ Elasticsearch 初始化过程中出错: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        关闭 Elasticsearch 连接

        Args:
            app (FastAPI): FastAPI应用实例
        """
        logger.info("正在关闭 Elasticsearch 连接...")

        if self._es_factory:
            try:
                await self._es_factory.close_all_clients()
                logger.info("✅ Elasticsearch 连接关闭完成")
            except Exception as e:
                logger.error("❌ 关闭 Elasticsearch 连接时出错: %s", str(e))
