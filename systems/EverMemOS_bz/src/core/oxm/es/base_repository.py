"""
Elasticsearch 基础仓库类

基于 elasticsearch-dsl 的基础仓库类，提供通用的基础 CRUD 操作。
所有 Elasticsearch 仓库都应该继承这个基类以获得统一的操作支持。
"""

from abc import ABC
from typing import Optional, TypeVar, Generic, Type, List, Dict, Any
from elasticsearch import AsyncElasticsearch
from core.oxm.es.doc_base import DocBase
from core.observation.logger import get_logger
from core.di.utils import get_bean

logger = get_logger(__name__)

# 泛型类型变量
T = TypeVar('T', bound=DocBase)


class BaseRepository(ABC, Generic[T]):
    """
    Elasticsearch 基础仓库类

    提供通用的基础操作，所有 Elasticsearch 仓库都应该继承这个类。

    特性：
    - 异步 Elasticsearch 客户端管理
    - 基础 CRUD 操作模板
    - 统一的错误处理和日志记录
    - 索引管理
    """

    def __init__(self, model: Type[T]):
        """
        初始化基础仓库

        Args:
            model: Elasticsearch 文档模型类
        """
        self.model = model
        self.model_name = model.__name__
        self._client: Optional[AsyncElasticsearch] = None

    # ==================== 客户端管理 ====================

    async def get_client(self) -> AsyncElasticsearch:
        """
        获取 Elasticsearch 异步客户端

        Returns:
            AsyncElasticsearch: 异步客户端实例
        """
        if self._client is None:
            try:
                es_factory = get_bean("elasticsearch_client_factory")
                client_wrapper = await es_factory.get_default_client()
                self._client = client_wrapper.async_client
            except Exception as e:
                logger.error(
                    "❌ 获取 Elasticsearch 客户端失败 [%s]: %s", self.model_name, e
                )
                raise
        return self._client

    def get_index_name(self) -> str:
        """
        获取索引名称

        Returns:
            str: 索引别名
        """
        if hasattr(self.model, '_index') and hasattr(self.model._index, '_name'):
            return self.model._index._name
        raise ValueError(f"文档类 {self.model_name} 没有正确的索引配置")

    # ==================== 基础 CRUD 模板方法 ====================

    async def create(self, document: T, refresh: bool = False) -> T:
        """
        创建新文档

        Args:
            document: 文档实例
            refresh: 是否立即刷新索引

        Returns:
            创建成功的文档实例
        """
        try:
            client = await self.get_client()
            await document.save(using=client, refresh=refresh)
            logger.debug(
                "✅ 创建文档成功 [%s]: %s",
                self.model_name,
                getattr(document, 'meta', {}).get('id', 'unknown'),
            )
            return document
        except Exception as e:
            logger.error("❌ 创建文档失败 [%s]: %s", self.model_name, e)
            raise

    async def get_by_id(self, doc_id: str) -> Optional[T]:
        """
        根据文档 ID 获取文档

        Args:
            doc_id: 文档 ID

        Returns:
            文档实例或 None
        """
        try:
            client = await self.get_client()
            return await self.model.get(id=doc_id, using=client)
        except Exception as e:
            logger.error("❌ 根据 ID 获取文档失败 [%s]: %s", self.model_name, e)
            return None

    async def update(self, document: T, refresh: bool = False) -> T:
        """
        更新文档

        Args:
            document: 要更新的文档实例
            refresh: 是否立即刷新索引

        Returns:
            更新后的文档实例
        """
        try:
            client = await self.get_client()
            await document.save(using=client, refresh=refresh)
            doc_id = getattr(getattr(document, 'meta', None), 'id', 'unknown')
            logger.debug("✅ 更新文档成功 [%s]: %s", self.model_name, doc_id)
            return document
        except Exception as e:
            logger.error("❌ 更新文档失败 [%s]: %s", self.model_name, e)
            raise

    async def delete_by_id(self, doc_id: str, refresh: bool = False) -> bool:
        """
        根据文档 ID 删除文档

        Args:
            doc_id: 文档 ID
            refresh: 是否立即刷新索引

        Returns:
            删除成功返回 True，否则返回 False
        """
        try:
            document = await self.get_by_id(doc_id)
            if document:
                client = await self.get_client()
                await document.delete(using=client, refresh=refresh)
                logger.debug("✅ 删除文档成功 [%s]: %s", self.model_name, doc_id)
                return True
            return False
        except Exception as e:
            logger.error("❌ 删除文档失败 [%s]: %s", self.model_name, e)
            return False

    async def delete(self, document: T, refresh: bool = False) -> bool:
        """
        删除文档实例

        Args:
            document: 要删除的文档实例
            refresh: 是否立即刷新索引

        Returns:
            删除成功返回 True，否则返回 False
        """
        try:
            client = await self.get_client()
            await document.delete(using=client, refresh=refresh)
            logger.debug(
                "✅ 删除文档成功 [%s]: %s",
                self.model_name,
                getattr(document, 'meta', {}).get('id', 'unknown'),
            )
            return True
        except Exception as e:
            logger.error("❌ 删除文档失败 [%s]: %s", self.model_name, e)
            return False

    # ==================== 批量操作 ====================

    async def create_batch(self, documents: List[T], refresh: bool = False) -> List[T]:
        """
        批量创建文档

        Args:
            documents: 文档列表
            refresh: 是否立即刷新索引

        Returns:
            成功创建的文档列表
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            # 构建批量操作
            actions = []
            for doc in documents:
                action = {"_index": index_name, "_source": doc.to_dict()}
                actions.append(action)

            # 执行批量操作
            from elasticsearch.helpers import async_bulk

            await async_bulk(client, actions, refresh=refresh)

            logger.debug(
                "✅ 批量创建文档成功 [%s]: %d 条记录", self.model_name, len(documents)
            )
            return documents
        except Exception as e:
            logger.error("❌ 批量创建文档失败 [%s]: %s", self.model_name, e)
            raise

    # ==================== 搜索方法 ====================

    async def search(
        self, query: Dict[str, Any], size: int = 10, from_: int = 0
    ) -> Dict[str, Any]:
        """
        执行搜索查询

        Args:
            query: Elasticsearch 查询 DSL
            size: 返回结果数量
            from_: 分页起始位置

        Returns:
            搜索结果
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            response = await client.search(
                index=index_name, body={"query": query, "size": size, "from": from_}
            )

            logger.debug(
                "✅ 搜索执行成功 [%s]: 找到 %d 条结果",
                self.model_name,
                response.get('hits', {}).get('total', {}).get('value', 0),
            )
            return response
        except Exception as e:
            logger.error("❌ 搜索执行失败 [%s]: %s", self.model_name, e)
            raise

    async def match_all(self, size: int = 10, from_: int = 0) -> List[T]:
        """
        获取所有文档

        Args:
            size: 返回结果数量
            from_: 分页起始位置

        Returns:
            文档列表
        """
        try:
            response = await self.search(
                query={"match_all": {}}, size=size, from_=from_
            )

            documents = []
            for hit in response.get('hits', {}).get('hits', []):
                doc = self.model.from_dict(hit['_source'])
                doc.meta.id = hit['_id']
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error("❌ 获取所有文档失败 [%s]: %s", self.model_name, e)
            return []

    # ==================== 统计方法 ====================

    async def exists_by_id(self, doc_id: str) -> bool:
        """
        检查文档是否存在

        Args:
            doc_id: 文档 ID

        Returns:
            存在返回 True，否则返回 False
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            response = await client.exists(index=index_name, id=doc_id)
            return response
        except Exception:
            return False

    # ==================== 索引管理 ====================

    async def refresh_index(self) -> bool:
        """
        手动刷新索引

        使用 connection.indices.refresh(index=index_name) 来手动刷新索引，
        确保新写入的数据立即可搜索。

        Returns:
            刷新成功返回 True，否则返回 False
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            await client.indices.refresh(index=index_name)
            logger.debug("✅ 手动刷新索引成功 [%s]: %s", self.model_name, index_name)
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error("❌ 手动刷新索引失败 [%s]: %s", self.model_name, e)
            return False
        except Exception as e:
            logger.error("❌ 手动刷新索引失败（未知错误） [%s]: %s", self.model_name, e)
            return False

    async def create_index(self) -> bool:
        """
        创建索引

        Returns:
            创建成功返回 True，否则返回 False
        """
        try:
            client = await self.get_client()

            # 使用文档类的 init 方法创建索引
            if hasattr(self.model, 'dest'):
                index_name = self.model.dest()
            else:
                from common_utils.datetime_utils import get_now_with_timezone

                now = get_now_with_timezone()
                alias = self.get_index_name()
                index_name = f"{alias}-{now.strftime('%Y%m%d%H%M%S%f')}"

            await self.model.init(index=index_name, using=client)

            # 创建别名
            alias = self.get_index_name()
            await client.indices.update_aliases(
                body={
                    "actions": [
                        {
                            "add": {
                                "index": index_name,
                                "alias": alias,
                                "is_write_index": True,
                            }
                        }
                    ]
                }
            )

            logger.debug(
                "✅ 创建索引成功 [%s]: %s -> %s", self.model_name, index_name, alias
            )
            return True
        except Exception as e:
            logger.error("❌ 创建索引失败 [%s]: %s", self.model_name, e)
            return False

    async def delete_index(self) -> bool:
        """
        删除索引

        Returns:
            删除成功返回 True，否则返回 False
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            await client.indices.delete(index=index_name)
            logger.debug("✅ 删除索引成功 [%s]: %s", self.model_name, index_name)
            return True
        except Exception as e:
            logger.error("❌ 删除索引失败 [%s]: %s", self.model_name, e)
            return False

    async def index_exists(self) -> bool:
        """
        检查索引是否存在

        Returns:
            存在返回 True，否则返回 False
        """
        try:
            client = await self.get_client()
            index_name = self.get_index_name()

            return await client.indices.exists(index=index_name)
        except Exception:
            return False

    # ==================== 辅助方法 ====================

    def get_model_name(self) -> str:
        """
        获取模型名称

        Returns:
            模型类名
        """
        return self.model_name

    def get_collection_name(self) -> str:
        """
        获取索引名称（兼容 MongoDB 仓库接口）

        Returns:
            Elasticsearch 索引名称
        """
        return self.get_index_name()


# 导出
__all__ = ["BaseRepository"]
