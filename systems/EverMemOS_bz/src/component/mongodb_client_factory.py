"""
MongoDB 客户端工厂

提供基于配置的 MongoDB 客户端缓存和管理功能。
支持从环境变量读取配置，提供默认客户端。
"""

import os
import asyncio
from typing import Dict, Optional, List
from urllib.parse import quote_plus
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from core.class_annotations.utils import get_annotation
from core.oxm.mongo.constant.annotations import ClassAnnotationKey, Toggle

from core.di.decorators import component
from core.observation.logger import get_logger
from common_utils.datetime_utils import timezone
from core.oxm.mongo.document_base import DEFAULT_DATABASE

logger = get_logger(__name__)


class MongoDBConfig:
    """MongoDB 配置类"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "memsys",
        uri: Optional[str] = None,
        uri_params: Optional[str] = None,
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.uri = uri
        self.uri_params = uri_params
        self.kwargs = kwargs

    def get_connection_string(self) -> str:
        """获取连接字符串，并拼接统一的 URI 参数（如果有）"""
        # 基础 URI
        if self.uri:
            base_uri = self.uri
        else:
            if self.username and self.password:
                # URL编码用户名和密码
                encoded_username = quote_plus(self.username)
                encoded_password = quote_plus(self.password)
                base_uri = f"mongodb://{encoded_username}:{encoded_password}@{self.host}:{self.port}/{self.database}"
            else:
                base_uri = f"mongodb://{self.host}:{self.port}/{self.database}"

        # 追加统一参数
        uri_params: Optional[str] = self.uri_params
        if uri_params:
            separator = '&' if ('?' in base_uri) else '?'
            return f"{base_uri}{separator}{uri_params}"
        return base_uri

    def get_cache_key(self) -> str:
        """获取缓存键

        仅基于基础信息 + 统一 URI 参数字符串生成签名，避免不同参数复用同一客户端。
        """
        base = f"{self.host}:{self.port}:{self.database}:{self.username or 'anonymous'}"
        uri_params: Optional[str] = self.uri_params
        signature = uri_params.strip() if isinstance(uri_params, str) else ""
        return f"{base}:{signature}" if signature else base

    @classmethod
    def from_env(cls, prefix: str = "") -> 'MongoDBConfig':
        """
        从环境变量创建配置。

        prefix 规则：若提供 prefix，将按 "{prefix}_XXX" 的形式读取变量，否则读取 "XXX"。
        例如：prefix="a" 则读取 "A_MONGODB_URI"、"A_MONGODB_HOST" 等。
        """

        def _env(name: str, default: Optional[str] = None) -> Optional[str]:
            if prefix == DEFAULT_DATABASE:
                key = name
            else:
                prefix_upper = prefix.upper()
                key = f"{prefix_upper}_{name}" if prefix else name
            return os.getenv(key, default) if default is not None else os.getenv(key)

        # 优先使用 MONGODB_URI
        uri = _env("MONGODB_URI")
        if uri:
            return cls(uri=uri, database=_env("MONGODB_DATABASE", "memsys"))

        # 分别读取各个配置项
        host = _env("MONGODB_HOST", "localhost")
        port = int(_env("MONGODB_PORT", "27017"))
        username = _env("MONGODB_USERNAME")
        password = _env("MONGODB_PASSWORD")
        database = _env("MONGODB_DATABASE", "memsys")
        uri_params = _env("MONGODB_URI_PARAMS", "")

        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            uri_params=uri_params,
        )

    def __repr__(self) -> str:
        return f"MongoDBConfig(host={self.host}, port={self.port}, database={self.database})"


class MongoDBClientWrapper:
    """MongoDB 客户端包装器"""

    def __init__(self, client: AsyncIOMotorClient, config: MongoDBConfig):
        self.client = client
        self.config = config
        self.database = client[config.database]
        self._initialized = False
        self._document_models: List = []

    async def initialize_beanie(self, document_models: Optional[List] = None):
        """初始化 Beanie ODM"""
        if self._initialized:
            return

        if document_models:
            try:
                # 将模型分组：可写组（需要索引）、只读组（跳过索引）
                writable_models = []
                readonly_models = []
                for model in document_models:
                    readonly_flag = get_annotation(model, ClassAnnotationKey.READONLY)
                    if readonly_flag == Toggle.ENABLED:
                        readonly_models.append(model)
                    else:
                        writable_models.append(model)

                if writable_models and readonly_models:
                    # 这里多次init_beanie，代码上看问题不大，可能需要考虑两种类型引用的问题；但是目前业务上也不会出现“同一个db，不同读写模式模型”，要防止未来业务有这个需求，所以这里还是提示一下
                    raise ValueError("可写组和只读组不能同时存在")

                logger.info(
                    "正在初始化 Beanie ODM（可写组），数据库: %s，模型数: %d",
                    self.config.database,
                    len(writable_models),
                )
                if writable_models:
                    await init_beanie(
                        database=self.database,
                        document_models=writable_models,
                        skip_indexes=False,
                    )

                logger.info(
                    "正在初始化 Beanie ODM（只读组），数据库: %s，模型数: %d",
                    self.config.database,
                    len(readonly_models),
                )
                if readonly_models:
                    await init_beanie(
                        database=self.database,
                        document_models=readonly_models,
                        skip_indexes=True,
                    )

                self._document_models = document_models
                self._initialized = True
                logger.info(
                    "✅ Beanie ODM 初始化成功，注册了 %d 个模型", len(document_models)
                )

                for model in document_models:
                    logger.info(
                        "📋 注册模型: database=%s, model=%s -> %s",
                        self.config.database,
                        model.__name__,
                        model.get_collection_name(),
                    )

            except Exception as e:
                logger.error("❌ Beanie 初始化失败: %s", e)
                raise

    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            await self.client.admin.command('ping')
            logger.info("✅ MongoDB 连接测试成功: %s", self.config)
            return True
        except Exception as e:
            logger.error("❌ MongoDB 连接测试失败: %s, 错误: %s", self.config, e)
            return False

    async def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        try:
            stats = {}
            collections = await self.database.list_collection_names()

            for collection_name in collections:
                try:
                    collection_stats = await self.database.command(
                        "collStats", collection_name
                    )
                    stats[collection_name] = {
                        "count": collection_stats.get("count", 0),
                        "size": collection_stats.get("size", 0),
                        "avgObjSize": collection_stats.get("avgObjSize", 0),
                        "storageSize": collection_stats.get("storageSize", 0),
                        "indexes": collection_stats.get("nindexes", 0),
                    }
                except Exception as e:
                    logger.warning("获取集合 %s 统计信息失败: %s", collection_name, e)

            return stats
        except Exception as e:
            logger.error("获取统计信息失败: %s", e)
            return {}

    async def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB 连接已关闭: %s", self.config)

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化 Beanie"""
        return self._initialized


@component(name="mongodb_client_factory", primary=True)
class MongoDBClientFactory:
    """MongoDB 客户端工厂"""

    def __init__(self):
        """初始化工厂"""
        self._clients: Dict[str, MongoDBClientWrapper] = {}
        self._default_config: Optional[MongoDBConfig] = None
        self._default_client: Optional[MongoDBClientWrapper] = None
        self._lock = asyncio.Lock()

    async def get_client(
        self, config: Optional[MongoDBConfig] = None, **connection_kwargs
    ) -> MongoDBClientWrapper:
        """
        获取 MongoDB 客户端

        Args:
            config: MongoDB 配置，如果为 None 则使用默认配置
            **connection_kwargs: 额外的连接参数

        Returns:
            MongoDBClientWrapper: MongoDB 客户端包装器
        """

        if config is None:
            config = await self._get_default_config()

        cache_key = config.get_cache_key()

        async with self._lock:
            # 检查缓存
            if cache_key in self._clients:
                return self._clients[cache_key]

            # 创建新客户端
            logger.info("正在创建新的 MongoDB 客户端: %s", config)

            # 合并连接参数
            conn_kwargs = {
                "serverSelectionTimeoutMS": 5000,
                "maxPoolSize": 50,
                "minPoolSize": 5,
                "tz_aware": True,
                "tzinfo": timezone,
                **config.kwargs,
                **connection_kwargs,
            }

            try:
                client = AsyncIOMotorClient(
                    config.get_connection_string(), **conn_kwargs
                )

                client_wrapper = MongoDBClientWrapper(client, config)

                # 测试连接
                if not await client_wrapper.test_connection():
                    await client_wrapper.close()
                    raise RuntimeError(f"MongoDB 连接测试失败: {config}")

                # 缓存客户端
                self._clients[cache_key] = client_wrapper
                logger.info("✅ MongoDB 客户端创建成功并已缓存: %s", config)

                return client_wrapper

            except Exception as e:
                logger.error("❌ 创建 MongoDB 客户端失败: %s, 错误: %s", config, e)
                raise

    async def get_default_client(self) -> MongoDBClientWrapper:
        """
        获取默认 MongoDB 客户端

        Returns:
            MongoDBClientWrapper: 默认 MongoDB 客户端包装器
        """
        if self._default_client is None:
            config = await self._get_default_config()
            self._default_client = await self.get_client(config)

        return self._default_client

    async def get_named_client(self, name: str) -> MongoDBClientWrapper:
        """
        按名称获取 MongoDB 客户端。

        约定：name 作为环境变量前缀，从 "{name}_MONGODB_XXX" 读取配置。
        例如 name="A" 时，将尝试读取 "A_MONGODB_URI"、"A_MONGODB_HOST" 等。

        Args:
            name: 前缀名称（即环境变量前缀）

        Returns:
            MongoDBClientWrapper: MongoDB 客户端包装器
        """
        if name == DEFAULT_DATABASE:
            return await self.get_default_client()
        config = MongoDBConfig.from_env(prefix=name)
        logger.info("📋 加载命名 MongoDB 配置[name=%s]: %s", name, config)
        return await self.get_client(config)

    async def _get_default_config(self) -> MongoDBConfig:
        """获取默认配置"""
        if self._default_config is None:
            self._default_config = MongoDBConfig.from_env()
            logger.info("📋 加载默认 MongoDB 配置: %s", self._default_config)

        return self._default_config

    async def create_client_with_config(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "memsys",
        **kwargs,
    ) -> MongoDBClientWrapper:
        """
        使用指定配置创建客户端

        Args:
            host: MongoDB 主机
            port: MongoDB 端口
            username: 用户名
            password: 密码
            database: 数据库名
            **kwargs: 其他连接参数

        Returns:
            MongoDBClientWrapper: MongoDB 客户端包装器
        """
        config = MongoDBConfig(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            **kwargs,
        )

        return await self.get_client(config)

    async def close_client(self, config: Optional[MongoDBConfig] = None):
        """
        关闭指定客户端

        Args:
            config: 配置，如果为 None 则关闭默认客户端
        """
        if config is None:
            if self._default_client:
                await self._default_client.close()
                self._default_client = None
                return

        cache_key = config.get_cache_key()

        async with self._lock:
            if cache_key in self._clients:
                await self._clients[cache_key].close()
                del self._clients[cache_key]

    async def close_all_clients(self):
        """关闭所有客户端"""
        async with self._lock:
            for client_wrapper in self._clients.values():
                await client_wrapper.close()

            self._clients.clear()

            if self._default_client:
                self._default_client = None

            logger.info("🔌 所有 MongoDB 客户端已关闭")

    def get_cached_clients_info(self) -> Dict[str, Dict]:
        """获取缓存的客户端信息"""
        return {
            cache_key: {
                "config": str(wrapper.config),
                "initialized": wrapper.is_initialized,
                "document_models": [
                    model.__name__ for model in wrapper._document_models
                ],  # pylint: disable=protected-access
            }
            for cache_key, wrapper in self._clients.items()
        }

    async def get_default_mongodb_client(self) -> MongoDBClientWrapper:
        """
        获取默认 MongoDB 客户端的便捷函数

        Returns:
            MongoDBClientWrapper: 默认 MongoDB 客户端包装器
        """
        return await self.get_default_client()
