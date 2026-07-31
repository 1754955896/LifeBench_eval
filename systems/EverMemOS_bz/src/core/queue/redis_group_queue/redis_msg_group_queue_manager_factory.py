"""
Redis消息分组队列管理器工厂

提供基于配置的 RedisGroupQueueManager 实例缓存和管理功能。
支持从环境变量读取配置，提供默认实例和命名实例。
参考 mongodb_client_factory.py 的设计模式。
"""

import os
import asyncio
from typing import Dict, Optional, Callable, Type
from core.di.decorators import component
from core.observation.logger import get_logger
from component.redis_provider import RedisProvider
from .redis_msg_group_queue_manager import RedisGroupQueueManager
from .redis_group_queue_item import RedisGroupQueueItem, SerializationMode

logger = get_logger(__name__)


class RedisGroupQueueConfig:
    """Redis分组队列配置类"""

    def __init__(
        self,
        key_prefix: str = "default",
        serialization_mode: SerializationMode = SerializationMode.JSON,
        sort_key_func: Optional[Callable[[RedisGroupQueueItem], int]] = None,
        max_total_messages: int = 1000,
        queue_expire_seconds: int = 12 * 3600,  # 12小时
        activity_expire_seconds: int = 7 * 24 * 3600,  # 7天
        enable_metrics: bool = True,
        log_interval_seconds: int = 60,
        cleanup_interval_seconds: int = 300,  # 5分钟
        **kwargs,
    ):
        self.key_prefix = key_prefix
        self.serialization_mode = serialization_mode
        self.sort_key_func = sort_key_func
        self.max_total_messages = max_total_messages
        self.queue_expire_seconds = queue_expire_seconds
        self.activity_expire_seconds = activity_expire_seconds
        self.enable_metrics = enable_metrics
        self.log_interval_seconds = log_interval_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.kwargs = kwargs

    def get_cache_key(self) -> str:
        """
        获取缓存键

        基于核心配置参数生成唯一标识
        """
        # 排序函数使用函数名或默认值
        sort_func_name = (
            getattr(self.sort_key_func, '__name__', 'default')
            if self.sort_key_func
            else 'default'
        )

        return (
            f"{self.key_prefix}:{self.serialization_mode.value}:{sort_func_name}:"
            f"{self.max_total_messages}:{self.queue_expire_seconds}:"
            f"{self.activity_expire_seconds}:{self.enable_metrics}:"
            f"{self.log_interval_seconds}:{self.cleanup_interval_seconds}"
        )

    @classmethod
    def from_env(cls, prefix: str = "") -> 'RedisGroupQueueConfig':
        """
        从环境变量创建配置

        prefix 规则：若提供 prefix，将按 "{prefix}_XXX" 的形式读取变量，否则读取 "XXX"。
        例如：prefix="CLIENT" 则读取 "CLIENT_REDIS_QUEUE_KEY_PREFIX"、"CLIENT_REDIS_QUEUE_MAX_TOTAL_MESSAGES" 等。

        Args:
            prefix: 环境变量前缀

        Returns:
            RedisGroupQueueConfig: 配置实例
        """

        def _env(name: str, default: str) -> str:
            key = f"{prefix}_{name}" if prefix else name
            return os.getenv(key, default)

        # 读取配置项
        base_key_prefix = _env("REDIS_QUEUE_KEY_PREFIX", "default")
        # 支持全局Redis前缀
        global_redis_prefix = _env("GLOBAL_REDIS_PREFIX", "")
        key_prefix = (
            f"{global_redis_prefix}:{base_key_prefix}"
            if global_redis_prefix
            else base_key_prefix
        )
        # 序列化模式配置
        serialization_mode_str = _env("REDIS_QUEUE_SERIALIZATION_MODE", "json").lower()
        serialization_mode = (
            SerializationMode.JSON
            if serialization_mode_str == "json"
            else SerializationMode.BSON
        )
        max_total_messages = int(_env("REDIS_QUEUE_MAX_TOTAL_MESSAGES", "20000"))
        queue_expire_seconds = int(_env("REDIS_QUEUE_EXPIRE_SECONDS", str(24 * 3600)))
        activity_expire_seconds = int(
            _env("REDIS_QUEUE_ACTIVITY_EXPIRE_SECONDS", str(24 * 3600))
        )
        enable_metrics = _env("REDIS_QUEUE_ENABLE_METRICS", "true").lower() == "true"
        log_interval_seconds = int(_env("REDIS_QUEUE_LOG_INTERVAL_SECONDS", "600"))
        cleanup_interval_seconds = int(
            _env("REDIS_QUEUE_CLEANUP_INTERVAL_SECONDS", "300")
        )

        return cls(
            key_prefix=key_prefix,
            serialization_mode=serialization_mode,
            max_total_messages=max_total_messages,
            queue_expire_seconds=queue_expire_seconds,
            activity_expire_seconds=activity_expire_seconds,
            enable_metrics=enable_metrics,
            log_interval_seconds=log_interval_seconds,
            cleanup_interval_seconds=cleanup_interval_seconds,
        )

    def __repr__(self) -> str:
        return (
            f"RedisGroupQueueConfig(key_prefix={self.key_prefix}, "
            f"max_total_messages={self.max_total_messages})"
        )


@component(name="redis_group_queue_manager_factory", primary=True)
class RedisGroupQueueManagerFactory:
    """Redis消息分组队列管理器工厂"""

    def __init__(self, redis_provider: RedisProvider):
        """
        初始化工厂

        Args:
            redis_provider: Redis连接提供者
        """
        self.redis_provider = redis_provider
        self._managers: Dict[str, RedisGroupQueueManager] = {}
        self._default_config: Optional[RedisGroupQueueConfig] = None
        self._default_manager: Optional[RedisGroupQueueManager] = None
        self._lock = asyncio.Lock()

    async def get_manager(
        self,
        config: Optional[RedisGroupQueueConfig] = None,
        item_class: Optional[Type[RedisGroupQueueItem]] = None,
        auto_start: bool = True,
        redis_client_name: str = "default",
    ) -> RedisGroupQueueManager:
        """
        获取Redis消息分组队列管理器

        Args:
            config: 队列管理器配置，如果为 None 则使用默认配置
            item_class: 队列项类型，必须继承自RedisGroupQueueItem，默认使用SimpleQueueItem
            auto_start: 是否自动启动管理器
            redis_client_name: Redis客户端名称

        Returns:
            RedisGroupQueueManager: 队列管理器
        """
        if config is None:
            config = await self._get_default_config()

        # 生成缓存键，包含 item_class 信息
        item_class_name = item_class.__name__ if item_class else 'default'
        cache_key = f"{config.get_cache_key()}:{item_class_name}:{redis_client_name}"

        async with self._lock:
            # 检查缓存
            if cache_key in self._managers:
                manager = self._managers[cache_key]
                return manager

            # 创建新管理器
            logger.info("正在创建新的 RedisGroupQueueManager: %s", config)

            try:
                # 根据序列化模式获取Redis客户端
                if config.serialization_mode == SerializationMode.BSON:
                    # BSON模式：使用binary_cache，不解码响应以支持字节数据
                    redis_client = await self.redis_provider.get_named_client(
                        "binary_cache", decode_responses=False
                    )
                else:
                    # JSON模式：使用default客户端，自动解码响应
                    redis_client = await self.redis_provider.get_client()

                manager = RedisGroupQueueManager(
                    redis_client=redis_client,
                    key_prefix=config.key_prefix,
                    serialization_mode=config.serialization_mode,
                    item_class=item_class,
                    sort_key_func=config.sort_key_func,
                    max_total_messages=config.max_total_messages,
                    queue_expire_seconds=config.queue_expire_seconds,
                    activity_expire_seconds=config.activity_expire_seconds,
                    enable_metrics=config.enable_metrics,
                    log_interval_seconds=config.log_interval_seconds,
                    cleanup_interval_seconds=config.cleanup_interval_seconds,
                    **config.kwargs,
                )

                if auto_start:
                    await manager.start()

                # 缓存管理器
                self._managers[cache_key] = manager
                logger.info("✅ RedisGroupQueueManager 创建成功并已缓存: %s", config)

                return manager

            except Exception as e:
                logger.error(
                    "❌ 创建 RedisGroupQueueManager 失败: %s, 错误: %s", config, e
                )
                raise

    async def _get_default_config(self) -> RedisGroupQueueConfig:
        """获取默认配置"""
        if self._default_config is None:
            self._default_config = RedisGroupQueueConfig.from_env()
            logger.info(
                "📋 加载默认 RedisGroupQueueManager 配置: %s", self._default_config
            )

        return self._default_config

    async def get_manager_with_config(
        self,
        key_prefix: str = "default",
        serialization_mode: SerializationMode = SerializationMode.JSON,
        item_class: Optional[Type[RedisGroupQueueItem]] = None,
        sort_key_func: Optional[Callable[[RedisGroupQueueItem], int]] = None,
        max_total_messages: int = 2 * 10000,
        queue_expire_seconds: int = 24 * 3600,
        activity_expire_seconds: int = 24 * 3600,
        enable_metrics: bool = True,
        log_interval_seconds: int = 600,
        cleanup_interval_seconds: int = 300,
        auto_start: bool = True,
        redis_client_name: str = "default",
        **kwargs,
    ) -> RedisGroupQueueManager:
        """
        使用指定配置创建管理器

        Args:
            key_prefix: Redis键前缀，用于区分不同的管理器实例
            serialization_mode: 序列化模式（JSON或BSON）
            item_class: 队列项类型，必须继承自RedisGroupQueueItem，默认使用SimpleQueueItem
            sort_key_func: 排序键生成函数
            max_total_messages: 最大总消息数量
            queue_expire_seconds: 队列过期时间
            activity_expire_seconds: 活动记录过期时间
            enable_metrics: 是否启用统计
            log_interval_seconds: 日志间隔
            cleanup_interval_seconds: 清理间隔
            auto_start: 是否自动启动
            redis_client_name: Redis客户端名称
            **kwargs: 其他参数

        Returns:
            RedisGroupQueueManager: 队列管理器
        """
        config = RedisGroupQueueConfig(
            key_prefix=key_prefix,
            serialization_mode=serialization_mode,
            sort_key_func=sort_key_func,
            max_total_messages=max_total_messages,
            queue_expire_seconds=queue_expire_seconds,
            activity_expire_seconds=activity_expire_seconds,
            enable_metrics=enable_metrics,
            log_interval_seconds=log_interval_seconds,
            cleanup_interval_seconds=cleanup_interval_seconds,
            **kwargs,
        )

        return await self.get_manager(config, item_class, auto_start, redis_client_name)

    async def stop_manager(
        self,
        config: Optional[RedisGroupQueueConfig] = None,
        item_class: Optional[Type[RedisGroupQueueItem]] = None,
        redis_client_name: str = "default",
    ):
        """
        停止指定管理器

        Args:
            config: 配置，如果为 None 则停止默认管理器
            item_class: 队列项类型，必须继承自RedisGroupQueueItem
            redis_client_name: Redis客户端名称
        """
        if config is None:
            if self._default_manager:
                await self._default_manager.shutdown()
                return

        # 生成缓存键，包含 item_class 信息
        item_class_name = item_class.__name__ if item_class else 'default'
        cache_key = f"{config.get_cache_key()}:{item_class_name}:{redis_client_name}"

        async with self._lock:
            if cache_key in self._managers:
                await self._managers[cache_key].shutdown()

    async def stop_all_managers(self):
        """停止所有管理器"""
        async with self._lock:
            for manager in self._managers.values():
                await manager.shutdown()

            self._managers.clear()

            if self._default_manager:
                self._default_manager = None

            logger.info("🔌 所有 RedisGroupQueueManager 已停止")

    def get_cached_managers_info(self) -> Dict[str, Dict]:
        """获取缓存的管理器信息"""
        return {
            cache_key: {
                "key_prefix": manager.key_prefix,
                "max_total_messages": manager.max_total_messages,
                "manager_stats": "需要异步调用get_manager_stats()获取",
            }
            for cache_key, manager in self._managers.items()
        }
