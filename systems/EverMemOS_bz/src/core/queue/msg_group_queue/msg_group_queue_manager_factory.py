"""
消息分组队列管理器工厂

提供基于配置的 MsgGroupQueueManager 实例缓存和管理功能。
支持从环境变量读取配置，提供默认实例和命名实例。
参考 mongodb_client_factory.py 的设计模式。
"""

import os
import asyncio
from typing import Dict, Optional
from core.di.decorators import component
from core.observation.logger import get_logger
from .msg_group_queue_manager import MsgGroupQueueManager

logger = get_logger(__name__)


class MsgGroupQueueConfig:
    """消息分组队列配置类"""

    def __init__(
        self,
        name: str = "default",
        num_queues: int = 10,
        max_total_messages: int = 100,
        enable_metrics: bool = True,
        log_interval_seconds: int = 30,
        **kwargs,
    ):
        self.name = name
        self.num_queues = num_queues
        self.max_total_messages = max_total_messages
        self.enable_metrics = enable_metrics
        self.log_interval_seconds = log_interval_seconds
        self.kwargs = kwargs

    def get_cache_key(self) -> str:
        """
        获取缓存键

        基于核心配置参数生成唯一标识
        """
        return f"{self.name}:{self.num_queues}:{self.max_total_messages}:{self.enable_metrics}:{self.log_interval_seconds}"

    @classmethod
    def from_env(cls, prefix: str = "") -> 'MsgGroupQueueConfig':
        """
        从环境变量创建配置

        prefix 规则：若提供 prefix，将按 "{prefix}_XXX" 的形式读取变量，否则读取 "XXX"。
        例如：prefix="CLIENT" 则读取 "CLIENT_MSG_QUEUE_NUM_QUEUES"、"CLIENT_MSG_QUEUE_MAX_TOTAL_MESSAGES" 等。

        Args:
            prefix: 环境变量前缀

        Returns:
            MsgGroupQueueConfig: 配置实例
        """

        def _env(name: str, default: str) -> str:
            key = f"{prefix}_{name}" if prefix else name
            return os.getenv(key, default)

        # 读取配置项
        name = _env("MSG_QUEUE_NAME", "default")
        num_queues = int(_env("MSG_QUEUE_NUM_QUEUES", "10"))
        max_total_messages = int(_env("MSG_QUEUE_MAX_TOTAL_MESSAGES", "100"))
        enable_metrics = _env("MSG_QUEUE_ENABLE_METRICS", "true").lower() == "true"
        log_interval_seconds = int(_env("MSG_QUEUE_LOG_INTERVAL_SECONDS", "30"))

        return cls(
            name=name,
            num_queues=num_queues,
            max_total_messages=max_total_messages,
            enable_metrics=enable_metrics,
            log_interval_seconds=log_interval_seconds,
        )

    def __repr__(self) -> str:
        return (
            f"MsgGroupQueueConfig(name={self.name}, "
            f"num_queues={self.num_queues}, "
            f"max_total_messages={self.max_total_messages})"
        )


@component(name="msg_group_queue_manager_factory", primary=True)
class MsgGroupQueueManagerFactory:
    """消息分组队列管理器工厂"""

    def __init__(self):
        """初始化工厂"""
        self._managers: Dict[str, MsgGroupQueueManager] = {}
        self._default_config: Optional[MsgGroupQueueConfig] = None
        self._default_manager: Optional[MsgGroupQueueManager] = None
        self._lock = asyncio.Lock()

    async def get_manager(
        self, config: Optional[MsgGroupQueueConfig] = None, auto_start: bool = True
    ) -> MsgGroupQueueManager:
        """
        获取消息分组队列管理器

        Args:
            config: 队列管理器配置，如果为 None 则使用默认配置
            auto_start: 是否自动启动管理器

        Returns:
            MsgGroupQueueManager: 队列管理器
        """
        if config is None:
            config = await self._get_default_config()

        cache_key = config.get_cache_key()

        async with self._lock:
            # 检查缓存
            if cache_key in self._managers:
                manager = self._managers[cache_key]
                return manager

            # 创建新管理器
            logger.info("正在创建新的 MsgGroupQueueManager: %s", config)

            try:
                manager = MsgGroupQueueManager(
                    name=config.name,
                    num_queues=config.num_queues,
                    max_total_messages=config.max_total_messages,
                    enable_metrics=config.enable_metrics,
                    log_interval_seconds=config.log_interval_seconds,
                    **config.kwargs,
                )

                if auto_start:
                    await manager.start_periodic_logging()

                # 缓存管理器
                self._managers[cache_key] = manager
                logger.info("✅ MsgGroupQueueManager 创建成功并已缓存: %s", config)

                return manager

            except Exception as e:
                logger.error(
                    "❌ 创建 MsgGroupQueueManager 失败: %s, 错误: %s", config, e
                )
                raise

    async def get_default_manager(
        self, auto_start: bool = True
    ) -> MsgGroupQueueManager:
        """
        获取默认消息分组队列管理器

        Args:
            auto_start: 是否自动启动管理器

        Returns:
            MsgGroupQueueManager: 默认队列管理器
        """
        if self._default_manager is None:
            config = await self._get_default_config()
            self._default_manager = await self.get_manager(config, auto_start)

        return self._default_manager

    async def get_named_manager(
        self, name: str, auto_start: bool = True
    ) -> MsgGroupQueueManager:
        """
        按名称获取消息分组队列管理器

        约定：name 作为环境变量前缀，从 "{name}_MSG_QUEUE_XXX" 读取配置。
        例如 name="CLIENT" 时，将尝试读取 "CLIENT_MSG_QUEUE_NUM_QUEUES"、"CLIENT_MSG_QUEUE_MAX_TOTAL_MESSAGES" 等。

        Args:
            name: 前缀名称（即环境变量前缀）
            auto_start: 是否自动启动管理器

        Returns:
            MsgGroupQueueManager: 队列管理器
        """
        if name.lower() == "default":
            return await self.get_default_manager(auto_start)

        config = MsgGroupQueueConfig.from_env(prefix=name)
        # 确保配置名称与请求名称一致
        config.name = name.lower()

        logger.info("📋 加载命名 MsgGroupQueueManager 配置[name=%s]: %s", name, config)
        return await self.get_manager(config, auto_start)

    async def _get_default_config(self) -> MsgGroupQueueConfig:
        """获取默认配置"""
        if self._default_config is None:
            self._default_config = MsgGroupQueueConfig.from_env()
            logger.info(
                "📋 加载默认 MsgGroupQueueManager 配置: %s", self._default_config
            )

        return self._default_config

    async def create_manager_with_config(
        self,
        name: str = "default",
        num_queues: int = 10,
        max_total_messages: int = 100,
        enable_metrics: bool = True,
        log_interval_seconds: int = 30,
        auto_start: bool = True,
        **kwargs,
    ) -> MsgGroupQueueManager:
        """
        使用指定配置创建管理器

        Args:
            name: 管理器名称
            num_queues: 队列数量
            max_total_messages: 最大总消息数量
            enable_metrics: 是否启用统计
            log_interval_seconds: 日志间隔
            auto_start: 是否自动启动
            **kwargs: 其他参数

        Returns:
            MsgGroupQueueManager: 队列管理器
        """
        config = MsgGroupQueueConfig(
            name=name,
            num_queues=num_queues,
            max_total_messages=max_total_messages,
            enable_metrics=enable_metrics,
            log_interval_seconds=log_interval_seconds,
            **kwargs,
        )

        return await self.get_manager(config, auto_start)

    async def stop_manager(self, config: Optional[MsgGroupQueueConfig] = None):
        """
        停止指定管理器

        Args:
            config: 配置，如果为 None 则停止默认管理器
        """
        if config is None:
            if self._default_manager:
                await self._default_manager.shutdown()
                return

        cache_key = config.get_cache_key()

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

            logger.info("🔌 所有 MsgGroupQueueManager 已停止")

    def get_cached_managers_info(self) -> Dict[str, Dict]:
        """获取缓存的管理器信息"""
        return {
            cache_key: {
                "name": manager.name,
                "num_queues": manager.num_queues,
                "max_total_messages": manager.max_total_messages,
                "manager_stats": "需要异步调用get_manager_stats()获取",
            }
            for cache_key, manager in self._managers.items()
        }

    async def get_default_msg_group_queue_manager(
        self, auto_start: bool = True
    ) -> MsgGroupQueueManager:
        """
        获取默认消息分组队列管理器的便捷函数

        Args:
            auto_start: 是否自动启动管理器

        Returns:
            MsgGroupQueueManager: 默认队列管理器
        """
        return await self.get_default_manager(auto_start)
