"""
Redis连接提供者

提供Redis连接池管理和基础操作的技术组件
"""

import os
import asyncio
from typing import Optional, Union
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


@component(name="redis_provider", primary=True)
class RedisProvider:
    """Redis连接提供者"""

    def __init__(self):
        """初始化Redis连接提供者"""
        # 从环境变量读取Redis配置
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.redis_ssl = os.getenv("REDIS_SSL", "false").lower() == "true"

        # 构建Redis URL
        protocol = "rediss" if self.redis_ssl else "redis"
        if self.redis_password:
            self.redis_url = f"{protocol}://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            self.redis_url = (
                f"{protocol}://{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )

        # 其他配置使用默认值
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "60"))
        self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "15"))
        self.socket_connect_timeout = int(
            os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")
        )

        # 命名客户端缓存
        self._named_clients = {}
        self._named_pools = {}
        self._named_initialized = set()
        self._lock = asyncio.Lock()

    async def get_client(self) -> redis.Redis:
        """
        获取Redis客户端（默认客户端）

        Returns:
            redis.Redis: Redis客户端实例
        """
        return await self.get_named_client("default")

    async def get_named_client(self, name: str, **overrides) -> redis.Redis:
        """
        获取命名Redis客户端，支持参数覆盖

        Args:
            name: 客户端名称，用于缓存
            **overrides: 覆盖默认参数，如 decode_responses=False

        Returns:
            redis.Redis: Redis客户端实例
        """
        if name in self._named_initialized:
            return self._named_clients[name]

        async with self._lock:
            # 双重检查锁定
            if name in self._named_initialized:
                return self._named_clients[name]

            try:
                # 构建连接参数，使用默认值 + 覆盖参数
                conn_params = {
                    "max_connections": self.max_connections,
                    "socket_timeout": self.socket_timeout,
                    "socket_connect_timeout": self.socket_connect_timeout,
                    "decode_responses": True,  # 默认值
                }
                conn_params.update(overrides)
                logger.info("构建Redis客户端连接参数: %s, %s", name, conn_params)
                # 创建命名连接池
                named_pool = ConnectionPool.from_url(self.redis_url, **conn_params)

                # 创建命名Redis客户端
                named_client = redis.Redis(connection_pool=named_pool)

                # 测试连接
                await named_client.ping()

                # 缓存客户端和连接池
                self._named_clients[name] = named_client
                self._named_pools[name] = named_pool
                self._named_initialized.add(name)

                logger.info(
                    "命名Redis客户端初始化成功: %s (参数覆盖: %s)",
                    name,
                    overrides if overrides else "无",
                )

                return named_client

            except Exception as e:
                logger.error("命名Redis客户端初始化失败: %s, error=%s", name, str(e))
                # 清理部分初始化的资源
                if name in self._named_clients:
                    await self._named_clients[name].aclose()
                    del self._named_clients[name]
                if name in self._named_pools:
                    await self._named_pools[name].aclose()
                    del self._named_pools[name]
                self._named_initialized.discard(name)
                raise

    async def set(
        self,
        key: str,
        value: Union[str, bytes, int, float],
        ex: Optional[int] = None,
        nx: bool = False,
    ) -> bool:
        """
        设置键值对

        Args:
            key: 键名
            value: 值
            ex: 过期时间（秒）
            nx: 如果为True，只有键不存在时才设置

        Returns:
            bool: 设置是否成功
        """
        client = await self.get_client()
        try:
            result = await client.set(key, value, ex=ex, nx=nx)
            return result is not None and result is not False
        except Exception as e:
            logger.error("Redis SET操作失败: key=%s, error=%s", key, str(e))
            return False

    async def get(self, key: str) -> Optional[str]:
        """
        获取键值

        Args:
            key: 键名

        Returns:
            Optional[str]: 键值，不存在时返回None
        """
        client = await self.get_client()
        try:
            return await client.get(key)
        except Exception as e:
            logger.error("Redis GET操作失败: key=%s, error=%s", key, str(e))
            return None

    async def exists(self, key: str) -> bool:
        """
        检查键是否存在

        Args:
            key: 键名

        Returns:
            bool: 键是否存在
        """
        client = await self.get_client()
        try:
            result = await client.exists(key)
            return result > 0
        except Exception as e:
            logger.error("Redis EXISTS操作失败: key=%s, error=%s", key, str(e))
            return False

    async def delete(self, *keys: str) -> int:
        """
        删除键

        Args:
            keys: 要删除的键名列表

        Returns:
            int: 成功删除的键数量
        """
        if not keys:
            return 0

        client = await self.get_client()
        try:
            return await client.delete(*keys)
        except Exception as e:
            logger.error("Redis DELETE操作失败: keys=%s, error=%s", keys, str(e))
            return 0

    async def expire(self, key: str, seconds: int) -> bool:
        """
        设置键的过期时间

        Args:
            key: 键名
            seconds: 过期时间（秒）

        Returns:
            bool: 设置是否成功
        """
        client = await self.get_client()
        try:
            return await client.expire(key, seconds)
        except Exception as e:
            logger.error(
                "Redis EXPIRE操作失败: key=%s, seconds=%s, error=%s",
                key,
                seconds,
                str(e),
            )
            return False

    async def ttl(self, key: str) -> int:
        """
        获取键的剩余生存时间

        Args:
            key: 键名

        Returns:
            int: 剩余生存时间（秒），-1表示永不过期，-2表示键不存在
        """
        client = await self.get_client()
        try:
            return await client.ttl(key)
        except Exception as e:
            logger.error("Redis TTL操作失败: key=%s, error=%s", key, str(e))
            return -2

    async def keys(self, pattern: str) -> list:
        """
        获取匹配模式的键列表

        Args:
            pattern: 匹配模式（如 "prefix:*"）

        Returns:
            list: 匹配的键列表
        """
        client = await self.get_client()
        try:
            return await client.keys(pattern)
        except Exception as e:
            logger.error("Redis KEYS操作失败: pattern=%s, error=%s", pattern, str(e))
            return []

    async def ping(self) -> bool:
        """
        测试Redis连接

        Returns:
            bool: 连接是否正常
        """
        try:
            client = await self.get_client()
            result = await client.ping()
            return result is True
        except Exception as e:
            logger.error("Redis PING失败: %s", str(e))
            return False

    async def lpush(self, key: str, *values: Union[str, bytes]) -> int:
        """
        向列表左侧推入元素

        Args:
            key: 键名
            values: 要推入的值列表

        Returns:
            int: 推入后列表的长度
        """
        if not values:
            return 0

        client = await self.get_client()
        try:
            return await client.lpush(key, *values)
        except Exception as e:
            logger.error("Redis LPUSH操作失败: key=%s, error=%s", key, str(e))
            return 0

    async def lrange(self, key: str, start: int, end: int) -> list:
        """
        获取列表指定范围内的元素

        Args:
            key: 键名
            start: 起始索引
            end: 结束索引（-1表示到列表末尾）

        Returns:
            list: 元素列表
        """
        client = await self.get_client()
        try:
            return await client.lrange(key, start, end)
        except Exception as e:
            logger.error("Redis LRANGE操作失败: key=%s, error=%s", key, str(e))
            return []

    async def close(self):
        """关闭所有Redis连接池"""
        # 关闭所有命名客户端
        for name, client in self._named_clients.items():
            try:
                await client.aclose()
                logger.info("命名Redis客户端已关闭: %s", name)
            except Exception as e:
                logger.error("关闭命名Redis客户端失败: %s, error=%s", name, str(e))

        for name, pool in self._named_pools.items():
            try:
                await pool.aclose()
                logger.info("命名Redis连接池已关闭: %s", name)
            except Exception as e:
                logger.error("关闭命名Redis连接池失败: %s, error=%s", name, str(e))

        # 清理缓存
        self._named_clients.clear()
        self._named_pools.clear()
        self._named_initialized.clear()

    def is_initialized(self) -> bool:
        """检查默认客户端是否已初始化"""
        return "default" in self._named_initialized
