"""
Redis分布式锁实现

基于Redis实现的支持协程级别可重入的分布式锁服务
使用contextvar来管理协程上下文，确保线程安全和协程安全
"""

import asyncio
from typing import Optional, Union
from contextlib import asynccontextmanager

from core.di.decorators import component
from core.observation.logger import get_logger
from component.redis_provider import RedisProvider
from core.di.utils import get_bean_by_type

logger = get_logger(__name__)

# 默认配置常量
DEFAULT_LOCK_TIMEOUT = 60.0  # 默认锁超时时间（秒）
DEFAULT_BLOCKING_TIMEOUT = 80.0  # 默认阻塞获取锁的超时时间（秒）
DEFAULT_RETRY_INTERVAL = 3  # 默认重试间隔（秒）


class DistributedLockError(Exception):
    """分布式锁相关异常"""


class RedisDistributedLock:
    """
    Redis分布式可重入锁

    单个锁的实例，负责特定资源的锁操作
    """

    def __init__(self, resource: str, lock_manager: 'RedisDistributedLockManager'):
        """
        初始化分布式锁

        Args:
            resource: 锁的资源名称
            lock_manager: 锁管理器实例
        """
        self.resource = resource
        self.lock_manager = lock_manager
        self._acquired = False
        self._reentry_count = 0

    @asynccontextmanager
    async def acquire(
        self, timeout: Optional[float] = None, blocking_timeout: Optional[float] = None
    ):
        """
        获取锁的异步上下文管理器

        Args:
            timeout: 锁超时时间（秒）
            blocking_timeout: 阻塞获取锁的超时时间（秒）

        Yields:
            bool: 是否成功获取锁
        """
        timeout = timeout or DEFAULT_LOCK_TIMEOUT
        blocking_timeout = blocking_timeout or DEFAULT_BLOCKING_TIMEOUT

        acquired = False
        try:
            # 调用锁管理器的内部方法获取锁
            acquired = await self.lock_manager._acquire_lock(  # pylint: disable=protected-access
                self.resource, timeout, blocking_timeout
            )
            if acquired:
                self._acquired = True

            yield acquired

        finally:
            if acquired:
                try:
                    # 调用锁管理器的内部方法释放锁
                    await self.lock_manager._release_lock(
                        self.resource
                    )  # pylint: disable=protected-access
                    self._acquired = False
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.error("释放锁失败: %s, error: %s", self.resource, e)

    async def is_locked(self) -> bool:
        """检查锁是否被持有"""
        return await self.lock_manager.is_locked(self.resource)

    async def is_owned_by_current_coroutine(self) -> bool:
        """检查锁是否被当前协程持有"""
        return await self.lock_manager.is_owned_by_current_coroutine(self.resource)

    async def get_reentry_count(self) -> int:
        """获取当前协程的重入计数"""
        return await self.lock_manager.get_reentry_count(self.resource)


@component(name="redis_distributed_lock_manager")
class RedisDistributedLockManager:
    """
    Redis分布式锁管理器

    负责管理多个锁实例，提供锁的创建和全局操作
    """

    # 锁键模板
    LOCK_KEY_TEMPLATE = "reentrant_lock:{resource}"

    # Lua脚本：获取可重入锁
    LUA_ACQUIRE_SCRIPT = """
        local lock_key = KEYS[1]
        local owner_id = ARGV[1]
        local timeout_ms = tonumber(ARGV[2])
        
        -- 获取当前锁信息
        -- 注意：当lock_key不存在时，HMGET返回{false, false}
        local lock_info = redis.call('HMGET', lock_key, 'owner', 'count')
        local current_owner = lock_info[1]  -- 不存在时为false
        local current_count = tonumber(lock_info[2]) or 0  -- tonumber(false)为nil，使用0作为默认值
        
        if current_owner == false or current_owner == owner_id then
            -- 锁未被占用（current_owner == false）或被当前协程持有，可以获取/重入
            local new_count = current_count + 1
            redis.call('HMSET', lock_key, 'owner', owner_id, 'count', new_count)
            if timeout_ms > 0 then
                redis.call('PEXPIRE', lock_key, timeout_ms)
            end
            return new_count
        else
            -- 锁被其他协程持有
            return 0
        end
    """

    # Lua脚本：释放可重入锁
    LUA_RELEASE_SCRIPT = """
        local lock_key = KEYS[1]
        local owner_id = ARGV[1]
        
        -- 获取当前锁信息
        -- 注意：当lock_key不存在时，HMGET返回{false, false}
        local lock_info = redis.call('HMGET', lock_key, 'owner', 'count')
        local current_owner = lock_info[1]  -- 不存在时为false
        local current_count = tonumber(lock_info[2]) or 0  -- tonumber(false)为nil，使用0作为默认值
        
        if current_owner ~= owner_id then
            -- 不是锁的持有者，无法释放 或者 锁不存在
            return 0
        end
        
        local new_count = current_count - 1
        if new_count <= 0 then
            -- 重入计数归零，完全释放锁
            redis.call('DEL', lock_key)
            return -1
        else
            -- 减少重入计数，但保持锁
            redis.call('HSET', lock_key, 'count', new_count)
            return new_count
        end
    """

    # Lua脚本：检查锁状态
    LUA_STATUS_SCRIPT = """
        local lock_key = KEYS[1]
        local owner_id = ARGV[1]
        
        -- 获取当前锁信息
        -- 注意：当lock_key不存在时，HMGET返回{false, false}
        local lock_info = redis.call('HMGET', lock_key, 'owner', 'count')
        local current_owner = lock_info[1]  -- 不存在时为false
        local current_count = tonumber(lock_info[2]) or 0  -- tonumber(false)为nil，使用0作为默认值
        
        if current_owner == false then
            return {0, 0}  -- 未锁定
        elseif current_owner == owner_id then
            return {1, current_count}  -- 被当前协程持有
        else
            return {2, current_count}  -- 被其他协程持有
        end
    """

    def __init__(self, redis_provider: RedisProvider):
        """
        初始化Redis分布式锁管理器

        Args:
            redis_provider: Redis提供者
        """
        self.redis_provider = redis_provider

        # Lua脚本缓存
        self._lua_acquire = None
        self._lua_release = None
        self._lua_status = None

    def get_lock(self, resource: str) -> RedisDistributedLock:
        """
        获取指定资源的锁实例

        Args:
            resource: 锁的资源名称

        Returns:
            RedisDistributedLock: 锁实例
        """
        return RedisDistributedLock(resource, self)

    async def _ensure_scripts(self):
        """确保Lua脚本已注册"""
        if self._lua_acquire is None:
            redis_client = await self.redis_provider.get_client()
            self._lua_acquire = redis_client.register_script(self.LUA_ACQUIRE_SCRIPT)
            self._lua_release = redis_client.register_script(self.LUA_RELEASE_SCRIPT)
            self._lua_status = redis_client.register_script(self.LUA_STATUS_SCRIPT)

    def _get_owner_id(self) -> str:
        """
        获取当前协程的唯一标识符

        Returns:
            str: 协程唯一标识符

        Raises:
            DistributedLockError: 如果不在协程环境中
        """
        # 首先尝试从上下文变量获取
        try:
            current_task = asyncio.current_task()
            if current_task is None:
                raise DistributedLockError(
                    "分布式锁必须在协程环境中使用，当前没有运行的协程任务"
                )

            # 使用任务的id作为协程标识
            task_id = id(current_task)
            owner_id = f"task_{task_id}"
            return owner_id

        except RuntimeError as e:
            raise DistributedLockError(f"分布式锁必须在协程环境中使用: {e}") from e

    async def _acquire_lock(
        self, resource: str, timeout: float, blocking_timeout: float
    ) -> bool:
        """
        内部方法：获取锁

        Args:
            resource: 资源名称
            timeout: 锁超时时间
            blocking_timeout: 阻塞获取锁的超时时间

        Returns:
            bool: 是否成功获取锁
        """
        await self._ensure_scripts()

        lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
        owner_id = self._get_owner_id()
        timeout_ms = int(timeout * 1000) if timeout > 0 else 0

        # 计算重试次数
        retry_count = max(1, int(blocking_timeout / DEFAULT_RETRY_INTERVAL))

        for attempt in range(retry_count):
            try:
                redis_client = await self.redis_provider.get_client()
                result = await self._lua_acquire(
                    keys=[lock_key], args=[owner_id, timeout_ms], client=redis_client
                )

                if result > 0:
                    logger.debug(
                        "成功获取可重入锁: %s, 协程: %s, 重入计数: %s (attempt %s)",
                        resource,
                        owner_id,
                        result,
                        attempt + 1,
                    )
                    return True
                else:
                    if attempt < retry_count - 1:
                        await asyncio.sleep(DEFAULT_RETRY_INTERVAL)

            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(
                    "获取锁失败 (attempt %s): %s, error: %s", attempt + 1, resource, e
                )
                if attempt < retry_count - 1:
                    await asyncio.sleep(DEFAULT_RETRY_INTERVAL)

        logger.warning("获取可重入分布式锁超时: %s, 协程: %s", resource, owner_id)
        return False

    async def _release_lock(self, resource: str):
        """
        内部方法：释放锁

        Args:
            resource: 资源名称
        """
        lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
        owner_id = self._get_owner_id()

        try:
            redis_client = await self.redis_provider.get_client()
            result = await self._lua_release(
                keys=[lock_key], args=[owner_id], client=redis_client
            )

            if result == -1:
                logger.debug("完全释放可重入锁: %s, 协程: %s", resource, owner_id)
            elif result > 0:
                logger.debug(
                    "减少可重入锁计数: %s, 协程: %s, 剩余计数: %s",
                    resource,
                    owner_id,
                    result,
                )
            else:
                logger.warning(
                    "无法释放不属于当前协程的锁或者锁不存在: %s, 协程: %s",
                    resource,
                    owner_id,
                )

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(
                "释放可重入锁时发生异常: %s, 协程: %s, error: %s", resource, owner_id, e
            )

    async def is_locked(self, resource: str) -> bool:
        """
        检查资源是否被锁定

        Args:
            resource: 锁的资源名称

        Returns:
            bool: 是否被锁定
        """
        try:
            await self._ensure_scripts()
            redis_client = await self.redis_provider.get_client()
            lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
            owner_id = self._get_owner_id()

            result = await self._lua_status(
                keys=[lock_key], args=[owner_id], client=redis_client
            )

            status_code = result[0] if result else 0
            return status_code > 0  # 1或2都表示被锁定

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("检查可重入锁状态失败: %s, error: %s", resource, e)
            return False

    async def is_owned_by_current_coroutine(self, resource: str) -> bool:
        """
        检查锁是否被当前协程持有

        Args:
            resource: 锁的资源名称

        Returns:
            bool: 是否被当前协程持有
        """
        try:
            await self._ensure_scripts()
            redis_client = await self.redis_provider.get_client()
            lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
            owner_id = self._get_owner_id()

            result = await self._lua_status(
                keys=[lock_key], args=[owner_id], client=redis_client
            )

            status_code = result[0] if result else 0
            return status_code == 1  # 1表示被当前协程持有

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("检查可重入锁所有权失败: %s, error: %s", resource, e)
            return False

    async def get_reentry_count(self, resource: str) -> int:
        """
        获取当前协程对指定资源的重入计数

        Args:
            resource: 锁的资源名称

        Returns:
            int: 重入计数，0表示未持有锁
        """
        try:
            await self._ensure_scripts()
            redis_client = await self.redis_provider.get_client()
            lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
            owner_id = self._get_owner_id()

            result = await self._lua_status(
                keys=[lock_key], args=[owner_id], client=redis_client
            )

            if result and result[0] == 1:  # 被当前协程持有
                return result[1]
            else:
                return 0

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("获取重入计数失败: %s, error: %s", resource, e)
            return 0

    async def force_unlock(self, resource: str) -> bool:
        """
        强制释放锁（慎用）

        Args:
            resource: 锁的资源名称

        Returns:
            bool: 是否成功释放
        """
        try:
            redis_client = await self.redis_provider.get_client()
            lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
            result = await redis_client.delete(lock_key)

            logger.warning("强制释放可重入锁: %s, result: %s", resource, result)
            return result > 0

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("强制释放可重入锁失败: %s, error: %s", resource, e)
            return False

    @asynccontextmanager
    async def acquire_lock(
        self,
        resource: str,
        timeout: Optional[float] = None,
        blocking_timeout: Optional[float] = None,
    ):
        """
        获取可重入分布式锁的异步上下文管理器（兼容旧接口）

        Args:
            resource: 锁的资源名称（键名）
            timeout: 锁超时时间（秒）
            blocking_timeout: 阻塞获取锁的超时时间（秒）

        Yields:
            bool: 是否成功获取锁
        """
        lock = self.get_lock(resource)
        async with lock.acquire(timeout, blocking_timeout) as acquired:
            yield acquired

    async def close(self):
        """关闭服务并清理资源"""
        logger.info("Redis分布式锁管理器关闭")


# 便捷的上下文管理器函数
@asynccontextmanager
async def distributed_lock(
    resource: str,
    timeout: Optional[float] = None,
    blocking_timeout: Optional[float] = None,
):
    """
    便捷的分布式锁上下文管理器，用于在函数内部使用

    Args:
        resource: 锁的资源名称
        timeout: 锁超时时间（秒）
        blocking_timeout: 阻塞获取锁的超时时间（秒）

    Yields:
        bool: 是否成功获取锁

    Example:
        async def some_function():
            async with distributed_lock("user:balance:123") as acquired:
                if acquired:
                    # 执行需要加锁的代码
                    print("已获取锁，执行业务逻辑")
                else:
                    print("获取锁失败")

        # 支持可重入
        async def reentrant_function():
            async with distributed_lock("resource:123") as acquired1:
                if acquired1:
                    print("第一层获取锁")
                    async with distributed_lock("resource:123") as acquired2:
                        if acquired2:
                            print("第二层获取锁（可重入）")
    """

    # 获取锁管理器
    lock_manager = get_bean_by_type(RedisDistributedLockManager)

    # 获取锁并执行
    lock = lock_manager.get_lock(resource)
    async with lock.acquire(
        timeout=timeout, blocking_timeout=blocking_timeout
    ) as acquired:
        yield acquired


# 便捷的装饰器函数
def with_distributed_lock(
    resource_key: Union[str, callable],
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    blocking_timeout: float = DEFAULT_BLOCKING_TIMEOUT,
):
    """
    分布式锁装饰器（支持可重入）

    Args:
        resource_key: 锁的资源键，可以是字符串或返回字符串的函数
        timeout: 锁超时时间
        blocking_timeout: 阻塞获取锁的超时时间

    Example:
        @with_distributed_lock("user:balance:{user_id}")
        async def update_user_balance(user_id: int, amount: float):
            # 这个函数可以在同一协程中被递归调用而不会死锁
            if amount > 100:
                await update_user_balance(user_id, amount / 2)
                await update_user_balance(user_id, amount / 2)
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):

            # 获取锁管理器
            lock_manager = get_bean_by_type(RedisDistributedLockManager)

            # 计算资源键
            if callable(resource_key):
                resource = resource_key(*args, **kwargs)
            else:
                # 支持格式化字符串
                try:
                    resource = resource_key.format(*args, **kwargs)
                except (IndexError, KeyError):
                    resource = resource_key

            # 获取锁并执行函数
            lock = lock_manager.get_lock(resource)
            async with lock.acquire(
                timeout=timeout, blocking_timeout=blocking_timeout
            ) as acquired:
                if acquired:
                    return await func(*args, **kwargs)
                else:
                    raise RuntimeError(f"获取分布式锁失败: {resource}")

        return wrapper

    return decorator
