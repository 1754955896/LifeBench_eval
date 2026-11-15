"""
Redis 时间窗口缓存管理器

基于 Redis Sorted Set 实现的时间窗口缓存，支持：
- 按 key 追加数据到时间窗口队列
- 获取指定时间范围内的数据
- 自动过期清理
- 随机清理过期数据
"""

import time
import random
from typing import List, Dict, Any, Optional, Union

from datetime import datetime

from core.di.decorators import component
from core.observation.logger import get_logger
from component.redis_provider import RedisProvider
from .redis_data_processor import RedisDataProcessor

# 配置常量
DEFAULT_EXPIRE_MINUTES = 10  # 默认过期时间 10 分钟
DEFAULT_CLEANUP_PROBABILITY = 0.1  # 10% 概率执行随机清理
DEFAULT_CLEANUP_MULTIPLIER = 2  # 清理阈值倍数（清理超过过期时间2倍的数据）

# Lua 脚本：按时间戳范围获取数据（带分数）
FETCH_BY_DATE_TIMESTAMP_RANGE_LUA_SCRIPT = """
local queue_key = KEYS[1]
local min_score = ARGV[1]  -- 最小时间戳
local max_score = ARGV[2]  -- 最大时间戳
local limit = tonumber(ARGV[3]) or -1  -- 限制数量，-1表示不限制

-- 按score范围获取数据和分数，按时间戳升序排列
local messages
if limit > 0 then
    messages = redis.call('ZRANGEBYSCORE', queue_key, min_score, max_score, 'WITHSCORES', 'LIMIT', 0, limit)
else
    messages = redis.call('ZRANGEBYSCORE', queue_key, min_score, max_score, 'WITHSCORES')
end

return messages
"""

logger = get_logger(__name__)


@component(name="redis_windows_cache_factory")
class RedisWindowsCacheFactory:
    """Redis 时间窗口缓存管理器工厂"""

    def __init__(self, redis_provider: RedisProvider):
        """
        初始化缓存工厂

        Args:
            redis_provider: Redis 连接提供者
        """
        self.redis_provider = redis_provider
        self._timestamp_range_script = None
        logger.info("Redis 时间窗口缓存工厂初始化完成")

    async def _ensure_timestamp_range_script_registered(self):
        """确保时间戳范围查询 Lua 脚本已注册（只注册一次）"""
        if self._timestamp_range_script is None:
            # 使用binary_cache客户端注册脚本，确保与执行时使用相同的连接
            binary_client = await self.redis_provider.get_named_client(
                "binary_cache", decode_responses=False
            )
            self._timestamp_range_script = binary_client.register_script(
                FETCH_BY_DATE_TIMESTAMP_RANGE_LUA_SCRIPT
            )
            logger.info("时间戳范围查询 Lua 脚本注册完成")
        return self._timestamp_range_script

    async def create_cache_manager(
        self,
        expire_minutes: int = DEFAULT_EXPIRE_MINUTES,
        cleanup_probability: float = DEFAULT_CLEANUP_PROBABILITY,
    ) -> 'RedisWindowsCacheManager':
        """
        创建缓存管理器实例

        Args:
            expire_minutes: 默认过期时间（分钟）
            cleanup_probability: 随机清理概率 (0.0-1.0)

        Returns:
            RedisWindowsCacheManager: 缓存管理器实例
        """
        timestamp_range_script = await self._ensure_timestamp_range_script_registered()
        return RedisWindowsCacheManager(
            redis_provider=self.redis_provider,
            fetch_by_timestamp_range_script=timestamp_range_script,
            expire_minutes=expire_minutes,
            cleanup_probability=cleanup_probability,
        )


class RedisWindowsCacheManager:
    """
    Redis 时间窗口缓存管理器

    基于 Redis Sorted Set (ZSET) 实现的时间窗口队列缓存：
    - Score: 毫秒级时间戳，用于时间排序
    - Member: 唯一标识符:数据内容，确保数据唯一性
    - 支持自定义过期时间，每次 append 时延长
    - 提供随机清理机制，避免内存无限增长
    """

    def __init__(
        self,
        redis_provider: RedisProvider,
        fetch_by_timestamp_range_script,
        expire_minutes: int = DEFAULT_EXPIRE_MINUTES,
        cleanup_probability: float = DEFAULT_CLEANUP_PROBABILITY,
    ):
        """
        初始化 Redis 时间窗口缓存管理器

        Args:
            redis_provider: Redis 连接提供者
            fetch_by_timestamp_range_script: 预注册的时间戳范围查询 Lua 脚本对象
            expire_minutes: 默认过期时间（分钟）
            cleanup_probability: 随机清理概率 (0.0-1.0)
        """
        self.redis_provider = redis_provider
        self.default_expire_minutes = expire_minutes
        self.cleanup_probability = cleanup_probability
        self._fetch_by_timestamp_range_script = fetch_by_timestamp_range_script

        logger.info(
            "Redis 时间窗口缓存管理器初始化完成: expire=%d分钟, cleanup_prob=%.1f%%",
            expire_minutes,
            cleanup_probability * 100,
        )

    def _convert_timestamp(self, timestamp: Optional[Union[int, datetime]]) -> int:
        """
        将时间戳转换为毫秒级整数

        Args:
            timestamp: 时间戳（毫秒）或datetime对象，如果不提供则使用当前时间

        Returns:
            int: 毫秒级时间戳
        """
        if timestamp is None:
            return int(time.time() * 1000)
        elif isinstance(timestamp, datetime):
            # 如果是datetime对象，转换为毫秒时间戳
            return int(timestamp.timestamp() * 1000)
        else:
            # 如果是整数时间戳
            return int(timestamp)

    async def _cleanup_expired_if_needed(self, key: str, client) -> int:
        """
        根据概率决定是否执行过期数据清理

        Args:
            key: 缓存键名
            client: Redis客户端

        Returns:
            int: 清理的数据条数
        """
        if random.random() < self.cleanup_probability:
            # 计算清理阈值（默认过期时间的倍数之前的数据）
            current_timestamp = int(time.time() * 1000)
            cleanup_threshold = current_timestamp - (
                self.default_expire_minutes * DEFAULT_CLEANUP_MULTIPLIER * 60 * 1000
            )

            # 清理过期数据
            cleaned_count = await client.zremrangebyscore(
                key, '-inf', cleanup_threshold
            )

            if cleaned_count > 0:
                logger.debug("随机清理过期数据: key=%s, cleaned=%d", key, cleaned_count)

            return cleaned_count
        return 0

    async def append(
        self,
        key: str,
        data: Union[str, Dict, List, Any],
        expire_minutes: Optional[int] = None,
    ) -> bool:
        """
        向指定 key 的时间窗口队列追加数据

        Args:
            key: 缓存键名
            data: 要追加的数据，支持字符串、字典、列表等
            expire_minutes: 过期时间（分钟），默认使用实例配置

        Returns:
            bool: 操作是否成功

        Examples:
            # 追加字符串数据
            await cache.append("user_actions", "user_login")

            # 追加字典数据
            await cache.append("api_calls", {"method": "GET", "path": "/api/users"})

            # 自定义过期时间
            await cache.append("temp_data", "some_data", expire_minutes=5)
        """
        try:
            client = await self.redis_provider.get_client()

            # 获取当前毫秒级时间戳
            current_timestamp = int(time.time() * 1000)

            # 使用数据处理器处理数据
            unique_member = RedisDataProcessor.process_data_for_storage(data)

            # 使用 pipeline 进行原子操作
            pipe = client.pipeline()

            # 1. 添加数据到有序集合
            pipe.zadd(key, {unique_member: current_timestamp})

            # 2. 设置或延长过期时间
            expire_seconds = (expire_minutes or self.default_expire_minutes) * 60
            pipe.expire(key, expire_seconds)

            # 3. 随机清理过期数据（使用独立的清理函数）
            # 注意：这里不能在pipeline中调用，因为需要获取返回值
            # 所以先执行pipeline，再单独执行清理

            # 执行管道操作
            results = await pipe.execute()

            # 检查操作结果
            zadd_result = results[0]  # ZADD 返回添加的成员数量
            expire_result = results[1]  # EXPIRE 返回是否成功

            # 单独执行随机清理
            cleaned_count = await self._cleanup_expired_if_needed(key, client)

            if zadd_result is not None and expire_result:
                if cleaned_count > 0:
                    logger.debug(
                        "数据追加成功并清理了过期数据: key=%s, member_length=%d, timestamp=%d, expire=%ds, cleaned=%d",
                        key,
                        len(unique_member),
                        current_timestamp,
                        expire_seconds,
                        cleaned_count,
                    )
                else:
                    logger.debug(
                        "数据追加成功: key=%s, member_length=%d, timestamp=%d, expire=%ds",
                        key,
                        len(unique_member),
                        current_timestamp,
                        expire_seconds,
                    )
                return True
            else:
                logger.warning(
                    "数据追加部分失败: key=%s, zadd_result=%s, expire_result=%s",
                    key,
                    zadd_result,
                    expire_result,
                )
                return False

        except Exception as e:
            logger.error("Redis 追加数据失败: key=%s, error=%s", key, str(e))
            return False

    async def get_queue_size(self, key: str) -> int:
        """
        获取指定队列的当前大小

        Args:
            key: 缓存键名

        Returns:
            int: 队列中的元素数量
        """
        try:
            client = await self.redis_provider.get_client()
            size = await client.zcard(key)
            return size or 0
        except Exception as e:
            logger.error("获取队列大小失败: key=%s, error=%s", key, str(e))
            return 0

    async def clear_queue(self, key: str) -> bool:
        """
        清空指定队列的所有数据

        Args:
            key: 缓存键名

        Returns:
            bool: 操作是否成功
        """
        try:
            client = await self.redis_provider.get_client()
            result = await client.delete(key)
            logger.info("清空队列: key=%s, result=%d", key, result)
            return result > 0
        except Exception as e:
            logger.error("清空队列失败: key=%s, error=%s", key, str(e))
            return False

    async def cleanup_expired(self, key: str) -> int:
        """
        手动清理指定队列中的过期数据

        Args:
            key: 缓存键名

        Returns:
            int: 清理的数据条数
        """
        try:
            client = await self.redis_provider.get_client()

            # 强制执行清理（忽略概率）
            # 计算清理阈值（默认过期时间的倍数之前的数据）
            current_timestamp = int(time.time() * 1000)
            cleanup_threshold = current_timestamp - (
                self.default_expire_minutes * DEFAULT_CLEANUP_MULTIPLIER * 60 * 1000
            )

            # 清理过期数据
            cleaned_count = await client.zremrangebyscore(
                key, '-inf', cleanup_threshold
            )

            if cleaned_count > 0:
                logger.info("手动清理过期数据: key=%s, cleaned=%d", key, cleaned_count)

            return cleaned_count

        except Exception as e:
            logger.error("清理过期数据失败: key=%s, error=%s", key, str(e))
            return 0

    async def get_by_timestamp_range(
        self,
        key: str,
        start_timestamp: Optional[Union[int, datetime]] = None,
        end_timestamp: Optional[Union[int, datetime]] = None,
        limit: int = -1,
    ) -> List[Dict[str, Any]]:
        """
        根据时间戳范围获取队列数据

        Args:
            key: 缓存键名
            start_timestamp: 开始时间戳（毫秒）或datetime对象，None表示不限制
            end_timestamp: 结束时间戳（毫秒）或datetime对象，None表示不限制
            limit: 限制返回数量，-1表示不限制

        Returns:
            List[Dict[str, Any]]: 时间戳范围内的数据列表，每个元素包含：
                - id: 唯一标识符
                - data: 原始数据
                - timestamp: 时间戳（毫秒）
                - datetime: 格式化的时间字符串
        """
        try:
            # 转换时间戳
            min_score = "-inf"
            max_score = "+inf"

            if start_timestamp is not None:
                min_score = str(self._convert_timestamp(start_timestamp))

            if end_timestamp is not None:
                max_score = str(self._convert_timestamp(end_timestamp))

            # 执行预注册的 Lua 脚本
            messages = await self._fetch_by_timestamp_range_script(
                keys=[key], args=[min_score, max_score, limit]
            )

            if not messages:
                logger.debug(
                    "时间戳范围内无数据: key=%s, range=[%s, %s]",
                    key,
                    min_score,
                    max_score,
                )
                return []

            # 解析消息数据（WITHSCORES返回格式：[member1, score1, member2, score2, ...]）
            result = []

            # 处理WITHSCORES返回的结果，每两个元素为一组：消息内容和分数
            if len(messages) % 2 != 0:
                logger.warning(
                    "WITHSCORES返回数据长度异常: %d，应该是偶数", len(messages)
                )
                return []

            for i in range(0, len(messages), 2):
                try:
                    if i + 1 >= len(messages):
                        logger.warning(
                            "WITHSCORES数据索引越界: i=%d, len=%d", i, len(messages)
                        )
                        break

                    message = messages[i]  # 消息内容
                    score_raw = messages[i + 1]  # 分数（可能是bytes或字符串）

                    # 安全地转换分数为时间戳
                    try:
                        if isinstance(score_raw, bytes):
                            score_str = score_raw.decode('utf-8')
                        else:
                            score_str = str(score_raw)
                        timestamp = int(float(score_str))
                    except (ValueError, UnicodeDecodeError) as score_e:
                        logger.warning(
                            "分数转换失败: score_raw=%s (type=%s), error=%s",
                            score_raw,
                            type(score_raw),
                            str(score_e),
                        )
                        timestamp = int(time.time() * 1000)  # 使用当前时间作为fallback

                    # 使用数据处理器解析数据
                    processed_data = RedisDataProcessor.process_data_from_storage(
                        message
                    )

                    result.append(
                        {
                            "id": processed_data["id"],
                            "data": processed_data["data"],
                            "timestamp": timestamp,
                            "datetime": datetime.fromtimestamp(
                                timestamp / 1000
                            ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        }
                    )

                except Exception as e:
                    logger.warning(
                        "解析消息失败: i=%d, message=%s, error=%s",
                        i,
                        message if i < len(messages) else "索引越界",
                        str(e),
                    )
                    continue

            # 按时间戳排序（最新的在前）
            result.sort(key=lambda x: x["timestamp"], reverse=True)

            logger.debug(
                "按时间戳范围获取数据成功: key=%s, range=[%s, %s], count=%d",
                key,
                min_score,
                max_score,
                len(result),
            )

            return result

        except Exception as e:
            logger.error("按时间戳范围获取数据失败: key=%s, error=%s", key, str(e))
            return []

    async def get_queue_stats(self, key: str) -> Dict[str, Any]:
        """
        获取队列的统计信息

        Args:
            key: 缓存键名

        Returns:
            Dict[str, Any]: 队列统计信息
        """
        try:
            client = await self.redis_provider.get_client()

            # 获取队列大小
            total_count = await client.zcard(key) or 0

            if total_count == 0:
                return {
                    "key": key,
                    "total_count": 0,
                    "oldest_timestamp": None,
                    "newest_timestamp": None,
                    "ttl_seconds": -1,
                }

            # 获取最旧和最新的时间戳
            oldest_data = await client.zrange(key, 0, 0, withscores=True)
            newest_data = await client.zrange(key, -1, -1, withscores=True)

            oldest_timestamp = int(oldest_data[0][1]) if oldest_data else None
            newest_timestamp = int(newest_data[0][1]) if newest_data else None

            # 获取 TTL
            ttl = await client.ttl(key)

            return {
                "key": key,
                "total_count": total_count,
                "oldest_timestamp": oldest_timestamp,
                "newest_timestamp": newest_timestamp,
                "oldest_datetime": (
                    datetime.fromtimestamp(oldest_timestamp / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if oldest_timestamp
                    else None
                ),
                "newest_datetime": (
                    datetime.fromtimestamp(newest_timestamp / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if newest_timestamp
                    else None
                ),
                "ttl_seconds": ttl,
            }

        except Exception as e:
            logger.error("获取队列统计失败: key=%s, error=%s", key, str(e))
            return {
                "key": key,
                "total_count": 0,
                "oldest_timestamp": None,
                "newest_timestamp": None,
                "ttl_seconds": -1,
                "error": str(e),
            }


# 为了保持向后兼容，提供一个默认的组件实例
@component(name="redis_windows_cache_manager")
class DefaultRedisWindowsCacheManager:
    """默认的 Redis 时间窗口缓存管理器（向后兼容）"""

    def __init__(self, redis_provider: RedisProvider):
        self.redis_provider = redis_provider
        self._manager = None
        self._factory = None

    async def _get_manager(self):
        """延迟初始化管理器"""
        if self._manager is None:
            if self._factory is None:
                self._factory = RedisWindowsCacheFactory(self.redis_provider)
            self._manager = await self._factory.create_cache_manager()
        return self._manager

    async def append(
        self,
        key: str,
        data: Union[str, Dict, List, Any],
        expire_minutes: Optional[int] = None,
    ) -> bool:
        manager = await self._get_manager()
        return await manager.append(key, data, expire_minutes)

    async def get_queue_size(self, key: str) -> int:
        manager = await self._get_manager()
        return await manager.get_queue_size(key)

    async def clear_queue(self, key: str) -> bool:
        manager = await self._get_manager()
        return await manager.clear_queue(key)

    async def cleanup_expired(self, key: str) -> int:
        manager = await self._get_manager()
        return await manager.cleanup_expired(key)

    async def get_queue_stats(self, key: str) -> Dict[str, Any]:
        manager = await self._get_manager()
        return await manager.get_queue_stats(key)

    async def get_by_timestamp_range(
        self,
        key: str,
        start_timestamp: Optional[Union[int, datetime]] = None,
        end_timestamp: Optional[Union[int, datetime]] = None,
        limit: int = -1,
    ) -> List[Dict[str, Any]]:
        manager = await self._get_manager()
        return await manager.get_by_timestamp_range(
            key, start_timestamp, end_timestamp, limit
        )
