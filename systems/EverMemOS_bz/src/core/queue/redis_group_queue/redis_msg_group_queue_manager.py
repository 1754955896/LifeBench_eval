"""
Redis消息分组队列管理器

基于Redis实现的固定分区队列管理器。
核心特性：
1. 固定50个分区，编号001-050，不可配置
2. group_key通过hash路由到固定分区
3. 支持多队列并发消费，基于owner机制防冲突
4. 使用Redis的有序集合(ZSET)存储消息，支持按分数排序和时间过滤

⚠️ 警告：分区数量固定为50，修改此配置会导致严重的数据路由错误和消息丢失！
"""

import asyncio
import time
import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Callable, Type
from dataclasses import dataclass, field
from enum import Enum

import redis.asyncio as redis

from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from core.queue.redis_group_queue.redis_group_queue_item import SimpleQueueItem
from core.queue.redis_group_queue.redis_group_queue_item import (
    RedisGroupQueueItem,
    SerializationMode,
)
from core.queue.redis_group_queue.redis_group_queue_lua_scripts import (
    ENQUEUE_SCRIPT,
    GET_QUEUE_STATS_SCRIPT,
    GET_ALL_PARTITIONS_STATS_SCRIPT,
    REBALANCE_PARTITIONS_SCRIPT,
    JOIN_CONSUMER_SCRIPT,
    EXIT_CONSUMER_SCRIPT,
    KEEPALIVE_CONSUMER_SCRIPT,
    CLEANUP_INACTIVE_OWNERS_SCRIPT,
    FORCE_CLEANUP_SCRIPT,
    GET_MESSAGES_SCRIPT,
)
from core.rate_limit.rate_limiter import rate_limit

logger = get_logger(__name__)


class ShutdownMode(Enum):
    """关闭模式枚举"""

    SOFT = "soft"  # 软性关闭：检查是否有消息，有延迟时间控制
    HARD = "hard"  # 硬性关闭：直接关闭，记录未处理消息数


class ManagerState(Enum):
    """管理器状态枚举"""

    CREATED = "created"  # 已创建，未启动
    STARTED = "started"  # 已启动，正在运行
    SHUTDOWN = "shutdown"  # 已关闭，不可再启动


@dataclass
class RedisPartitionStats:
    """Redis分区统计信息"""

    partition: str
    current_size: int
    min_score: int
    max_score: int

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "partition": self.partition,
            "current_size": self.current_size,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }


@dataclass
class RedisQueueStats:
    """Redis队列统计信息"""

    queue_name: str
    current_size: int
    last_activity_time: float
    min_score: int
    max_score: int
    total_delivered: int = 0
    total_consumed: int = 0
    last_deliver_time: Optional[str] = None
    last_consume_time: Optional[str] = None
    partitions: Optional[List[RedisPartitionStats]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "queue_name": self.queue_name,
            "current_size": self.current_size,
            "last_activity_time": self.last_activity_time,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "total_delivered": self.total_delivered,
            "total_consumed": self.total_consumed,
            "last_deliver_time": self.last_deliver_time,
            "last_consume_time": self.last_consume_time,
        }
        if self.partitions:
            result["partitions"] = [p.to_dict() for p in self.partitions]
        return result


@dataclass
class RedisManagerStats:
    """Redis管理器整体统计信息"""

    total_queues: int
    total_current_messages: int
    total_delivered_messages: int = 0
    total_consumed_messages: int = 0
    total_rejected_messages: int = 0
    start_time: str = field(
        default_factory=lambda: to_iso_format(get_now_with_timezone())
    )
    uptime_seconds: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_queues": self.total_queues,
            "total_current_messages": self.total_current_messages,
            "total_delivered_messages": self.total_delivered_messages,
            "total_consumed_messages": self.total_consumed_messages,
            "total_rejected_messages": self.total_rejected_messages,
            "start_time": self.start_time,
            "uptime_seconds": self.uptime_seconds,
        }


class RedisGroupQueueManager:
    """
    Redis消息分组队列管理器（动态owner管理版本）

    核心特性：
    1. 基于owner_activate_time_zset管理消费者活跃状态
    2. 每个owner拥有独立的queue_list，记录分配的分区
    3. 支持动态rebalance，自动分配分区给活跃消费者
    4. 消费者加入/退出自动触发rebalance
    5. 定期清理不活跃消费者（默认5分钟不活跃）
    6. 消费者保活机制（建议每30秒调用）
    7. 支持强制清理和重置
    8. 消费消息时检查score差值阈值
    9. 所有操作通过Lua脚本保证原子性
    """

    # 固定分区数量，可配置但建议保持50
    FIXED_PARTITION_COUNT = 50

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "default",
        serialization_mode: SerializationMode = SerializationMode.JSON,
        item_class: Type[RedisGroupQueueItem] = None,
        sort_key_func: Optional[Callable[[RedisGroupQueueItem], int]] = None,
        max_total_messages: int = 20000,  # 2w
        queue_expire_seconds: int = 24 * 3600,  # 1天
        activity_expire_seconds: int = 24 * 3600,  # 1天
        enable_metrics: bool = True,
        log_interval_seconds: int = 600,  # 10分钟
        owner_expire_seconds: int = 3600,  # owner过期时间，默认1小时
        inactive_threshold_seconds: int = 300,  # 不活跃阈值，默认5分钟
        cleanup_interval_seconds: int = 300,  # 定期清理间隔，默认5分钟
    ):
        """
        初始化Redis消息分组队列管理器

        Args:
            redis_client: Redis客户端
            key_prefix: Redis键前缀，用于区分不同的管理器实例
            serialization_mode: 序列化模式（JSON或BSON）
            item_class: 队列项类型，必须继承自RedisGroupQueueItem，默认使用SimpleQueueItem
            sort_key_func: 排序键生成函数，接收RedisGroupQueueItem返回int分数
            max_total_messages: 最大总消息数量限制
            queue_expire_seconds: 队列过期时间（秒）
            activity_expire_seconds: 活动记录过期时间（秒）
            enable_metrics: 是否启用统计功能
            log_interval_seconds: 日志打印间隔（秒）
            owner_expire_seconds: owner过期时间（秒，默认1小时）
            inactive_threshold_seconds: 不活跃阈值（秒，默认5分钟）
            cleanup_interval_seconds: 定期清理间隔（秒，默认5分钟）
        """
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.serialization_mode = serialization_mode
        # 设置默认的item_class为SimpleQueueItem
        if item_class is None:
            self.item_class = SimpleQueueItem
        else:
            self.item_class = item_class
        self.sort_key_func = sort_key_func or self._default_sort_key
        self.max_total_messages = max_total_messages
        self.queue_expire_seconds = queue_expire_seconds
        self.activity_expire_seconds = activity_expire_seconds
        self.enable_metrics = enable_metrics
        self.log_interval_seconds = log_interval_seconds
        self.owner_expire_seconds = owner_expire_seconds
        self.inactive_threshold_seconds = inactive_threshold_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # Redis键模式 - 新的动态owner管理模式
        self.queue_prefix = f"{key_prefix}:queue:"  # 队列键前缀，用于Lua脚本
        self.queue_key_pattern = (
            f"{key_prefix}:queue:{{partition}}"  # partition为001-050
        )
        self.owner_activate_time_zset_key = (
            f"{key_prefix}:owner_activate_time_zset"  # owner活跃时间zset
        )
        self.queue_list_prefix = f"{key_prefix}:queue_list:"  # owner的queue_list前缀
        self.counter_key = f"{key_prefix}:counter"

        # 进程级别的owner ID（启动时生成，全局唯一）
        self.owner_id = (
            f"{self.key_prefix}_{int(time.time())}_{random.randint(10000, 99999)}"
        )

        # 维护owner最后keepalive时间戳mapping（毫秒时间戳）
        self.owner_last_keepalive_time = {}

        # 生成固定分区名称列表：001, 002, ..., 050
        self.partition_names = [
            f"{i:03d}" for i in range(1, self.FIXED_PARTITION_COUNT + 1)
        ]

        # 管理器统计信息
        self._manager_stats = RedisManagerStats(
            total_queues=0, total_current_messages=0
        )

        # 异步锁，保护统计信息
        self._stats_lock = asyncio.Lock()

        # 启动时间
        self._start_time = time.time()

        # 定期任务
        self._log_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # 管理器状态
        self._state = ManagerState.CREATED

        # 预编译Lua脚本
        self._enqueue_script = None
        self._get_stats_script = None
        self._get_all_partitions_stats_script = None
        self._rebalance_partitions_script = None
        self._join_consumer_script = None
        self._exit_consumer_script = None
        self._keepalive_consumer_script = None
        self._cleanup_inactive_owners_script = None
        self._force_cleanup_script = None
        self._get_messages_script = None

        logger.info(
            "🚀 RedisGroupQueueManager[%s] 初始化完成: key_prefix=%s, max_total_messages=%d",
            self.key_prefix,
            self.key_prefix,
            self.max_total_messages,
        )

    def _default_sort_key(self, _item: RedisGroupQueueItem) -> int:
        """
        默认排序键生成函数：使用当前时间戳（毫秒）

        Args:
            item: 队列项

        Returns:
            int: 排序分数（毫秒时间戳）
        """
        return int(time.time() * 1000)  # 转换为毫秒整数

    def _hash_group_key_to_partition(self, group_key: str) -> str:
        """
        将group_key通过hash路由到固定分区

        Args:
            group_key: 分组键

        Returns:
            str: 分区名称（001-100）
        """
        # 使用MD5 hash确保分布均匀
        hash_value = hashlib.md5(group_key.encode('utf-8')).hexdigest()
        # 取前8位转为整数，再取模
        partition_index = int(hash_value[:8], 16) % self.FIXED_PARTITION_COUNT
        return self.partition_names[partition_index]

    def _get_queue_key(self, partition: str) -> str:
        """获取队列Redis键"""
        return self.queue_key_pattern.format(partition=partition)

    def _get_queue_list_key(self, owner_id: Optional[str] = None) -> str:
        """获取owner的queue_list Redis键"""
        if owner_id is None:
            owner_id = self.owner_id
        return f"{self.queue_list_prefix}{owner_id}"

    def _parse_rebalance_result(
        self, result: Any, expected_count: int
    ) -> Tuple[bool, Tuple]:
        """
        解析rebalance相关脚本的返回结果

        Args:
            result: Lua脚本返回的结果
            expected_count: 期望的返回值数量 (2 for rebalance/join/exit, 3 for cleanup)

        Returns:
            Tuple[bool, Tuple]: (是否成功, 解析后的结果)
        """
        # 检查返回结果格式
        if not isinstance(result, (list, tuple)) or len(result) < expected_count:
            logger.error(
                "❌ RedisGroupQueueManager[%s] 脚本返回格式错误: 期望%d个值，实际得到%s",
                self.key_prefix,
                expected_count,
                result,
            )
            return False, tuple([0] * expected_count)

        # 提取基本值
        if expected_count == 2:
            owner_count, assigned_partitions_flat = result
            parsed_result = (
                owner_count,
                self._convert_flat_to_dict(assigned_partitions_flat),
            )
        elif expected_count == 3:
            cleaned_count, owner_count, assigned_partitions_flat = result
            parsed_result = (
                cleaned_count,
                owner_count,
                self._convert_flat_to_dict(assigned_partitions_flat),
            )
        else:
            return False, tuple([0] * expected_count)

        return True, parsed_result

    def _convert_flat_to_dict(
        self, assigned_partitions_flat: Any
    ) -> Dict[str, List[str]]:
        """
        将扁平数组转换为字典格式

        Args:
            assigned_partitions_flat: 扁平数组 [owner_id1, [partitions1], owner_id2, [partitions2], ...]

        Returns:
            Dict[str, List[str]]: 分配结果字典
        """
        assigned_partitions = {}
        if (
            isinstance(assigned_partitions_flat, list)
            and len(assigned_partitions_flat) > 0
        ):
            for i in range(0, len(assigned_partitions_flat), 2):
                if i + 1 < len(assigned_partitions_flat):
                    owner_id = self._safe_decode_redis_value(
                        assigned_partitions_flat[i]
                    )
                    partitions_raw = assigned_partitions_flat[i + 1]
                    # 处理分区列表，每个分区名也需要解码
                    if isinstance(partitions_raw, list):
                        partitions = [
                            self._safe_decode_redis_value(p) for p in partitions_raw
                        ]
                    else:
                        partitions = [self._safe_decode_redis_value(partitions_raw)]
                    assigned_partitions[owner_id] = partitions
        return assigned_partitions

    def _safe_decode_redis_value(self, value: Any) -> str:
        """
        安全解码Redis返回值，兼容bytes和str类型

        当Redis客户端使用decode_responses=False时，返回值为bytes类型
        当Redis客户端使用decode_responses=True时，返回值为str类型

        Args:
            value: Redis返回的值，可能是bytes或str

        Returns:
            str: 解码后的字符串
        """
        if isinstance(value, bytes):
            return value.decode('utf-8')
        elif isinstance(value, str):
            return value
        else:
            return str(value)

    async def _check_and_keepalive_if_needed(self, owner_id: str) -> bool:
        """
        检查并按需执行keepalive

        检查owner上次keepalive时间，如果不存在记录或超过30秒，则触发一次keepalive。

        Args:
            owner_id: 消费者ID

        Returns:
            bool: 是否执行了keepalive操作
        """
        current_time_ms = int(time.time() * 1000)
        last_keepalive_time = self.owner_last_keepalive_time.get(owner_id, 0)

        # 如果不存在记录或者超过30秒，触发keepalive
        if (
            last_keepalive_time == 0 or (current_time_ms - last_keepalive_time) > 30000
        ):  # 30秒 = 30000毫秒
            logger.debug(
                "💓 RedisGroupQueueManager[%s] 按需触发keepalive: owner_id=%s, 距离上次=%.1f秒",
                self.key_prefix,
                owner_id,
                (current_time_ms - last_keepalive_time) / 1000.0,
            )
            # 触发keepalive并更新时间戳
            try:
                success = await self.keepalive_consumer(owner_id)
                if success:
                    self.owner_last_keepalive_time[owner_id] = current_time_ms
                    return True
                else:
                    logger.warning(
                        "⚠️ RedisGroupQueueManager[%s] 按需keepalive失败: owner_id=%s, keepalive_consumer返回False",
                        self.key_prefix,
                        owner_id,
                    )
                    return False
            except (redis.RedisError, ValueError, TypeError) as e:
                logger.warning(
                    "⚠️ RedisGroupQueueManager[%s] 按需keepalive异常: owner_id=%s, 错误=%s",
                    self.key_prefix,
                    owner_id,
                    e,
                )
                return False
        else:
            logger.debug(
                "💓 RedisGroupQueueManager[%s] 无需keepalive: owner_id=%s, 距离上次=%.1f秒",
                self.key_prefix,
                owner_id,
                (current_time_ms - last_keepalive_time) / 1000.0,
            )
            return False

    async def _ensure_scripts_loaded(self):
        """确保Lua脚本已加载"""
        if self._enqueue_script is None:
            self._enqueue_script = self.redis_client.register_script(ENQUEUE_SCRIPT)
            self._get_stats_script = self.redis_client.register_script(
                GET_QUEUE_STATS_SCRIPT
            )
            self._get_all_partitions_stats_script = self.redis_client.register_script(
                GET_ALL_PARTITIONS_STATS_SCRIPT
            )
            self._rebalance_partitions_script = self.redis_client.register_script(
                REBALANCE_PARTITIONS_SCRIPT
            )
            self._join_consumer_script = self.redis_client.register_script(
                JOIN_CONSUMER_SCRIPT
            )
            self._exit_consumer_script = self.redis_client.register_script(
                EXIT_CONSUMER_SCRIPT
            )
            self._keepalive_consumer_script = self.redis_client.register_script(
                KEEPALIVE_CONSUMER_SCRIPT
            )
            self._cleanup_inactive_owners_script = self.redis_client.register_script(
                CLEANUP_INACTIVE_OWNERS_SCRIPT
            )
            self._force_cleanup_script = self.redis_client.register_script(
                FORCE_CLEANUP_SCRIPT
            )
            self._get_messages_script = self.redis_client.register_script(
                GET_MESSAGES_SCRIPT
            )

    @rate_limit(max_rate=200, time_period=1)
    async def deliver_message(
        self,
        group_key: str,
        item: RedisGroupQueueItem,
        return_mode: str = "normal",
        max_total_messages: int = None,
    ) -> bool:
        """
        投递消息到指定分组队列

        Args:
            group_key: 分组键，通过hash路由到固定分区
            item: 消息数据项，必须实现RedisGroupQueueItem接口
            return_mode: 返回模式，normal只返回bool，reject_reason也返回拒绝原因
        Returns:
            bool: 投递是否成功
        """
        try:
            await self._ensure_scripts_loaded()

            # 通过hash路由到固定分区
            partition = self._hash_group_key_to_partition(group_key)

            # 生成排序分数
            sort_score = self.sort_key_func(item)

            # 根据序列化模式序列化消息
            if self.serialization_mode == SerializationMode.BSON:
                message_data = item.to_bson_bytes()
            else:  # JSON模式
                message_data = item.to_json_str()

            # 获取队列键
            queue_key = self._get_queue_key(partition)

            # 执行Lua脚本投递消息
            result = await self._enqueue_script(
                keys=[queue_key, self.counter_key],
                args=[
                    message_data,
                    sort_score,
                    self.queue_expire_seconds,
                    self.activity_expire_seconds,
                    (
                        max_total_messages
                        if max_total_messages is not None
                        else self.max_total_messages
                    ),
                ],
            )

            success, new_count, message = result

            # 安全解码消息内容，兼容bytes和str类型
            message_str = self._safe_decode_redis_value(message)

            if success:
                # 更新统计信息
                async with self._stats_lock:
                    self._manager_stats.total_delivered_messages += 1
                    self._manager_stats.total_current_messages = new_count

                logger.debug(
                    "✅ RedisGroupQueueManager[%s] 消息投递成功: group_key=%s->partition=%s, score=%.3f, 总留存=%d",
                    self.key_prefix,
                    group_key,
                    partition,
                    sort_score,
                    new_count,
                )
                if return_mode == "normal":
                    return True
                else:
                    return True, message_str
            else:
                # 投递失败
                async with self._stats_lock:
                    self._manager_stats.total_rejected_messages += 1

                logger.warning(
                    "❌ RedisGroupQueueManager[%s] 投递被拒绝: group_key=%s->partition=%s, 原因=%s",
                    self.key_prefix,
                    group_key,
                    partition,
                    message_str,
                )
                if return_mode == "normal":
                    return False
                else:
                    return False, message_str

        except (redis.RedisError, ValueError, TypeError) as e:
            # 注意：这里partition可能未定义，需要安全处理
            try:
                partition = self._hash_group_key_to_partition(group_key)
                logger.error(
                    "❌ RedisGroupQueueManager[%s] 投递消息失败: group_key=%s->partition=%s, 错误=%s",
                    self.key_prefix,
                    group_key,
                    partition,
                    e,
                )
            except (redis.RedisError, ValueError, TypeError):
                logger.error(
                    "❌ RedisGroupQueueManager[%s] 投递消息失败: group_key=%s, 错误=%s",
                    self.key_prefix,
                    group_key,
                    e,
                )
            if return_mode == "normal":
                return False
            else:
                return False, "投递报错"

    @rate_limit(
        max_rate=4, time_period=1, key_func=lambda owner_id: f"get_messages_{owner_id}"
    )
    async def get_messages(
        self,
        score_threshold: int,
        current_score: Optional[int] = None,
        owner_id: Optional[str] = None,
        _retry_depth: int = 2,
    ) -> List[RedisGroupQueueItem]:
        """
        获取消息

        遍历所有分配给该owner的分区，每个分区尝试获取1个消息。
        按需keepalive机制：检查上次keepalive时间，超过30秒则触发一次keepalive。

        Args:
            score_threshold: score差值阈值（毫秒），必传参数
            current_score: 当前score，用于空队列时的threshold比较，可选参数
            owner_id: 消费者ID，默认使用self.owner_id
            _retry_depth: 内部参数，递归重试深度限制，防止无限循环

        Returns:
            List[RedisGroupQueueItem]: 消息列表
        """
        try:
            await self._ensure_scripts_loaded()

            if owner_id is None:
                owner_id = self.owner_id

            # 按需keepalive机制
            await self._check_and_keepalive_if_needed(owner_id)

            # 执行获取消息脚本
            result = await self._get_messages_script(
                keys=[
                    self.owner_activate_time_zset_key,
                    self.queue_list_prefix,
                    self.queue_prefix,
                    self.counter_key,
                ],
                args=[
                    owner_id,
                    self.owner_expire_seconds,
                    score_threshold,
                    (
                        current_score
                        if current_score is not None
                        else self._default_sort_key(None)
                    ),
                ],
            )

            status, messages_data = result

            # 安全解码状态值，兼容bytes和str类型
            status_str = self._safe_decode_redis_value(status)

            if status_str == "JOIN_REQUIRED":
                # 检查递归深度，防止无限循环
                if _retry_depth <= 0:
                    logger.error(
                        "❌ RedisGroupQueueManager[%s] JOIN_REQUIRED重试次数耗尽: owner_id=%s",
                        self.key_prefix,
                        owner_id,
                    )
                    raise RuntimeError(
                        f"JOIN_REQUIRED重试次数耗尽: owner_id={owner_id}"
                    )

                logger.info(
                    "🔄 RedisGroupQueueManager[%s] 需要加入消费者: owner_id=%s, 剩余重试次数=%d",
                    self.key_prefix,
                    owner_id,
                    _retry_depth - 1,
                )
                # 自动加入消费者
                await self.join_consumer(owner_id)
                # 重新获取消息，递减重试深度
                return await self.get_messages(
                    score_threshold, current_score, owner_id, _retry_depth - 1
                )

            if status_str == "NO_QUEUES":
                logger.warning(
                    "📭 RedisGroupQueueManager[%s] 消费者无分配队列: owner_id=%s",
                    self.key_prefix,
                    owner_id,
                )
                return []

            # 解析消息数据
            messages = []
            for message_data in messages_data:
                try:
                    # 根据序列化模式反序列化消息
                    if self.serialization_mode == SerializationMode.BSON:
                        # BSON 字节数据
                        item = self.item_class.from_bson_bytes(message_data)
                    else:
                        # JSON 字符串
                        item = self.item_class.from_json_str(message_data)
                    messages.append(item)
                except (redis.RedisError, ValueError, TypeError) as e:
                    logger.warning(
                        "⚠️ RedisGroupQueueManager[%s] 消息反序列化失败: %s",
                        self.key_prefix,
                        e,
                    )

            if messages:
                # 更新统计信息
                async with self._stats_lock:
                    self._manager_stats.total_consumed_messages += len(messages)

                logger.debug(
                    "📤 RedisGroupQueueManager[%s] 获取消息成功: owner_id=%s, 获取数量=%d",
                    self.key_prefix,
                    owner_id,
                    len(messages),
                )
            else:
                logger.debug(
                    "📭 RedisGroupQueueManager[%s] 无可消费消息: owner_id=%s",
                    self.key_prefix,
                    owner_id,
                )

            return messages

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] 获取消息失败: owner_id=%s, 错误=%s",
                self.key_prefix,
                owner_id,
                e,
            )
            return []

    # ==================== 新的动态owner管理方法 ====================

    @rate_limit(max_rate=1, time_period=1, key_func=lambda: "rebalance_partitions")
    async def rebalance_partitions(self) -> Tuple[int, Dict[str, List[str]]]:
        """
        Rebalance重新分区

        基于owner_activate_time_zset清理掉所有的owner的queue_list，
        平均分配一下分区，给每个owner一个新的queue_list。

        Returns:
            Tuple[int, Dict[str, List[str]]]: (owner数量, 分配结果字典)
        """
        try:
            await self._ensure_scripts_loaded()

            # 执行rebalance脚本
            result = await self._rebalance_partitions_script(
                keys=[self.owner_activate_time_zset_key, self.queue_list_prefix],
                args=[self.FIXED_PARTITION_COUNT, self.owner_expire_seconds],
            )

            # 解析返回结果
            success, (owner_count, assigned_partitions) = self._parse_rebalance_result(
                result, 2
            )
            if not success:
                return 0, {}

            logger.info(
                "🔄 RedisGroupQueueManager[%s] Rebalance分区完成: owner数量=%d, 分区分配=%s",
                self.key_prefix,
                owner_count,
                assigned_partitions,
            )

            return owner_count, assigned_partitions

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Rebalance分区失败: 错误=%s",
                self.key_prefix,
                e,
            )
            return 0, {}

    @rate_limit(
        max_rate=1, time_period=1, key_func=lambda owner_id: f"join_consumer_{owner_id}"
    )
    async def join_consumer(
        self, owner_id: Optional[str] = None
    ) -> Tuple[int, Dict[str, List[str]]]:
        """
        加入消费者

        加入owner_activate_time_zset，然后rebalance重新分区。

        Args:
            owner_id: 消费者ID，默认使用self.owner_id

        Returns:
            Tuple[int, Dict[str, List[str]]]: (owner数量, 分配结果字典)
        """
        try:
            await self._ensure_scripts_loaded()

            if owner_id is None:
                owner_id = self.owner_id

            current_time = int(time.time() * 1000)  # 毫秒时间戳

            # 执行加入消费者脚本
            result = await self._join_consumer_script(
                keys=[self.owner_activate_time_zset_key, self.queue_list_prefix],
                args=[
                    owner_id,
                    current_time,
                    self.owner_expire_seconds,
                    self.FIXED_PARTITION_COUNT,
                ],
            )

            # 解析返回结果
            success, (owner_count, assigned_partitions) = self._parse_rebalance_result(
                result, 2
            )
            if not success:
                return 0, {}

            # 初始化owner的keepalive时间戳
            current_time_ms = int(time.time() * 1000)
            self.owner_last_keepalive_time[owner_id] = current_time_ms

            logger.info(
                "✅ RedisGroupQueueManager[%s] 消费者加入成功: owner_id=%s, owner数量=%d, 分配结果=%s",
                self.key_prefix,
                owner_id,
                owner_count,
                assigned_partitions,
            )

            return owner_count, assigned_partitions

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] 消费者加入失败: owner_id=%s, 错误=%s",
                self.key_prefix,
                owner_id,
                e,
            )
            return 0, {}

    @rate_limit(
        max_rate=1, time_period=1, key_func=lambda owner_id: f"exit_consumer_{owner_id}"
    )
    async def exit_consumer(
        self, owner_id: Optional[str] = None
    ) -> Tuple[int, Dict[str, List[str]]]:
        """
        消费者退出

        从owner_activate_time_zset删除，然后rebalance重新分区。

        Args:
            owner_id: 消费者ID，默认使用self.owner_id

        Returns:
            Tuple[int, Dict[str, List[str]]]: (owner数量, 分配结果字典)
        """
        try:
            await self._ensure_scripts_loaded()

            if owner_id is None:
                owner_id = self.owner_id

            # 执行消费者退出脚本
            result = await self._exit_consumer_script(
                keys=[self.owner_activate_time_zset_key, self.queue_list_prefix],
                args=[owner_id, self.owner_expire_seconds, self.FIXED_PARTITION_COUNT],
            )

            # 解析返回结果
            success, (owner_count, assigned_partitions) = self._parse_rebalance_result(
                result, 2
            )
            if not success:
                return 0, {}

            # 将退出的消费者从keepalive时间戳mapping中移除
            self.owner_last_keepalive_time.pop(owner_id, None)

            logger.info(
                "👋 RedisGroupQueueManager[%s] 消费者退出成功: owner_id=%s, 剩余owner数量=%d, 重新分配结果=%s",
                self.key_prefix,
                owner_id,
                owner_count,
                assigned_partitions,
            )

            return owner_count, assigned_partitions

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] 消费者退出失败: owner_id=%s, 错误=%s",
                self.key_prefix,
                owner_id,
                e,
            )
            return 0, {}

    @rate_limit(
        max_rate=1,
        time_period=2,
        key_func=lambda owner_id: f"keepalive_consumer_{owner_id}",
    )
    async def keepalive_consumer(self, owner_id: Optional[str] = None) -> bool:
        """
        消费者保活

        消费者定时更新owner_activate_time_zset的时间。
        建议每30秒调用一次。

        Args:
            owner_id: 消费者ID

        Returns:
            bool: 保活是否成功
        """
        try:
            await self._ensure_scripts_loaded()

            current_time = int(time.time() * 1000)  # 毫秒时间戳

            # 执行消费者保活脚本
            result = await self._keepalive_consumer_script(
                keys=[self.owner_activate_time_zset_key, self.queue_list_prefix],
                args=[owner_id, current_time, self.owner_expire_seconds],
            )

            success = bool(result)

            if success:
                logger.debug(
                    "💓 RedisGroupQueueManager[%s] 消费者保活成功: owner_id=%s",
                    self.key_prefix,
                    owner_id,
                )
            else:
                logger.warning(
                    "⚠️ RedisGroupQueueManager[%s] 消费者保活失败，queue_list不存在: owner_id=%s",
                    self.key_prefix,
                    owner_id,
                )

            return success

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] 消费者保活失败: owner_id=%s, 错误=%s",
                self.key_prefix,
                owner_id,
                e,
            )
            return False

    @rate_limit(max_rate=1, time_period=5, key_func=lambda: "cleanup_inactive_owners")
    async def cleanup_inactive_owners(self) -> Tuple[int, int, Dict[str, List[str]]]:
        """
        定期清理和重置

        遍历清理掉所有已经不活跃的owner（比如说5分钟没有活跃），
        如果有不活跃的，rebalance重新分区。

        Returns:
            Tuple[int, int, Dict[str, List[str]]]: (清理的owner数量, 剩余owner数量, 重新分配结果)
        """
        try:
            await self._ensure_scripts_loaded()

            current_time = int(time.time() * 1000)  # 毫秒时间戳
            inactive_threshold = current_time - (
                self.inactive_threshold_seconds * 1000
            )  # 转换为毫秒

            # 执行清理不活跃owner脚本
            result = await self._cleanup_inactive_owners_script(
                keys=[
                    self.owner_activate_time_zset_key,
                    self.queue_list_prefix,
                    self.queue_prefix,
                    self.counter_key,
                ],
                args=[
                    inactive_threshold,
                    current_time,
                    self.owner_expire_seconds,
                    self.FIXED_PARTITION_COUNT,
                ],
            )

            # 解析返回结果
            success, (cleaned_count, owner_count, assigned_partitions) = (
                self._parse_rebalance_result(result, 3)
            )
            if not success:
                return 0, 0, {}

            if cleaned_count > 0:
                logger.info(
                    "🧹 RedisGroupQueueManager[%s] 清理不活跃owner完成: 清理数量=%d, 剩余owner数量=%d, 重新分配结果=%s",
                    self.key_prefix,
                    cleaned_count,
                    owner_count,
                    assigned_partitions,
                )
            else:
                logger.debug(
                    "🧹 RedisGroupQueueManager[%s] 清理不活跃owner完成: 无需清理",
                    self.key_prefix,
                )

            return cleaned_count, owner_count, assigned_partitions

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] 清理不活跃owner失败: 错误=%s",
                self.key_prefix,
                e,
            )
            return 0, 0, {}

    @rate_limit(max_rate=1, time_period=5, key_func=lambda: "force_cleanup_and_reset")
    async def force_cleanup_and_reset(self, purge_all: bool = False) -> int:
        """
        强制清理和重置

        - purge_all=False（默认）：清理 owner_activate_time_zset 与所有 owner 的 queue_list，
          不删除各分区队列，仅重算计数器。
        - purge_all=True：在上述基础上额外删除所有分区队列，并将计数器置0（危险：全量清库）。

        Returns:
            int: 当 purge_all=False 时返回清理的 owner 数量；当 purge_all=True 时返回删除的分区数量
        """
        try:
            await self._ensure_scripts_loaded()

            if purge_all:
                # 危险：清空所有分区队列 + owner + 重置计数器（通过统一脚本，purge_all='1'）
                purged_partitions = await self._force_cleanup_script(
                    keys=[
                        self.owner_activate_time_zset_key,
                        self.queue_list_prefix,
                        self.queue_prefix,
                        self.counter_key,
                    ],
                    args=[self.FIXED_PARTITION_COUNT, "1"],
                )

                # 重置本地统计
                async with self._stats_lock:
                    self._manager_stats.total_current_messages = 0
                    self._manager_stats.total_delivered_messages = 0
                    self._manager_stats.total_consumed_messages = 0
                    self._manager_stats.total_rejected_messages = 0

                logger.warning(
                    "💥 RedisGroupQueueManager[%s] 已清空所有队列与owner: 分区数量=%d",
                    self.key_prefix,
                    purged_partitions,
                )
                return int(purged_partitions or 0)
            else:
                # 仅重置owner及队列分配，不删除分区队列
                cleaned_count = await self._force_cleanup_script(
                    keys=[
                        self.owner_activate_time_zset_key,
                        self.queue_list_prefix,
                        self.queue_prefix,
                        self.counter_key,
                    ],
                    args=[self.FIXED_PARTITION_COUNT, "0"],
                )

                logger.warning(
                    "💥 RedisGroupQueueManager[%s] 强制清理和重置完成: 清理owner数量=%d",
                    self.key_prefix,
                    cleaned_count,
                )
                return cleaned_count

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] 强制清理和重置失败: 错误=%s",
                self.key_prefix,
                e,
            )
            return 0

    @rate_limit(max_rate=1, time_period=5, key_func=lambda: "get_stats")
    async def get_stats(
        self,
        group_key: Optional[str] = None,
        include_all_partitions: bool = False,
        include_partition_details: bool = False,
        include_consumer_info: bool = False,
    ) -> Dict[str, Any]:
        """
        获取统计信息（统一接口）

        Args:
            group_key: 分组键，如果提供则获取特定队列统计，否则获取管理器整体统计
            include_all_partitions: 是否包含所有分区的统计信息
            include_partition_details: 是否包含分区详细信息
            include_consumer_info: 是否包含消费者信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            await self._ensure_scripts_loaded()

            # 如果指定了group_key，返回特定队列统计
            if group_key is not None and not include_all_partitions:
                # 获取单个分区的统计信息
                partition = self._hash_group_key_to_partition(group_key)
                queue_key = self._get_queue_key(partition)

                result = await self._get_stats_script(
                    keys=[queue_key, self.counter_key], args=[]
                )

                queue_size, _total_count, min_score, max_score = result

                return {
                    "type": "queue_stats",
                    "queue_name": f"{group_key}->partition={partition}",
                    "current_size": queue_size,
                    "last_activity_time": time.time(),
                    "min_score": min_score,
                    "max_score": max_score,
                    "partition": partition,
                }

            # 获取所有分区的统计信息（管理器级别或全分区统计）
            result = await self._get_all_partitions_stats_script(
                keys=[self.queue_prefix, self.counter_key],
                args=[str(self.FIXED_PARTITION_COUNT)],
            )

            (
                total_count,
                total_messages_in_queues,
                global_min_score,
                global_max_score,
                partition_stats_raw,
            ) = result

            # 构建基础统计信息
            async with self._stats_lock:
                # 更新运行时间和统计信息
                self._manager_stats.uptime_seconds = time.time() - self._start_time
                self._manager_stats.total_current_messages = total_messages_in_queues
                self._manager_stats.total_queues = self.FIXED_PARTITION_COUNT

                stats = self._manager_stats.to_dict()

            # 添加实时统计信息
            stats.update(
                {
                    "type": (
                        "manager_stats" if group_key is None else "all_partitions_stats"
                    ),
                    "counter_total_count": total_count,
                    "actual_messages_in_queues": total_messages_in_queues,
                    "global_min_score": global_min_score,
                    "global_max_score": global_max_score,
                    "key_prefix": self.key_prefix,
                }
            )

            # 如果需要消费者信息
            if include_consumer_info:
                try:
                    active_owners_raw = await self.redis_client.zrange(
                        self.owner_activate_time_zset_key, 0, -1
                    )
                    # 安全解码owner列表
                    active_owners = [
                        self._safe_decode_redis_value(owner)
                        for owner in active_owners_raw
                    ]
                    stats["active_consumers_count"] = len(active_owners)
                    stats["active_consumers"] = active_owners

                    # 获取分区分配情况
                    partition_assignments = {}
                    for owner in active_owners:
                        queue_list_key = f"{self.queue_list_prefix}{owner}"
                        assigned_partitions_raw = await self.redis_client.lrange(
                            queue_list_key, 0, -1
                        )
                        # 安全解码分区列表
                        assigned_partitions = [
                            self._safe_decode_redis_value(p)
                            for p in assigned_partitions_raw
                        ]
                        partition_assignments[owner] = assigned_partitions
                    stats["partition_assignments"] = partition_assignments

                except (redis.RedisError, ValueError, TypeError) as e:
                    logger.warning("获取消费者信息失败: %s", e)
                    stats["active_consumers_count"] = 0
                    stats["active_consumers"] = []
                    stats["partition_assignments"] = {}

            # 如果需要分区详细信息
            if include_partition_details:
                partitions = []
                non_empty_partitions = 0
                max_partition_size = 0
                min_partition_size = float('inf')

                for i in range(0, len(partition_stats_raw), 4):
                    if i + 3 < len(partition_stats_raw):
                        partition_size = partition_stats_raw[i + 1]
                        partitions.append(
                            {
                                "partition": self._safe_decode_redis_value(
                                    partition_stats_raw[i]
                                ),
                                "current_size": partition_size,
                                "min_score": partition_stats_raw[i + 2],
                                "max_score": partition_stats_raw[i + 3],
                            }
                        )

                        if partition_size > 0:
                            non_empty_partitions += 1
                            max_partition_size = max(max_partition_size, partition_size)
                            min_partition_size = min(min_partition_size, partition_size)

                stats["partitions"] = partitions
                stats["non_empty_partitions"] = non_empty_partitions
                stats["max_partition_size"] = (
                    max_partition_size if max_partition_size != 0 else 0
                )
                stats["min_partition_size"] = (
                    min_partition_size if min_partition_size != float('inf') else 0
                )

            return stats

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error("获取统计信息失败: group_key=%s, 错误=%s", group_key, e)

            # 降级处理：返回基础统计信息
            try:
                current_count = await self.redis_client.get(self.counter_key)
                total_current_messages = int(current_count) if current_count else 0
            except (redis.RedisError, ValueError, TypeError):
                total_current_messages = 0

            return {
                "type": "error_fallback",
                "total_current_messages": total_current_messages,
                "total_queues": self.FIXED_PARTITION_COUNT,
                "error": str(e),
            }

    @rate_limit(
        max_rate=1,
        time_period=5,
        key_func=lambda group_key: f"get_queue_stats_{group_key}",
    )
    async def get_queue_stats(self, group_key: str) -> Optional[Dict[str, Any]]:
        """兼容性方法：获取队列统计信息"""
        result = await self.get_stats(group_key=group_key)
        return result if result.get("type") != "error_fallback" else None

    @rate_limit(max_rate=1, time_period=5, key_func=lambda: "get_manager_stats")
    async def get_manager_stats(self) -> Dict[str, Any]:
        """兼容性方法：获取管理器统计信息"""
        return await self.get_stats()

    async def start(self):
        """
        启动管理器（开启周期任务）

        只能启动一次，shutdown后不能再启动

        Raises:
            RuntimeError: 如果管理器已经启动或已关闭
        """
        if self._state == ManagerState.STARTED:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] 已经启动，忽略重复启动请求",
                self.key_prefix,
            )
            return

        if self._state == ManagerState.SHUTDOWN:
            raise RuntimeError(
                f"RedisGroupQueueManager[{self.key_prefix}] 已关闭，不能重新启动"
            )

        # 状态必须是 CREATED
        if self._state != ManagerState.CREATED:
            raise RuntimeError(
                f"RedisGroupQueueManager[{self.key_prefix}] 状态异常: {self._state}"
            )

        logger.info("🚀 RedisGroupQueueManager[%s] 开始启动...", self.key_prefix)

        await self.start_periodic_tasks()

        # 更新状态为已启动
        self._state = ManagerState.STARTED

        logger.info("✅ RedisGroupQueueManager[%s] 启动完成", self.key_prefix)

    async def start_periodic_tasks(self):
        """启动定期任务"""
        if self._running:
            return

        self._running = True

        # 启动时立即执行一次清理
        try:
            await self.cleanup_inactive_owners()
            logger.info("🧹 RedisGroupQueueManager[%s] 启动时清理完成", self.key_prefix)
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] 启动时清理失败: %s", self.key_prefix, e
            )

        # 启动时立即执行一次log
        try:
            await self._log_manager_details()
            logger.info(
                "🔥 RedisGroupQueueManager[%s] 启动时日志打印完成", self.key_prefix
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] 启动时日志打印失败: %s",
                self.key_prefix,
                e,
            )

        # 启动定期任务
        self._log_task = asyncio.create_task(self._periodic_log_worker())
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup_worker())

        logger.info("📊 RedisGroupQueueManager[%s] 定期任务已启动", self.key_prefix)

    async def stop_periodic_tasks(self):
        """停止定期任务"""
        if not self._running:
            return

        self._running = False

        # 停止日志任务
        if self._log_task and not self._log_task.done():
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass

        # 停止cleanup任务
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("📊 RedisGroupQueueManager[%s] 定期任务已停止", self.key_prefix)

    async def _periodic_log_worker(self):
        """定期日志打印工作协程"""
        try:
            while self._running:
                await asyncio.sleep(self.log_interval_seconds)
                if self._running:
                    await self._log_manager_details()
        except asyncio.CancelledError:
            logger.debug(
                "📊 RedisGroupQueueManager[%s] 定期日志任务被取消", self.key_prefix
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "📊 RedisGroupQueueManager[%s] 定期日志任务异常: %s", self.key_prefix, e
            )

    async def _periodic_cleanup_worker(self):
        """定期清理工作协程"""
        try:
            while self._running:
                # 添加相对抖动，避免所有实例同时清理，并确保非负
                jitter = self.cleanup_interval_seconds * 0.3
                delay = self.cleanup_interval_seconds + random.uniform(-jitter, jitter)
                await asyncio.sleep(max(1.0, delay))
                if self._running:
                    await self.cleanup_inactive_owners()
        except asyncio.CancelledError:
            logger.debug(
                "🧹 RedisGroupQueueManager[%s] 定期清理任务被取消", self.key_prefix
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "🧹 RedisGroupQueueManager[%s] 定期清理任务异常: %s", self.key_prefix, e
            )

    async def _log_manager_details(self):
        """打印管理器详细信息"""
        try:
            manager_stats = await self.get_manager_stats()

            # 打印管理器整体状态
            logger.info(
                "📊 RedisGroupQueueManager[%s] 整体状态: "
                "活跃队列=%d, 总消息=%d, 总投递=%d, 总消费=%d, 总拒绝=%d, 运行时间=%.1f秒",
                self.key_prefix,
                manager_stats["total_queues"],
                manager_stats["total_current_messages"],
                manager_stats["total_delivered_messages"],
                manager_stats["total_consumed_messages"],
                manager_stats["total_rejected_messages"],
                manager_stats["uptime_seconds"],
            )

            # 统一一次性打印所有分区的详细信息
            partitions = self.partition_names
            details_lines = []
            for partition in partitions:
                try:
                    queue_key = self._get_queue_key(partition)
                    queue_size = await self.redis_client.zcard(queue_key)
                    if queue_size > 0:
                        # 获取最小和最大分数
                        min_result = await self.redis_client.zrange(
                            queue_key, 0, 0, withscores=True
                        )
                        max_result = await self.redis_client.zrange(
                            queue_key, -1, -1, withscores=True
                        )
                        min_score = min_result[0][1] if min_result else 0
                        max_score = max_result[0][1] if max_result else 0
                        details_lines.append(
                            f"   分区[{partition}]: 大小={queue_size}, 分数范围=[{min_score:.3f}, {max_score:.3f}]"
                        )
                    else:
                        details_lines.append(f"   分区[{partition}]: 大小=0")
                except (redis.RedisError, ValueError, TypeError) as e:
                    details_lines.append(f"   分区[{partition}]: 获取状态失败: {e}")

            if details_lines:
                logger.info(
                    "🔥 分区状态汇总: 共%d个分区\n%s",
                    len(partitions),
                    "\n".join(details_lines),
                )

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "📊 RedisGroupQueueManager[%s] 打印详情失败: %s", self.key_prefix, e
            )

    async def shutdown(self, mode: ShutdownMode = ShutdownMode.HARD) -> bool:
        """
        关闭管理器

        Args:
            mode: 关闭模式

        Returns:
            bool: 是否成功关闭
        """
        if self._state == ManagerState.SHUTDOWN:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] 已经关闭，忽略重复关闭请求",
                self.key_prefix,
            )
            return True

        if self._state == ManagerState.CREATED:
            logger.info(
                "ℹ️ RedisGroupQueueManager[%s] 未启动状态下关闭", self.key_prefix
            )
            self._state = ManagerState.SHUTDOWN
            return True

        # 状态必须是 STARTED
        if self._state != ManagerState.STARTED:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] 状态异常，强制关闭: %s",
                self.key_prefix,
                self._state,
            )

        logger.info("🔌 RedisGroupQueueManager[%s] 开始关闭...", self.key_prefix)

        # 停止定期任务
        await self.stop_periodic_tasks()

        if mode == ShutdownMode.SOFT:
            # 软性关闭：检查是否有消息
            stats = await self.get_manager_stats()
            remaining_messages = stats.get("total_current_messages", 0)
            if remaining_messages > 0:
                logger.warning(
                    "⚠️ RedisGroupQueueManager[%s] 软性关闭检测到剩余消息: %d条",
                    self.key_prefix,
                    remaining_messages,
                )
                # 软性关闭失败，但不改变状态，允许重试
                return False

        # 关闭前最后一次打印详细信息
        try:
            await self._log_manager_details()
            logger.info(
                "🔥 RedisGroupQueueManager[%s] 关闭前最终状态日志完成", self.key_prefix
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] 关闭前日志打印失败: %s",
                self.key_prefix,
                e,
            )

        # 更新状态为已关闭
        self._state = ManagerState.SHUTDOWN

        logger.info("🔌 RedisGroupQueueManager[%s] 已关闭", self.key_prefix)
        return True

    def get_state(self) -> ManagerState:
        """
        获取管理器当前状态

        Returns:
            ManagerState: 当前状态
        """
        return self._state

    def __repr__(self) -> str:
        return (
            f"RedisGroupQueueManager(key_prefix={self.key_prefix}, "
            f"max_total_messages={self.max_total_messages})"
        )
