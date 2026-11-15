"""
消息分组队列管理器

提供基于哈希路由的固定数量队列管理功能，解决kafka消息处理的阻塞问题。
支持消息投递、消费、统计和监控等功能。
"""

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format

logger = get_logger(__name__)


class ShutdownMode(Enum):
    """关闭模式枚举"""

    SOFT = "soft"  # 软性关闭：检查是否有消息，有延迟时间控制
    HARD = "hard"  # 硬性关闭：直接关闭，记录未处理消息数


@dataclass
class ShutdownState:
    """关闭状态"""

    is_shutting_down: bool = False
    first_soft_shutdown_time: Optional[float] = None
    max_delay_seconds: Optional[float] = None

    def reset(self):
        """重置关闭状态"""
        self.is_shutting_down = False
        self.first_soft_shutdown_time = None
        self.max_delay_seconds = None


@dataclass
class TimeWindowStats:
    """时间窗口统计"""

    delivered_1min: int = 0
    consumed_1min: int = 0
    delivered_1hour: int = 0
    consumed_1hour: int = 0


@dataclass
class QueueStats:
    """队列统计信息"""

    queue_id: int
    current_size: int
    total_delivered: int = 0
    total_consumed: int = 0
    last_deliver_time: Optional[str] = None
    last_consume_time: Optional[str] = None
    # 时间窗口统计
    time_window_stats: TimeWindowStats = field(default_factory=TimeWindowStats)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "queue_id": self.queue_id,
            "current_size": self.current_size,
            "total_delivered": self.total_delivered,
            "total_consumed": self.total_consumed,
            "last_deliver_time": self.last_deliver_time,
            "last_consume_time": self.last_consume_time,
            "delivered_1min": self.time_window_stats.delivered_1min,
            "consumed_1min": self.time_window_stats.consumed_1min,
            "delivered_1hour": self.time_window_stats.delivered_1hour,
            "consumed_1hour": self.time_window_stats.consumed_1hour,
        }


@dataclass
class ManagerStats:
    """管理器整体统计信息"""

    total_queues: int
    total_current_messages: int
    total_delivered_messages: int = 0
    total_consumed_messages: int = 0
    total_rejected_messages: int = 0
    start_time: str = field(
        default_factory=lambda: to_iso_format(get_now_with_timezone())
    )
    uptime_seconds: float = 0
    # 时间窗口统计
    time_window_stats: TimeWindowStats = field(default_factory=TimeWindowStats)

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
            "delivered_1min": self.time_window_stats.delivered_1min,
            "consumed_1min": self.time_window_stats.consumed_1min,
            "delivered_1hour": self.time_window_stats.delivered_1hour,
            "consumed_1hour": self.time_window_stats.consumed_1hour,
        }


class MsgGroupQueueManager:
    """
    消息分组队列管理器

    特性：
    1. 固定数量的队列（默认10个，可配置）
    2. 基于group_key哈希路由到固定分组
    3. 支持最大消息数量限制（默认100个）
    4. 空队列优先投递策略
    5. 支持wait/no-wait模式的消息获取
    6. 提供详细的metrics和日志
    """

    def __init__(
        self,
        name: str = "default",
        num_queues: int = 10,
        max_total_messages: int = 100,
        enable_metrics: bool = True,
        log_interval_seconds: int = 30,
    ):
        """
        初始化消息分组队列管理器

        Args:
            name: 管理器名称
            num_queues: 队列数量
            max_total_messages: 最大总消息数量限制
            enable_metrics: 是否启用统计功能
            log_interval_seconds: 日志打印间隔（秒）
        """
        self.name = name
        self.num_queues = num_queues
        self.max_total_messages = max_total_messages
        self.enable_metrics = enable_metrics
        self.log_interval_seconds = log_interval_seconds

        # 初始化队列 - 使用asyncio.Queue
        self._queues: List[asyncio.Queue] = [asyncio.Queue() for _ in range(num_queues)]

        # 队列统计信息
        self._queue_stats: List[QueueStats] = [
            QueueStats(queue_id=i, current_size=0) for i in range(num_queues)
        ]

        # 管理器统计信息
        self._manager_stats = ManagerStats(
            total_queues=num_queues, total_current_messages=0
        )

        # 异步锁，保护统计信息
        self._stats_lock = asyncio.Lock()

        # 启动时间
        self._start_time = time.time()

        # 定期日志任务
        self._log_task: Optional[asyncio.Task] = None
        self._running = False

        # 关闭状态
        self._shutdown_state = ShutdownState()

        # 时间窗口事件追踪 - 使用deque存储带时间戳的事件
        self._delivery_events: List[deque] = [
            deque() for _ in range(num_queues)
        ]  # 每个队列的投递事件
        self._consume_events: List[deque] = [
            deque() for _ in range(num_queues)
        ]  # 每个队列的消费事件
        self._manager_delivery_events = deque()  # 管理器总投递事件
        self._manager_consume_events = deque()  # 管理器总消费事件

        logger.info(
            "🚀 MsgGroupQueueManager[%s] 初始化完成: num_queues=%d, max_total_messages=%d",
            self.name,
            self.num_queues,
            self.max_total_messages,
        )

    def _hash_route(self, group_key: str) -> int:
        """
        基于group_key计算哈希路由到队列编号

        Args:
            group_key: 分组键

        Returns:
            int: 队列编号 (0 到 num_queues-1)
        """
        # 使用MD5哈希确保分布均匀
        hash_obj = hashlib.md5(group_key.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int % self.num_queues

    async def deliver_message(self, group_key: str, message_data: Any) -> bool:
        """
        投递消息到指定分组队列

        Args:
            group_key: 分组键，用于哈希路由
            message_data: 消息数据

        Returns:
            bool: 投递是否成功
        """
        try:
            # 计算目标队列
            target_queue_id = self._hash_route(group_key)
            target_queue = self._queues[target_queue_id]

            # 检查投递条件
            can_deliver, reject_reason = self._can_deliver_message()
            if not can_deliver:
                # 拒绝投递
                async with self._stats_lock:
                    self._manager_stats.total_rejected_messages += 1

                logger.warning(
                    "❌ MsgGroupQueueManager[%s] 投递被拒绝: group_key=%s, 原因=%s",
                    self.name,
                    group_key,
                    reject_reason,
                )
                return False

            # 执行投递
            message_tuple = (group_key, message_data)
            await target_queue.put(message_tuple)

            # 更新统计信息
            current_time = to_iso_format(get_now_with_timezone())
            timestamp = time.time()

            async with self._stats_lock:
                self._queue_stats[target_queue_id].current_size = target_queue.qsize()
                self._queue_stats[target_queue_id].total_delivered += 1
                self._queue_stats[target_queue_id].last_deliver_time = current_time

                self._manager_stats.total_delivered_messages += 1
                self._manager_stats.total_current_messages = (
                    self._get_total_current_messages()
                )

                # 记录时间窗口事件
                self._delivery_events[target_queue_id].append(timestamp)
                self._manager_delivery_events.append(timestamp)

            logger.debug(
                "✅ MsgGroupQueueManager[%s] 消息投递成功: group_key=%s -> queue_id=%d, 队列当前大小=%d, 总留存=%d",
                self.name,
                group_key,
                target_queue_id,
                target_queue.qsize(),
                self._get_total_current_messages(),
            )

            return True

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "❌ MsgGroupQueueManager[%s] 投递消息失败: group_key=%s, 错误=%s",
                self.name,
                group_key,
                e,
            )
            return False

    async def get_by_queue(
        self, queue_id: int, wait: bool = True, timeout: Optional[float] = None
    ) -> Optional[Tuple[str, Any]]:
        """
        从指定队列获取消息

        Args:
            queue_id: 队列编号
            wait: 是否等待消息（True=阻塞等待，False=立即返回）
            timeout: 等待超时时间（秒），仅在wait=True时有效

        Returns:
            Optional[Tuple[str, Any]]: 消息元组 (group_key, message_data)，None表示没有消息
        """
        if queue_id < 0 or queue_id >= self.num_queues:
            raise ValueError(
                f"队列编号超出范围: {queue_id}, 有效范围: 0-{self.num_queues-1}"
            )

        target_queue = self._queues[queue_id]

        try:
            if wait:
                # 阻塞等待模式
                if timeout is not None:
                    message_tuple = await asyncio.wait_for(
                        target_queue.get(), timeout=timeout
                    )
                else:
                    message_tuple = await target_queue.get()
            else:
                # 立即返回模式
                try:
                    message_tuple = target_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return None

            # 更新统计信息
            current_time = to_iso_format(get_now_with_timezone())
            timestamp = time.time()

            async with self._stats_lock:
                self._queue_stats[queue_id].current_size = target_queue.qsize()
                self._queue_stats[queue_id].total_consumed += 1
                self._queue_stats[queue_id].last_consume_time = current_time

                self._manager_stats.total_consumed_messages += 1
                self._manager_stats.total_current_messages = (
                    self._get_total_current_messages()
                )

                # 记录时间窗口事件
                self._consume_events[queue_id].append(timestamp)
                self._manager_consume_events.append(timestamp)

            group_key, _ = message_tuple
            logger.debug(
                "📤 MsgGroupQueueManager[%s] 消息消费成功: queue_id=%d, group_key=%s, 队列剩余=%d",
                self.name,
                queue_id,
                group_key,
                target_queue.qsize(),
            )

            return message_tuple

        except asyncio.TimeoutError:
            logger.debug(
                "⏰ MsgGroupQueueManager[%s] 获取消息超时: queue_id=%d, timeout=%s",
                self.name,
                queue_id,
                timeout,
            )
            return None
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "❌ MsgGroupQueueManager[%s] 获取消息失败: queue_id=%d, 错误=%s",
                self.name,
                queue_id,
                e,
            )
            return None

    async def get_queue_info(
        self, queue_id: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        获取队列信息

        Args:
            queue_id: 队列编号，None表示获取所有队列信息

        Returns:
            Union[Dict, List[Dict]]: 队列信息字典或队列信息列表
        """
        async with self._stats_lock:
            # 更新当前队列大小
            for i, queue in enumerate(self._queues):
                self._queue_stats[i].current_size = queue.qsize()

            # 更新时间窗口统计
            self._update_time_window_stats()

            if queue_id is not None:
                if queue_id < 0 or queue_id >= self.num_queues:
                    raise ValueError(
                        f"队列编号超出范围: {queue_id}, 有效范围: 0-{self.num_queues-1}"
                    )
                return self._queue_stats[queue_id].to_dict()
            else:
                return [stat.to_dict() for stat in self._queue_stats]

    async def get_manager_stats(self) -> Dict[str, Any]:
        """
        获取管理器整体统计信息

        Returns:
            Dict[str, Any]: 管理器统计信息
        """
        async with self._stats_lock:
            # 更新运行时间和当前消息总数
            self._manager_stats.uptime_seconds = time.time() - self._start_time
            self._manager_stats.total_current_messages = (
                self._get_total_current_messages()
            )

            # 更新时间窗口统计
            self._update_time_window_stats()

            return self._manager_stats.to_dict()

    async def get_summary(self) -> Dict[str, Any]:
        """
        获取完整的汇总信息

        Returns:
            Dict[str, Any]: 包含管理器统计和队列详情的完整信息
        """
        return {
            "manager": await self.get_manager_stats(),
            "queues": await self.get_queue_info(),
        }

    def _get_total_current_messages(self) -> int:
        """获取当前总消息数量"""
        return sum(queue.qsize() for queue in self._queues)

    def _can_deliver_message(self) -> Tuple[bool, str]:
        """
        检查是否可以投递消息

        Returns:
            Tuple[bool, str]: (是否可以投递, 拒绝原因)
        """
        current_total = self._get_total_current_messages()

        # 如果有空队列，允许投递（不受总数限制）
        has_empty_queue = any(q.qsize() == 0 for q in self._queues)

        # 投递条件：总数未超限 或者 有空队列
        if current_total >= self.max_total_messages and not has_empty_queue:
            return (
                False,
                f"当前总消息数={current_total}, 限制={self.max_total_messages}, 无空队列",
            )

        return True, ""

    def _clean_old_events(self, events: deque, max_age_seconds: float):
        """清理超过指定时间的事件"""
        current_time = time.time()
        while events and current_time - events[0] > max_age_seconds:
            events.popleft()

    def _count_events_in_window(self, events: deque, window_seconds: float) -> int:
        """统计指定时间窗口内的事件数量"""
        # 先清理旧事件
        self._clean_old_events(events, window_seconds)
        # 返回剩余事件数量
        return len(events)

    def _update_time_window_stats(self):
        """更新所有队列和管理器的时间窗口统计"""
        # 更新每个队列的时间窗口统计
        for i in range(self.num_queues):
            # 清理旧事件并统计
            self._queue_stats[i].time_window_stats.delivered_1min = (
                self._count_events_in_window(self._delivery_events[i], 60.0)
            )
            self._queue_stats[i].time_window_stats.consumed_1min = (
                self._count_events_in_window(self._consume_events[i], 60.0)
            )
            self._queue_stats[i].time_window_stats.delivered_1hour = (
                self._count_events_in_window(self._delivery_events[i], 3600.0)
            )
            self._queue_stats[i].time_window_stats.consumed_1hour = (
                self._count_events_in_window(self._consume_events[i], 3600.0)
            )

        # 更新管理器的时间窗口统计
        self._manager_stats.time_window_stats.delivered_1min = (
            self._count_events_in_window(self._manager_delivery_events, 60.0)
        )
        self._manager_stats.time_window_stats.consumed_1min = (
            self._count_events_in_window(self._manager_consume_events, 60.0)
        )
        self._manager_stats.time_window_stats.delivered_1hour = (
            self._count_events_in_window(self._manager_delivery_events, 3600.0)
        )
        self._manager_stats.time_window_stats.consumed_1hour = (
            self._count_events_in_window(self._manager_consume_events, 3600.0)
        )

    async def start_periodic_logging(self):
        """启动定期日志打印任务"""
        if self._running:
            return

        self._running = True
        self._log_task = asyncio.create_task(self._periodic_log_worker())
        logger.info("📊 MsgGroupQueueManager[%s] 定期日志任务已启动", self.name)

    async def stop_periodic_logging(self):
        """停止定期日志打印任务"""
        if not self._running:
            return

        self._running = False
        if self._log_task and not self._log_task.done():
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass

        logger.info("📊 MsgGroupQueueManager[%s] 定期日志任务已停止", self.name)

    async def _periodic_log_worker(self):
        """定期日志打印工作协程"""
        try:
            while self._running:
                await asyncio.sleep(self.log_interval_seconds)
                if self._running:
                    await self._log_queue_details()
        except asyncio.CancelledError:
            logger.debug("📊 MsgGroupQueueManager[%s] 定期日志任务被取消", self.name)
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "📊 MsgGroupQueueManager[%s] 定期日志任务异常: %s", self.name, e
            )

    async def _log_queue_details(self):
        """打印队列详细信息"""
        try:
            manager_stats = await self.get_manager_stats()
            queue_infos = await self.get_queue_info()

            # 打印管理器整体状态汇总
            logger.info(
                "📊 MsgGroupQueueManager[%s] 整体状态: "
                "总消息=%d, 总投递=%d, 总消费=%d, 总拒绝=%d, 运行时间=%.1f秒",
                self.name,
                manager_stats["total_current_messages"],
                manager_stats["total_delivered_messages"],
                manager_stats["total_consumed_messages"],
                manager_stats["total_rejected_messages"],
                manager_stats["uptime_seconds"],
            )

            # 打印管理器时间窗口统计
            logger.info(
                "⏱️  MsgGroupQueueManager[%s] 时间窗口统计: "
                "1分钟内(投递=%d, 消费=%d), 1小时内(投递=%d, 消费=%d)",
                self.name,
                manager_stats["delivered_1min"],
                manager_stats["consumed_1min"],
                manager_stats["delivered_1hour"],
                manager_stats["consumed_1hour"],
            )

            # 分别打印每个队列的详细信息
            active_queues = []
            idle_queues = []
            empty_queues = []  # 当前为空的队列

            for queue_info in queue_infos:
                queue_id = queue_info["queue_id"]
                current_size = queue_info["current_size"]
                total_delivered = queue_info["total_delivered"]
                total_consumed = queue_info["total_consumed"]
                last_deliver_time = queue_info["last_deliver_time"]
                last_consume_time = queue_info["last_consume_time"]

                # 获取时间窗口统计
                delivered_1min = queue_info.get("delivered_1min", 0)
                consumed_1min = queue_info.get("consumed_1min", 0)
                delivered_1hour = queue_info.get("delivered_1hour", 0)
                consumed_1hour = queue_info.get("consumed_1hour", 0)

                # 记录空队列
                if current_size == 0:
                    empty_queues.append(queue_id)

                # 计算队列的活跃程度：基于时间窗口活动和当前队列状态
                has_recent_activity = delivered_1min > 0 or consumed_1min > 0
                has_messages = current_size > 0
                has_historical_activity = total_delivered > 0 or total_consumed > 0

                # 活跃判断：近期有活动 或 当前有消息 或 历史上有活动
                is_active = (
                    has_recent_activity or has_messages or has_historical_activity
                )

                if is_active:
                    # 计算投递消费比率
                    delivery_rate = (
                        total_delivered / max(1, total_delivered + total_consumed) * 100
                    )
                    consume_rate = (
                        total_consumed / max(1, total_delivered + total_consumed) * 100
                    )

                    # 格式化时间显示
                    last_deliver_display = (
                        last_deliver_time[-8:] if last_deliver_time else "无"
                    )
                    last_consume_display = (
                        last_consume_time[-8:] if last_consume_time else "无"
                    )

                    active_queues.append(
                        {
                            "id": queue_id,
                            "current": current_size,
                            "delivered": total_delivered,
                            "consumed": total_consumed,
                            "delivery_rate": delivery_rate,
                            "consume_rate": consume_rate,
                            "last_deliver": last_deliver_display,
                            "last_consume": last_consume_display,
                            "delivered_1min": delivered_1min,
                            "consumed_1min": consumed_1min,
                            "delivered_1hour": delivered_1hour,
                            "consumed_1hour": consumed_1hour,
                        }
                    )
                else:
                    idle_queues.append(queue_id)

            # 打印活跃队列详情
            if active_queues:
                logger.info("🔥 活跃队列详情 (%d个):", len(active_queues))
                for queue in active_queues:
                    # 队列状态标识
                    status_indicators = []
                    if queue["current"] == 0:
                        status_indicators.append("空")
                    elif queue["current"] > self.max_total_messages * 0.3:
                        status_indicators.append("积压")

                    if queue["delivered_1min"] > 0:
                        status_indicators.append("近期投递")
                    if queue["consumed_1min"] > 0:
                        status_indicators.append("近期消费")

                    status_text = (
                        f"[{', '.join(status_indicators)}]" if status_indicators else ""
                    )

                    logger.info(
                        "   队列[%d]%s: 当前=%d, 总投递=%d(%.1f%%), 总消费=%d(%.1f%%), "
                        "最后投递=%s, 最后消费=%s",
                        queue["id"],
                        status_text,
                        queue["current"],
                        queue["delivered"],
                        queue["delivery_rate"],
                        queue["consumed"],
                        queue["consume_rate"],
                        queue["last_deliver"],
                        queue["last_consume"],
                    )

                    # 打印时间窗口统计
                    logger.info(
                        "      ⏱️  1分钟内(投递=%d, 消费=%d), 1小时内(投递=%d, 消费=%d)",
                        queue["delivered_1min"],
                        queue["consumed_1min"],
                        queue["delivered_1hour"],
                        queue["consumed_1hour"],
                    )

            # 打印空闲队列信息
            if idle_queues:
                logger.info(
                    "💤 空闲队列: %s (共%d个)",
                    ", ".join([f"队列[{qid}]" for qid in idle_queues]),
                    len(idle_queues),
                )

            # 打印空队列信息（当前无消息的队列）
            if empty_queues:
                logger.info(
                    "📭 空队列: %s (共%d个，可接受新消息)",
                    ", ".join([f"队列[{qid}]" for qid in empty_queues]),
                    len(empty_queues),
                )

            # 打印队列负载分析
            if active_queues:
                # 找出最繁忙和最空闲的队列
                busiest_queue = max(
                    active_queues, key=lambda q: q["delivered"] + q["consumed"]
                )
                most_backlogged = max(active_queues, key=lambda q: q["current"])

                logger.info(
                    "📈 队列负载分析: 最繁忙=队列[%d](处理%d条), 最积压=队列[%d](积压%d条)",
                    busiest_queue["id"],
                    busiest_queue["delivered"] + busiest_queue["consumed"],
                    most_backlogged["id"],
                    most_backlogged["current"],
                )

            # 如果有队列积压，给出警告
            high_backlog_queues = [
                q for q in active_queues if q["current"] > self.max_total_messages * 0.3
            ]
            if high_backlog_queues:
                logger.warning(
                    "⚠️ 队列积压警告: %s",
                    ", ".join(
                        [
                            f"队列[{q['id']}]({q['current']}条)"
                            for q in high_backlog_queues
                        ]
                    ),
                )

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "📊 MsgGroupQueueManager[%s] 打印队列详情失败: %s", self.name, e
            )

    async def shutdown(
        self,
        mode: ShutdownMode = ShutdownMode.HARD,
        max_delay_seconds: Optional[float] = None,
    ) -> bool:
        """
        关闭管理器，支持硬性和软性关闭模式

        Args:
            mode: 关闭模式 (HARD: 硬性关闭, SOFT: 软性关闭)
            max_delay_seconds: 软性关闭的最大延迟时间（秒），仅在首次软性关闭时设置

        Returns:
            bool: True表示成功关闭，False表示仍有消息需要处理（仅软性关闭）
        """
        current_time = time.time()

        if mode == ShutdownMode.SOFT:
            # 软性关闭逻辑
            if not self._shutdown_state.is_shutting_down:
                # 首次软性关闭
                self._shutdown_state.is_shutting_down = True
                self._shutdown_state.first_soft_shutdown_time = current_time
                self._shutdown_state.max_delay_seconds = max_delay_seconds

                logger.info(
                    "🔄 MsgGroupQueueManager[%s] 开始软性关闭，最大延迟时间: %s 秒",
                    self.name,
                    max_delay_seconds,
                )

            # 检查是否有消息
            total_remaining = self._get_total_current_messages()

            if total_remaining == 0:
                # 没有消息，可以安全关闭
                await self._perform_hard_shutdown()
                self._shutdown_state.reset()
                return True

            # 检查是否超过延迟时间
            if (
                self._shutdown_state.max_delay_seconds is not None
                and current_time - self._shutdown_state.first_soft_shutdown_time
                >= self._shutdown_state.max_delay_seconds
            ):
                # 超过延迟时间，强制关闭
                logger.warning(
                    "⏰ MsgGroupQueueManager[%s] 软性关闭超时，强制关闭。剩余消息: %d 条",
                    self.name,
                    total_remaining,
                )
                await self._perform_hard_shutdown()
                self._shutdown_state.reset()
                return True

            # 仍有消息且未超时，返回False
            elapsed_time = current_time - self._shutdown_state.first_soft_shutdown_time
            logger.info(
                "📋 MsgGroupQueueManager[%s] 软性关闭检查: 剩余消息=%d 条, 已等待=%.1f 秒",
                self.name,
                total_remaining,
                elapsed_time,
            )
            return False

        else:
            # 硬性关闭
            await self._perform_hard_shutdown()
            self._shutdown_state.reset()
            return True

    async def _perform_hard_shutdown(self):
        """执行硬性关闭"""
        await self.stop_periodic_logging()

        # 统计未处理消息
        total_remaining = 0
        queue_details = []

        for i, queue in enumerate(self._queues):
            queue_size = queue.qsize()
            total_remaining += queue_size

            if queue_size > 0:
                queue_details.append(f"队列[{i}]: {queue_size}条")

            # 清空队列
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if total_remaining > 0:
            logger.warning(
                "⚠️ MsgGroupQueueManager[%s] 硬性关闭，丢弃了 %d 条未处理消息。详情: %s",
                self.name,
                total_remaining,
                ", ".join(queue_details),
            )
        else:
            logger.info(
                "🔌 MsgGroupQueueManager[%s] 已安全关闭，无未处理消息", self.name
            )

    def __repr__(self) -> str:
        return (
            f"MsgGroupQueueManager(name={self.name}, "
            f"num_queues={self.num_queues}, "
            f"max_total_messages={self.max_total_messages})"
        )
