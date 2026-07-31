"""
Recycle consumer base implementation.
循环消费者基础实现。
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from datetime import datetime

from core.longjob.interfaces import (
    LongJobInterface,
    LongJobStatus,
    ConsumerConfig,
    ErrorHandler,
    MessageBatch,
)


class DefaultErrorHandler(ErrorHandler):
    """默认错误处理器"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        默认错误处理：记录错误日志，返回True继续执行

        Args:
            error: 发生的异常
            context: 错误上下文信息

        Returns:
            bool: 是否继续执行
        """
        self.logger.error(
            f"Error in consumer {context.get('job_id', 'unknown')}: {str(error)}",
            exc_info=True,
            extra=context,
        )
        return True


class RecycleConsumerBase(LongJobInterface, ABC):
    """
    循环消费者基础实现
    提供持续消费的基础框架，包含错误处理、重试逻辑、超时处理等功能
    """

    def __init__(
        self,
        job_id: str,
        config: Optional[Dict[str, Any]] = None,
        consumer_config: Optional[ConsumerConfig] = None,
    ):
        """
        初始化循环消费者

        Args:
            job_id: 任务ID
            config: 基础配置
            consumer_config: 消费者专用配置
        """
        super().__init__(job_id, config)
        self.consumer_config = consumer_config or ConsumerConfig()
        self.logger = logging.getLogger(f"{__name__}.{job_id}")
        self._task: Optional[asyncio.Task] = None
        self._error_handler = self.consumer_config.error_handler or DefaultErrorHandler(
            self.logger
        )

        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_errors': 0,
            'total_timeouts': 0,
            'start_time': None,
            'last_processed_time': None,
        }

    async def start(self) -> None:
        """启动消费者"""
        if self.status in [LongJobStatus.RUNNING, LongJobStatus.STARTING]:
            self.logger.warning(
                "Consumer %s is already running or starting", self.job_id
            )
            return

        self.logger.info("Starting consumer %s", self.job_id)
        self.status = LongJobStatus.STARTING

        try:
            # 初始化资源
            await self._initialize()

            # 启动消费循环
            self._task = asyncio.create_task(self._consume_loop())
            self.status = LongJobStatus.RUNNING
            self.stats['start_time'] = datetime.now()

            self.logger.info("Consumer %s started successfully", self.job_id)

        except Exception as e:
            self.status = LongJobStatus.ERROR
            self.logger.error(
                "Failed to start consumer %s: %s", self.job_id, str(e), exc_info=True
            )
            raise

    async def shutdown(
        self, timeout: float = 30.0, wait_for_current_task: bool = True
    ) -> None:
        """
        优雅关闭消费者

        Args:
            timeout: 关闭超时时间（秒）
            wait_for_current_task: 是否等待当前任务完成
        """
        if self.status in [LongJobStatus.STOPPED, LongJobStatus.STOPPING]:
            self.logger.warning(
                "Consumer %s is already stopped or stopping", self.job_id
            )
            return

        self.logger.info("Gracefully shutting down consumer %s", self.job_id)
        self.status = LongJobStatus.STOPPING

        # 请求停止
        self.request_stop()

        # 等待任务完成
        if self._task and not self._task.done():
            if wait_for_current_task:
                try:
                    # 等待当前消息处理完成
                    self.logger.info(
                        "Waiting for current task to complete in consumer %s",
                        self.job_id,
                    )
                    await asyncio.wait_for(self._task, timeout=timeout)
                    self.logger.info(
                        "Current task completed gracefully in consumer %s", self.job_id
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(
                        "Consumer %s shutdown timeout after %ss, cancelling task",
                        self.job_id,
                        timeout,
                    )
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError:
                        self.logger.info("Task cancelled in consumer %s", self.job_id)
            else:
                # 立即取消任务
                self.logger.info(
                    "Immediately cancelling task in consumer %s", self.job_id
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        # 清理资源
        try:
            await self._cleanup()
        except Exception as cleanup_error:
            self.logger.error(
                "Error during cleanup: %s", str(cleanup_error), exc_info=True
            )

        self.status = LongJobStatus.STOPPED
        self.logger.info("Consumer %s shutdown completed", self.job_id)

    async def _consume_loop(self) -> None:
        """主消费循环"""
        self.logger.info("Consumer %s entering consume loop", self.job_id)

        while not self.should_stop():
            try:
                # 检查是否有消息可消费
                if not await self._has_messages():
                    await asyncio.sleep(0.1)  # 短暂休眠避免CPU占用过高
                    continue

                # 消费消息
                await self._consume_messages()

            except Exception as e:
                # 错误处理
                context = {
                    'job_id': self.job_id,
                    'timestamp': datetime.now().isoformat(),
                    'stats': self.stats.copy(),
                }

                self.stats['total_errors'] += 1

                try:
                    should_continue = await self._error_handler.handle_error(e, context)
                    if not should_continue:
                        self.logger.error(
                            "Error handler requested stop for consumer %s", self.job_id
                        )
                        break
                except Exception as handler_error:
                    self.logger.error(
                        "Error in error handler for consumer %s: %s",
                        self.job_id,
                        str(handler_error),
                        exc_info=True,
                    )
                    # 如果错误处理器本身出错，短暂休眠后继续
                    await asyncio.sleep(1.0)

        self.logger.info("Consumer %s exiting consume loop", self.job_id)

    async def _consume_messages(self) -> None:
        """消费消息的核心逻辑"""
        timeout = self.consumer_config.timeout

        # 业务代码自己控制批量处理，这里只处理单个消息
        if self.should_stop():
            return

        try:
            # 使用超时控制单个消息的处理时间
            await asyncio.wait_for(self._process_single_message(), timeout=timeout)

            self.stats['total_processed'] += 1
            self.stats['last_processed_time'] = datetime.now()

        except asyncio.TimeoutError:
            self.stats['total_timeouts'] += 1
            self.logger.warning(
                "Message processing timeout in consumer %s (timeout: %ss)",
                self.job_id,
                timeout,
            )
            # 超时也算作错误，交给错误处理器处理
            timeout_error = TimeoutError(f"Message processing timeout ({timeout}s)")
            context = {
                'job_id': self.job_id,
                'error_type': 'timeout',
                'timeout': timeout,
                'timestamp': datetime.now().isoformat(),
            }

            try:
                should_continue = await self._error_handler.handle_error(
                    timeout_error, context
                )
                if not should_continue:
                    raise timeout_error
            except Exception as handler_error:
                self.logger.error(
                    "Error in timeout error handler: %s",
                    str(handler_error),
                    exc_info=True,
                )

        except Exception as e:
            # 其他异常会在外层循环中被捕获和处理
            raise

    async def _process_single_message(self) -> None:
        """
        处理单个消息，包含增强的重试逻辑
        先获取消息，然后在重试时传递相同的消息给处理逻辑
        """
        retry_config = self.consumer_config.retry_config
        last_error = None
        message_batch = None

        for attempt in range(retry_config.max_retries + 1):
            try:
                # 第一次尝试时获取消息，重试时使用相同消息批次
                if attempt == 0:
                    raw_message = await self._fetch_message()
                    if raw_message is None:
                        return  # 没有消息可处理

                    # 如果不是MessageBatch，自动包装
                    if isinstance(raw_message, MessageBatch):
                        message_batch = raw_message
                    else:
                        message_batch = MessageBatch(
                            data=raw_message,
                            batch_id=f"auto_wrapped_{id(raw_message)}",
                            metadata={'auto_wrapped': True},
                        )

                    if message_batch.is_empty:
                        return  # 没有消息可处理

                # 调用子类实现的具体消息处理逻辑，传递消息批次
                await self._handle_message(message_batch)
                return  # 成功处理，直接返回

            except Exception as e:
                last_error = e

                # 检查是否为致命错误且不允许重试致命错误
                if (
                    self._error_handler.is_fatal_error(e)
                    and not retry_config.retry_on_fatal
                ):
                    self.logger.error(
                        "Fatal error in consumer %s, not retrying: %s",
                        self.job_id,
                        str(e),
                    )
                    raise

                # 检查是否为不可重试的错误
                if (
                    not self._error_handler.is_retryable_error(e)
                    and not retry_config.retry_on_fatal
                ):
                    self.logger.error(
                        "Non-retryable error in consumer %s: %s", self.job_id, str(e)
                    )
                    raise

                if attempt < retry_config.max_retries:
                    # 计算重试延迟
                    delay = self._calculate_retry_delay(attempt, retry_config)

                    self.logger.warning(
                        "Message processing failed (attempt %d/%d) in consumer %s, retrying in %ss: %s",
                        attempt + 1,
                        retry_config.max_retries + 1,
                        self.job_id,
                        delay,
                        str(e),
                    )

                    await asyncio.sleep(delay)
                else:
                    # 达到最大重试次数，抛出异常
                    self.logger.error(
                        "Message processing failed after %d attempts in consumer %s: %s",
                        retry_config.max_retries + 1,
                        self.job_id,
                        str(e),
                    )
                    raise last_error

        # 理论上不会到达这里，但为了安全起见
        if last_error:
            raise last_error from None

    def _calculate_retry_delay(self, attempt: int, retry_config) -> float:
        """
        计算重试延迟时间，支持指数退避和随机抖动

        Args:
            attempt: 当前重试次数（从0开始）
            retry_config: 重试配置

        Returns:
            float: 延迟时间（秒）
        """
        if retry_config.exponential_backoff:
            # 指数退避
            delay = retry_config.retry_delay * (
                retry_config.backoff_multiplier**attempt
            )
        else:
            # 固定延迟
            delay = retry_config.retry_delay

        # 限制最大延迟
        delay = min(delay, retry_config.max_delay)

        # 添加随机抖动
        if retry_config.jitter:
            # 在 50% 到 150% 之间随机
            jitter_factor = 0.5 + random.random()
            delay = delay * jitter_factor

        return delay

    def get_stats(self) -> Dict[str, Any]:
        """获取消费者统计信息"""
        stats = self.stats.copy()
        stats['status'] = self.status.value
        stats['uptime'] = None

        if stats['start_time']:
            uptime = datetime.now() - stats['start_time']
            stats['uptime'] = uptime.total_seconds()

        return stats

    @abstractmethod
    async def _initialize(self) -> None:
        """
        初始化资源
        子类需要实现此方法来初始化特定的资源（如连接、队列等）
        """

    @abstractmethod
    async def _cleanup(self) -> None:
        """
        清理资源
        子类需要实现此方法来清理特定的资源
        """

    @abstractmethod
    async def _has_messages(self) -> bool:
        """
        检查是否有消息可消费
        子类需要实现此方法来检查消息源是否有新消息

        Returns:
            bool: 是否有消息可消费
        """

    @abstractmethod
    async def _fetch_message(self) -> Optional[Any]:
        """
        获取消息数据
        子类需要实现此方法来从消息源获取消息，可以返回任何类型的数据
        框架会自动判断类型，如果不是MessageBatch会自动包装

        Returns:
            Optional[Any]: 获取到的消息数据，可以是任何类型，如果没有消息返回None
        """

    @abstractmethod
    async def _handle_message(self, message_batch: MessageBatch) -> None:
        """
        处理消息批次的具体逻辑
        子类需要实现此方法来定义具体的消息处理逻辑

        Args:
            message_batch: 要处理的消息批次，由 _fetch_message 返回

        Note:
            此方法应该处理传入的消息批次，如果处理失败应该抛出异常
            重试逻辑由基类处理，重试时会传递相同的消息批次
            子类可以通过message_batch.messages获取所有消息，自己决定如何处理（单个或批量）
        """
