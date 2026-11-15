"""
Long job interfaces and base classes.
长任务接口和基础类定义。
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import asyncio
from enum import Enum
from dataclasses import dataclass

# 从 longjob_error 中导入错误类
from core.longjob.longjob_error import FatalError, BusinessLogicError


@dataclass
class MessageBatch:
    """
    消息包装类
    统一封装消息数据，不限制具体类型（可以是单个消息、列表、或任何业务定义的结构）
    """

    data: Any  # 消息数据，可以是任何类型：单个消息、列表、字典等
    batch_id: Optional[str] = None  # 批次ID，用于追踪和日志
    metadata: Optional[Dict[str, Any]] = None  # 额外的元数据信息

    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_empty(self) -> bool:
        """检查是否为空数据"""
        if self.data is None:
            return True

        # 如果是列表或类似容器，检查长度
        if hasattr(self.data, '__len__'):
            try:
                return len(self.data) == 0
            except (TypeError, AttributeError):
                pass

        return False


class LongJobStatus(Enum):
    """长任务状态枚举"""

    IDLE = "idle"  # 空闲状态
    STARTING = "starting"  # 启动中
    RUNNING = "running"  # 运行中
    STOPPING = "stopping"  # 停止中
    STOPPED = "stopped"  # 已停止
    ERROR = "error"  # 错误状态


class LongJobInterface(ABC):
    """
    长任务接口定义。
    所有长任务都需要实现这个接口。
    """

    def __init__(self, job_id: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化长任务

        Args:
            job_id: 任务ID，用于标识和管理
            config: 任务配置参数
        """
        self.job_id = job_id
        self.config = config or {}
        self.status = LongJobStatus.IDLE
        self._stop_event = asyncio.Event()

    @abstractmethod
    async def start(self) -> None:
        """
        启动长任务
        实现类需要在这里启动具体的工作逻辑
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        关闭长任务
        实现类需要在这里清理资源和停止工作
        """

    def get_status(self) -> LongJobStatus:
        """获取当前任务状态"""
        return self.status

    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        return self.status == LongJobStatus.RUNNING

    def should_stop(self) -> bool:
        """检查是否应该停止任务"""
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        """请求停止任务"""
        self._stop_event.set()


class ErrorHandler(ABC):
    """
    错误处理器接口
    用于处理长任务执行过程中的异常
    """

    @abstractmethod
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        处理错误

        Args:
            error: 发生的异常
            context: 错误上下文信息

        Returns:
            bool: True表示可以继续执行，False表示应该停止
        """

    def is_fatal_error(self, error: Exception) -> bool:
        """
        判断是否为致命错误

        Args:
            error: 异常实例

        Returns:
            bool: True表示致命错误，不应重试
        """
        # 检查是否为明确标识的致命错误
        if isinstance(error, FatalError):
            return True

        # 检查常见的致命错误类型
        fatal_error_types = (
            MemoryError,
            SystemExit,
            KeyboardInterrupt,
            ImportError,
            SyntaxError,
            TypeError,  # 通常表示编程错误
            AttributeError,  # 通常表示编程错误
        )

        return isinstance(error, fatal_error_types)

    def is_retryable_error(self, error: Exception) -> bool:
        """
        判断是否为可重试错误

        Args:
            error: 异常实例

        Returns:
            bool: True表示可以重试
        """
        # 如果是致命错误，不可重试
        if self.is_fatal_error(error):
            return False

        # 明确标识的业务逻辑错误可以重试
        if isinstance(error, BusinessLogicError):
            return True

        # 网络相关错误通常可以重试
        retryable_error_types = (ConnectionError, TimeoutError, OSError)  # 包含网络错误

        return isinstance(error, retryable_error_types)


class RetryConfig:
    """重试配置"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True,
        max_delay: float = 60.0,
        jitter: bool = True,
        backoff_multiplier: float = 2.0,
        retry_on_fatal: bool = False,
    ):
        """
        初始化重试配置

        Args:
            max_retries: 最大重试次数
            retry_delay: 初始重试延迟时间（秒）
            exponential_backoff: 是否使用指数退避
            max_delay: 最大延迟时间（秒）
            jitter: 是否添加随机抖动
            backoff_multiplier: 指数退避倍数
            retry_on_fatal: 是否对致命错误也进行重试（通常为False）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.max_delay = max_delay
        self.jitter = jitter
        self.backoff_multiplier = backoff_multiplier
        self.retry_on_fatal = retry_on_fatal


class ConsumerConfig:
    """消费者配置"""

    def __init__(
        self,
        timeout: float = 600.0,
        retry_config: Optional[RetryConfig] = None,
        error_handler: Optional[ErrorHandler] = None,
    ):
        """
        初始化消费者配置

        Args:
            timeout: 单个消息消费超时时间（秒），包括重试
            retry_config: 重试配置
            error_handler: 错误处理器
        """
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.error_handler = error_handler
