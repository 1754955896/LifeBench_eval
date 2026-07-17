import logging
import traceback
from typing import Any, Optional
from enum import Enum
from functools import lru_cache
import sys
import os
from datetime import datetime


class LogLevel(Enum):
    """日志级别枚举"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerProvider:
    """统一的日志管理类 - 混合模式+LRU缓存优化"""

    _instance: Optional['LoggerProvider'] = None
    _initialized: bool = False

    def __new__(cls) -> 'LoggerProvider':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化日志提供者"""
        if not self._initialized:
            self._setup_logging()
            self._initialized = True

    def _setup_root_logging(self):
        """设置日志配置"""
        # 获取环境变量中的日志级别，默认为INFO
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

        # 配置根日志器
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                # 可以添加文件处理器
                # logging.FileHandler('app.log', encoding='utf-8')
            ],
        )

    def _setup_logging(self):
        """设置日志配置"""
        self._setup_root_logging()

        # 禁用第三方库的冗余日志
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(logging.WARNING)
        logging.getLogger('googleapiclient').setLevel(logging.WARNING)
        # 禁用httpx的INFO级别日志，避免频繁的HTTP请求日志
        logging.getLogger('httpx').setLevel(logging.WARNING)
        # 禁用HTTP相关库的调试日志，避免冗余的网络请求日志
        logging.getLogger('hpack').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('pymongo').setLevel(logging.WARNING)
        logging.getLogger('aiokafka').setLevel(logging.WARNING)
        # 禁用websockets客户端的调试日志，避免冗余的连接日志
        # logging.getLogger('websockets.client').setLevel(logging.WARNING)

    @lru_cache(maxsize=1000)
    def _get_cached_logger(self, module_name: str) -> logging.Logger:
        """获取缓存的日志器（LRU缓存，最多1000个）

        Args:
            module_name: 模块名称

        Returns:
            logging.Logger: 缓存的日志器实例
        """
        return logging.getLogger(f'{module_name}')

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """获取指定名称的日志器（推荐用法：显式传入模块名）

        Args:
            name: 日志器名称，如果为None则使用调用者模块名（性能较低）

        Returns:
            logging.Logger: 日志器实例
        """
        if name is None:
            # 获取调用者的模块名（便捷但性能较低）
            frame = sys._getframe(1)
            name = frame.f_globals.get('__name__', 'unknown')

        # 使用LRU缓存避免重复创建logger
        return self._get_cached_logger(name)

    def debug(self, message: str, *args, **kwargs):
        """记录调试信息"""
        logger = self._get_caller_logger()
        logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """记录信息"""
        logger = self._get_caller_logger()
        logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """记录警告"""
        logger = self._get_caller_logger()
        logger.warning(message, *args, **kwargs)

    def warn(self, message: str, *args, **kwargs):
        """记录警告（别名）"""
        self.warning(message, *args, **kwargs)

    def error(self, message: str, exc_info: bool = True, *args, **kwargs):
        """记录错误信息

        Args:
            message: 错误消息
            exc_info: 是否包含异常堆栈信息，默认True
        """
        logger = self._get_caller_logger()
        logger.error(message, exc_info=exc_info, *args, **kwargs)

    def exception(self, message: str, exc_info: bool = True, *args, **kwargs):
        """记录异常信息（自动包含堆栈跟踪）

        Args:
            message: 异常消息
            exc_info: 是否包含异常堆栈信息，默认True
        """
        logger = self._get_caller_logger()
        logger.exception(message, exc_info=exc_info, *args, **kwargs)

    def critical(self, message: str, exc_info: bool = True, *args, **kwargs):
        """记录严重错误信息

        Args:
            message: 错误消息
            exc_info: 是否包含异常堆栈信息，默认True
        """
        logger = self._get_caller_logger()
        logger.critical(message, exc_info=exc_info, *args, **kwargs)

    def log_with_stack(self, level: LogLevel, message: str):
        """记录带有完整堆栈信息的日志

        Args:
            level: 日志级别
            message: 日志消息
        """
        logger = self._get_caller_logger()
        stack_trace = traceback.format_stack()
        full_message = f"{message}\n堆栈跟踪:\n{''.join(stack_trace)}"

        log_method = getattr(logger, level.value.lower())
        log_method(full_message)

    def _get_caller_logger(self) -> logging.Logger:
        """获取调用者的日志器（带LRU缓存优化）"""
        frame = sys._getframe(2)  # 跳过当前方法和调用的日志方法
        module_name = frame.f_globals.get('__name__', 'unknown')
        # 使用LRU缓存避免重复创建logger
        return self._get_cached_logger(module_name)


# 创建全局日志提供者实例
logger_provider = LoggerProvider()

# 混合模式接口：提供两种使用方式


# 方式1: 高性能用法（推荐）- 显式获取logger，适合频繁调用
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志器（推荐用法：显式传入__name__）

    推荐用法:
        logger = get_logger(__name__)  # 模块顶部获取一次
        logger.info("高频调用的日志")    # 后续直接使用

    Args:
        name: 模块名，推荐传入__name__。如果为None则自动获取（性能较低）
    """
    return logger_provider.get_logger(name)


# 方式2: 便捷用法 - 直接调用函数，适合偶尔调用（带LRU缓存优化）
def debug(message: str, *args, **kwargs):
    """记录调试信息（便捷用法，适合偶尔调用）"""
    logger_provider.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """记录信息（便捷用法，适合偶尔调用）"""
    logger_provider.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """记录警告（便捷用法，适合偶尔调用）"""
    logger_provider.warning(message, *args, **kwargs)


def warn(message: str, *args, **kwargs):
    """记录警告（别名）"""
    logger_provider.warn(message, *args, **kwargs)


def error(message: str, exc_info: bool = True, *args, **kwargs):
    """记录错误信息（自动包含堆栈跟踪）"""
    logger_provider.error(message, exc_info=exc_info, *args, **kwargs)


def exception(message: str, exc_info: bool = True, *args, **kwargs):
    """记录异常信息（自动包含堆栈跟踪）"""
    logger_provider.exception(message, exc_info=exc_info, *args, **kwargs)


def critical(message: str, exc_info: bool = True, *args, **kwargs):
    """记录严重错误信息（自动包含堆栈跟踪）"""
    logger_provider.critical(message, exc_info=exc_info, *args, **kwargs)


def log_with_stack(level: LogLevel, message: str):
    """记录带有完整堆栈信息的日志"""
    logger_provider.log_with_stack(level, message)
