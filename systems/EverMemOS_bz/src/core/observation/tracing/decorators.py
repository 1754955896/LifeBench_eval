"""
装饰器模块

此模块包含各种装饰器，用于在方法执行前进行验证和处理。
"""

from functools import wraps
from typing import Any, Dict, Callable, Optional
import logging
import time

logger = logging.getLogger(__name__)


def trace_logger(
    operation_name: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False,
    log_level: str = "debug",
):
    """
    自动添加 [trace] 日志的装饰器

    Args:
        operation_name: 操作名称，如果不提供则使用函数名
        include_args: 是否记录函数参数
        include_result: 是否记录函数返回值
        log_level: 日志级别 (debug, info, warning, error)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            operation = operation_name or func.__name__

            # 检查日志级别是否启用
            if not _is_log_level_enabled(logger, log_level):
                # 如果日志级别不启用，直接执行函数，避免性能损耗
                return await func(*args, **kwargs)

            start_time = time.time()

            # 记录开始日志
            log_message = f"\n\t[trace] {operation} - 开始处理"
            if include_args and (args or kwargs):
                args_str = _format_args(args, kwargs)
                log_message += f" | 参数: {args_str}"

            _log_message(logger, log_level, log_message)

            try:
                # 执行原函数
                result = await func(*args, **kwargs)

                # 记录成功完成日志
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)  # 毫秒

                log_message = f"\n\t[trace] {operation} - 处理完成 (耗时: {duration}ms)"
                if include_result and result is not None:
                    result_str = _format_result(result)
                    log_message += f" | 结果: {result_str}"

                _log_message(logger, log_level, log_message)
                return result

            except Exception as e:
                # 记录异常日志
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)

                log_message = f"\n\t[trace] {operation} - 处理失败 (耗时: {duration}ms) | 错误: {str(e)}"
                _log_message(logger, "error", log_message)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            operation = operation_name or func.__name__

            # 检查日志级别是否启用
            if not _is_log_level_enabled(logger, log_level):
                # 如果日志级别不启用，直接执行函数，避免性能损耗
                return func(*args, **kwargs)

            start_time = time.time()

            # 记录开始日志
            log_message = f"\n\t[trace] {operation} - 开始处理"
            if include_args and (args or kwargs):
                args_str = _format_args(args, kwargs)
                log_message += f" | 参数: {args_str}"

            _log_message(logger, log_level, log_message)

            try:
                # 执行原函数
                result = func(*args, **kwargs)

                # 记录成功完成日志
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)

                log_message = f"\n\t[trace] {operation} - 处理完成 (耗时: {duration}ms)"
                if include_result and result is not None:
                    result_str = _format_result(result)
                    log_message += f" | 结果: {result_str}"

                _log_message(logger, log_level, log_message)
                return result

            except Exception as e:
                # 记录异常日志
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)

                log_message = f"\n\t[trace] {operation} - 处理失败 (耗时: {duration}ms) | 错误: {str(e)}"
                _log_message(logger, "error", log_message)
                raise

        # 根据函数是否为协程函数返回对应的包装器
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _is_log_level_enabled(logger, level: str) -> bool:
    """检查日志级别是否启用"""
    level_num = getattr(logging, level.upper(), logging.INFO)
    return logger.isEnabledFor(level_num)


def _log_message(logger, level: str, message: str):
    """根据级别记录日志"""
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)


def _format_args(args, kwargs) -> str:
    """格式化函数参数"""
    args_str = []

    # 处理位置参数
    for i, arg in enumerate(args):
        if hasattr(arg, '__dict__'):  # 对象类型
            args_str.append(f"arg{i}: {type(arg).__name__}")
        elif isinstance(arg, (list, dict)) and len(str(arg)) > 100:  # 大对象
            args_str.append(f"arg{i}: {type(arg).__name__}(len={len(arg)})")
        else:
            args_str.append(f"arg{i}: {arg}")

    # 处理关键字参数
    for key, value in kwargs.items():
        if hasattr(value, '__dict__'):  # 对象类型
            args_str.append(f"{key}: {type(value).__name__}")
        elif isinstance(value, (list, dict)) and len(str(value)) > 100:  # 大对象
            args_str.append(f"{key}: {type(value).__name__}(len={len(value)})")
        else:
            args_str.append(f"{key}: {value}")

    return ", ".join(args_str)


def _format_result(result) -> str:
    """格式化函数返回值"""
    if hasattr(result, '__dict__'):  # 对象类型
        return f"{type(result).__name__}"
    elif isinstance(result, (list, dict)) and len(str(result)) > 100:  # 大对象
        return f"{type(result).__name__}(len={len(result)})"
    else:
        return str(result)
