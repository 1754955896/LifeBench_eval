"""
基于 aiolimiter 的异步限流装饰器模块

提供对异步函数的请求频率限制功能，支持灵活的限流配置。
"""

from functools import wraps
from typing import Callable, Any, Dict, Optional
from aiolimiter import AsyncLimiter


class RateLimitManager:
    """限流管理器，管理多个限流器实例"""

    def __init__(self):
        self._limiters: Dict[str, AsyncLimiter] = {}

    def get_limiter(self, key: str, max_rate: int, time_period: int) -> AsyncLimiter:
        """
        获取或创建限流器实例

        Args:
            key: 限流器唯一标识
            max_rate: 时间窗口内允许的最大请求数
            time_period: 时间窗口大小（秒）

        Returns:
            AsyncLimiter: 限流器实例
        """
        limiter_key = f"{key}_{max_rate}_{time_period}"

        if limiter_key not in self._limiters:
            self._limiters[limiter_key] = AsyncLimiter(max_rate, time_period)

        return self._limiters[limiter_key]


# 全局限流管理器实例
_rate_limit_manager = RateLimitManager()


def rate_limit(
    max_rate: int = 3,
    time_period: int = 10,
    key_func: Optional[Callable[..., str]] = None,
):
    """
    异步函数限流装饰器

    Args:
        max_rate: 时间窗口内允许的最大请求数，默认3次
        time_period: 时间窗口大小（秒），默认10秒
        key_func: 可选的键函数，用于为不同参数生成不同的限流键
                 如果不提供，则所有调用共享同一个限流器

    Raises:
        ValueError: 当 max_rate <= 0 或 time_period <= 0 时抛出

    Usage:
        @rate_limit(max_rate=3, time_period=10)
        async def my_api_call():
            pass

        @rate_limit(max_rate=5, time_period=60, key_func=lambda user_id: f"user_{user_id}")
        async def user_specific_call(user_id: str):
            pass
    """
    if max_rate <= 0:
        raise ValueError(f"max_rate must be positive, got {max_rate}")
    if time_period <= 0:
        raise ValueError(f"time_period must be positive, got {time_period}")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # 生成限流器的键
            if key_func:
                # 使用自定义键函数
                try:
                    limiter_key = key_func(*args, **kwargs)
                except (TypeError, ValueError, KeyError):
                    # 如果键函数执行失败，使用函数名作为默认键
                    limiter_key = func.__name__
            else:
                # 使用函数名作为默认键
                limiter_key = func.__name__

            # 获取限流器
            limiter = _rate_limit_manager.get_limiter(
                limiter_key, max_rate, time_period
            )

            # 等待限流器允许执行
            async with limiter:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# 预定义的常用限流装饰器
def rate_limit_3_per_10s(func: Callable) -> Callable:
    """每10秒最多3次请求的限流装饰器"""
    return rate_limit(max_rate=3, time_period=10)(func)


def rate_limit_5_per_minute(func: Callable) -> Callable:
    """每分钟最多5次请求的限流装饰器"""
    return rate_limit(max_rate=5, time_period=60)(func)


def rate_limit_10_per_hour(func: Callable) -> Callable:
    """每小时最多10次请求的限流装饰器"""
    return rate_limit(max_rate=10, time_period=3600)(func)
