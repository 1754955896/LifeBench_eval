import functools
import asyncio
from typing import Optional, Callable, Any
from fastapi import HTTPException

from .enums import Role
from .interfaces import AuthorizationStrategy, AuthorizationContext
from .strategies import DefaultAuthorizationStrategy
from core.context.context import get_current_user_info
from core.observation.logger import get_logger

logger = get_logger(__name__)


def authorize(
    required_role: Role = Role.ANONYMOUS,
    strategy: Optional[AuthorizationStrategy] = None,
    **kwargs,
):
    """
    授权装饰器

    Args:
        required_role: 需要的角色，默认为匿名
        strategy: 自定义授权策略，如果为None则使用默认策略
        **kwargs: 传递给策略的额外参数

    Returns:
        装饰后的函数
    """

    def decorator(func: Callable) -> Callable:
        # 创建授权上下文
        auth_context = AuthorizationContext(
            required_role=required_role,
            strategy=strategy or DefaultAuthorizationStrategy(),
            **kwargs,
        )

        # 将授权信息存储到函数上
        setattr(func, '__authorization_context__', auth_context)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _execute_with_authorization(
                func, auth_context, *args, **kwargs
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _execute_with_authorization_sync(func, auth_context, *args, **kwargs)

        # 根据函数类型返回相应的wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def _execute_with_authorization(
    func: Callable, auth_context: AuthorizationContext, *args, **kwargs
) -> Any:
    """
    异步执行函数并进行授权检查

    Args:
        func: 要执行的函数
        auth_context: 授权上下文
        *args: 函数参数
        **kwargs: 函数关键字参数

    Returns:
        函数的返回值

    Raises:
        HTTPException: 当授权失败时
    """
    # 获取当前用户信息
    user_info = get_current_user_info()

    # 执行授权检查
    has_permission = await auth_context.strategy.check_permission(
        user_info=user_info,
        required_role=auth_context.required_role,
        **auth_context.extra_kwargs,
    )

    if not has_permission:
        logger.warning(
            "授权失败: 用户=%s, 需要角色=%s", user_info, auth_context.required_role
        )
        raise HTTPException(
            status_code=403,
            detail=f"权限不足，需要角色: {auth_context.required_role.value}",
        )

    # 授权通过，执行原函数
    logger.debug("授权通过: 用户=%s, 角色=%s", user_info, auth_context.required_role)
    return await func(*args, **kwargs)


def _execute_with_authorization_sync(
    func: Callable, auth_context: AuthorizationContext, *args, **kwargs
) -> Any:
    """
    同步执行函数并进行授权检查

    Args:
        func: 要执行的函数
        auth_context: 授权上下文
        *args: 函数参数
        **kwargs: 函数关键字参数

    Returns:
        函数的返回值

    Raises:
        HTTPException: 当授权失败时
    """
    # 获取当前用户信息
    user_info = get_current_user_info()

    # 对于同步函数，我们需要在事件循环中运行异步授权检查
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # 执行授权检查
    has_permission = loop.run_until_complete(
        auth_context.strategy.check_permission(
            user_info=user_info,
            required_role=auth_context.required_role,
            **auth_context.extra_kwargs,
        )
    )

    if not has_permission:
        logger.warning(
            "授权失败: 用户=%s, 需要角色=%s", user_info, auth_context.required_role
        )
        raise HTTPException(
            status_code=403,
            detail=f"权限不足，需要角色: {auth_context.required_role.value}",
        )

    # 授权通过，执行原函数
    logger.debug("授权通过: 用户=%s, 角色=%s", user_info, auth_context.required_role)
    return func(*args, **kwargs)


# 便捷装饰器
def require_anonymous(func: Callable) -> Callable:
    """要求匿名访问的装饰器"""
    return authorize(Role.ANONYMOUS)(func)


def require_user(func: Callable) -> Callable:
    """要求用户登录的装饰器"""
    return authorize(Role.USER)(func)


def require_admin(func: Callable) -> Callable:
    """要求管理员权限的装饰器"""
    return authorize(Role.ADMIN)(func)


def require_signature(func: Callable) -> Callable:
    """要求HMAC签名验证的装饰器"""
    return authorize(Role.SIGNATURE)(func)


def custom_authorize(strategy: AuthorizationStrategy, **kwargs):
    """
    自定义授权装饰器

    Args:
        strategy: 自定义授权策略
        **kwargs: 传递给策略的额外参数

    Returns:
        装饰器函数
    """
    return authorize(strategy=strategy, **kwargs)


def check_and_apply_default_auth(func: Callable) -> Callable:
    """
    检查函数是否已有授权装饰器，如果没有则应用默认的 require_user 授权

    处理 bound function 和 unbound function 的情况：
    - 对于 bound method（类方法），需要正确处理 self 参数
    - 对于 unbound function（普通函数），直接应用装饰器

    Args:
        func: 要检查的函数，可能是 bound method 或 unbound function

    Returns:
        Callable: 应用了默认授权的函数（如果还没有授权装饰器）
    """
    # 检查函数是否已经有授权装饰器
    if hasattr(func, '__authorization_context__'):
        return func

    # 检查是否为 bound method（类方法）
    if hasattr(func, '__self__'):
        # 这是一个 bound method，需要获取原始函数
        original_func = func.__func__
        # 检查原始函数是否已有授权装饰器
        if hasattr(original_func, '__authorization_context__'):
            return func

        # 对原始函数应用装饰器，然后重新绑定
        decorated_func = require_user(original_func)
        # 重新绑定到原始对象
        return decorated_func.__get__(func.__self__)
    else:
        # 这是一个 unbound function，直接应用装饰器
        return require_user(func)
