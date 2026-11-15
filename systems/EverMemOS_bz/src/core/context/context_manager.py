from contextvars import copy_context, Context
from typing import Optional, Dict, Any, Callable, TypeVar, Coroutine, Union, Tuple
from functools import wraps
from sqlmodel.ext.asyncio.session import AsyncSession

from core.context.context import (
    set_current_session,
    clear_current_session,
    get_current_session,
    set_current_user_info,
    get_current_user_info,
    clear_current_user_context,
)
from component.database_session_provider import DatabaseSessionProvider
from core.di.decorators import component
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Coroutine[Any, Any, Any]])


@component(name="database_session_manager")
class DatabaseSessionManager:
    """
    数据库会话管理器

    负责数据库会话的创建、设置、提交、回滚和清理
    """

    def __init__(self, db_provider: DatabaseSessionProvider):
        self.db_provider = db_provider

    async def run_with_session(
        self,
        func: Callable,
        *args,
        session: Optional[AsyncSession] = None,
        auto_commit: bool = True,
        force_new_session: bool = False,
        **kwargs,
    ) -> Any:
        """
        在数据库会话中运行函数

        Args:
            func: 要运行的函数
            *args: 函数的位置参数
            session: 数据库会话（可选，不提供则自动创建）
            auto_commit: 是否自动提交事务
            force_new_session: 是否强制创建新会话（用于避免会话并发冲突）
            **kwargs: 函数的关键字参数

        Returns:
            函数的返回值
        """
        # 根据force_new_session参数决定session处理策略
        if force_new_session:
            # 强制创建新会话，忽略传入的session和当前上下文中的会话
            session = self.db_provider.create_session()
            need_cleanup = True
            logger.debug("强制创建新的数据库会话（避免并发冲突）")
        else:
            # 正常逻辑：优先使用传入的session，其次使用当前上下文中的session，最后创建新session
            if session is None:
                try:
                    current_session = get_current_session()
                except RuntimeError:
                    current_session = None
                if current_session is not None:
                    session = current_session
                    need_cleanup = False
                    logger.debug("使用当前上下文中的数据库会话")
                else:
                    session = self.db_provider.create_session()
                    need_cleanup = True
                    logger.debug("创建新的数据库会话")
            else:
                # 使用传入的session
                need_cleanup = False

        # 设置上下文
        db_token = set_current_session(session)

        try:
            # 运行函数
            result = await func(*args, **kwargs)

            # 如果没有异常且启用自动提交，提交事务
            if auto_commit and need_cleanup and session.is_active:
                await session.commit()
                logger.debug("数据库会话管理器：自动提交事务")

            return result

        except Exception as e:
            # 如果有异常，回滚事务
            if need_cleanup and session.is_active:
                try:
                    await session.rollback()
                    logger.debug("数据库会话管理器：自动回滚事务")
                except Exception as rollback_error:
                    logger.error(f"回滚事务时发生错误: {str(rollback_error)}")

            # 重新抛出异常
            raise e

        finally:
            # 清理上下文
            clear_current_session(db_token)

            # 关闭会话（如果是自动创建的）
            if need_cleanup:
                try:
                    await session.close()
                    logger.debug("数据库会话管理器：已关闭数据库会话")
                except Exception as close_error:
                    logger.error(f"关闭数据库会话时发生错误: {str(close_error)}")


@component(name="user_context_manager")
class UserContextManager:
    """
    用户上下文管理器

    负责用户上下文的设置、获取和清理
    """

    def __init__(self):
        pass

    async def run_with_user_context(
        self,
        func: Callable,
        *args,
        user_data: Optional[Dict[str, Any]] = None,
        auto_inherit: bool = True,
        **kwargs,
    ) -> Any:
        """
        在用户上下文中运行函数

        Args:
            func: 要运行的函数
            *args: 函数的位置参数
            user_data: 用户数据（可选）
            auto_inherit: 是否自动继承当前用户上下文
            **kwargs: 函数的关键字参数

        Returns:
            函数的返回值
        """
        # 决定使用的用户数据
        actual_user_data = user_data
        if auto_inherit and actual_user_data is None:
            actual_user_data = get_current_user_info()

        # 设置用户上下文
        user_token = None
        if actual_user_data is not None:
            user_token = set_current_user_info(actual_user_data)
            logger.debug(
                f"用户上下文管理器：设置用户上下文 user_id={actual_user_data.get('user_id')}"
            )

        try:
            # 运行函数
            result = await func(*args, **kwargs)
            return result

        finally:
            # 清理用户上下文
            if user_token is not None:
                clear_current_user_context(user_token)
                logger.debug("用户上下文管理器：已清理用户上下文")


@component(name="context_manager")
class ContextManager:
    """
    综合上下文管理器

    组合数据库会话管理器和用户上下文管理器，提供统一的上下文管理能力
    """

    def __init__(
        self,
        db_session_manager: DatabaseSessionManager,
        user_context_manager: UserContextManager,
    ):
        self.db_session_manager = db_session_manager
        self.user_context_manager = user_context_manager

    async def run_with_full_context(
        self,
        func: Callable,
        *args,
        user_data: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None,
        auto_commit: bool = True,
        auto_inherit_user: bool = True,
        force_new_session: bool = False,
        **kwargs,
    ) -> Any:
        """
        在完整上下文（数据库会话 + 用户上下文）中运行函数

        Args:
            func: 要运行的函数
            *args: 函数的位置参数
            user_data: 用户数据（可选）
            session: 数据库会话（可选）
            auto_commit: 是否自动提交事务
            auto_inherit_user: 是否自动继承用户上下文
            force_new_session: 是否强制创建新会话（用于避免会话并发冲突）
            **kwargs: 函数的关键字参数

        Returns:
            函数的返回值
        """
        # 先设置用户上下文，再设置数据库会话
        # 这样在数据库操作中就能访问到用户信息
        return await self.user_context_manager.run_with_user_context(
            self.db_session_manager.run_with_session,
            func,
            *args,
            session=session,
            auto_commit=auto_commit,
            force_new_session=force_new_session,
            user_data=user_data,
            auto_inherit=auto_inherit_user,
            **kwargs,
        )

    async def run_with_database_only(
        self,
        func: Callable,
        *args,
        session: Optional[AsyncSession] = None,
        auto_commit: bool = True,
        force_new_session: bool = False,
        **kwargs,
    ) -> Any:
        """
        仅在数据库会话中运行函数

        Args:
            func: 要运行的函数
            *args: 函数的位置参数
            session: 数据库会话（可选）
            auto_commit: 是否自动提交事务
            force_new_session: 是否强制创建新会话
            **kwargs: 函数的关键字参数

        Returns:
            函数的返回值
        """
        return await self.db_session_manager.run_with_session(
            func,
            *args,
            session=session,
            auto_commit=auto_commit,
            force_new_session=force_new_session,
            **kwargs,
        )

    async def run_with_user_only(
        self,
        func: Callable,
        *args,
        user_data: Optional[Dict[str, Any]] = None,
        auto_inherit: bool = True,
        **kwargs,
    ) -> Any:
        """
        仅在用户上下文中运行函数

        Args:
            func: 要运行的函数
            *args: 函数的位置参数
            user_data: 用户数据（可选）
            auto_inherit: 是否自动继承用户上下文
            **kwargs: 函数的关键字参数

        Returns:
            函数的返回值
        """
        return await self.user_context_manager.run_with_user_context(
            func, *args, user_data=user_data, auto_inherit=auto_inherit, **kwargs
        )

    def copy_current_context(self) -> Context:
        """
        复制当前上下文

        Returns:
            Context: 当前上下文的副本
        """
        return copy_context()

    def get_current_context_data(self) -> Dict[str, Any]:
        """
        获取当前上下文的数据

        Returns:
            Dict[str, Any]: 包含当前上下文数据的字典
        """
        user_data = get_current_user_info()
        return {
            "user_context": user_data,
            "user_id": user_data.get("user_id") if user_data else None,
            "has_session": get_current_session() is not None,
        }


# 装饰器工厂函数
def with_full_context(
    user_data: Optional[Dict[str, Any]] = None,
    session: Optional[AsyncSession] = None,
    auto_commit: bool = True,
    auto_inherit_user: bool = True,
):
    """
    装饰器：为函数提供完整上下文注入（数据库会话 + 用户上下文）
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context_manager = get_bean_by_type(ContextManager)
            return await context_manager.run_with_full_context(
                func,
                *args,
                user_data=user_data,
                session=session,
                auto_commit=auto_commit,
                auto_inherit_user=auto_inherit_user,
                **kwargs,
            )

        return wrapper

    return decorator


def with_database_session(
    session: Optional[AsyncSession] = None,
    auto_commit: bool = True,
    force_new_session: bool = False,
):
    """
    装饰器：为函数提供数据库会话注入

    Args:
        session: 数据库会话（可选）
        auto_commit: 是否自动提交事务
        force_new_session: 是否强制创建新会话（用于避免会话并发冲突）
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context_manager = get_bean_by_type(ContextManager)
            return await context_manager.run_with_database_only(
                func,
                *args,
                session=session,
                auto_commit=auto_commit,
                force_new_session=force_new_session,
                **kwargs,
            )

        return wrapper

    return decorator


def with_user_context(
    user_data: Optional[Dict[str, Any]] = None, auto_inherit: bool = True
):
    """
    装饰器：为函数提供用户上下文注入
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context_manager = get_bean_by_type(ContextManager)
            return await context_manager.run_with_user_only(
                func, *args, user_data=user_data, auto_inherit=auto_inherit, **kwargs
            )

        return wrapper

    return decorator
