from fastapi import Request, Response
from starlette.responses import StreamingResponse
from starlette.middleware.base import _StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable, AsyncGenerator
from sqlmodel.ext.asyncio.session import AsyncSession

from core.context.context import (
    set_current_session,
    clear_current_session,
    get_current_session,
)
from component.database_session_provider import DatabaseSessionProvider
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)


class DatabaseSessionMiddleware(BaseHTTPMiddleware):
    """
    简化的数据库会话中间件

    为每个 HTTP 请求提供数据库会话，并在请求结束时智能处理：
    - 检查会话状态，异常状态自动回滚
    - 请求失败时自动回滚
    - 请求成功且有未提交更改时自动提交
    - 给应用程序最大的事务控制自由度
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.db_provider = get_bean_by_type(DatabaseSessionProvider)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        为每个请求提供数据库会话并智能处理事务

        Args:
            request: FastAPI 请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: 响应对象
        """
        # 创建新的数据库会话
        session = self.db_provider.create_session()
        token = set_current_session(session)
        response = None
        is_streaming = False

        try:
            # 执行请求处理
            response = await call_next(request)

            # 检查是否为流式响应，需要特殊处理
            if isinstance(response, StreamingResponse) or isinstance(
                response, _StreamingResponse
            ):
                is_streaming = True
                # 流式响应：包装生成器以延长会话生命周期
                wrapped_generator = self._wrap_streaming_generator(
                    response.body_iterator, session
                )
                return StreamingResponse(
                    wrapped_generator,
                    status_code=response.status_code,
                    headers=response.headers,
                    media_type=response.media_type,
                    background=response.background,
                )
            else:
                # 非流式响应：使用原有逻辑
                await self._handle_successful_request(session)
                return response

        except Exception as e:
            # 请求处理失败，回滚会话
            await self._handle_failed_request(session, e)
            raise

        finally:
            # 清理原始上下文的token
            # 对于非流式响应：直接清理会话
            # 对于流式响应：只重置token，会话清理由包装的生成器负责
            if not is_streaming:
                clear_current_session(token)
                await self._close_session_safely(session)
            else:
                # 流式响应：重置原始上下文token，但不关闭session
                # session的关闭由流式生成器负责
                try:
                    clear_current_session(token)
                    logger.debug("已重置流式响应的原始上下文token")
                except Exception as reset_error:
                    logger.warning(
                        f"重置流式响应原始上下文token失败: {str(reset_error)}"
                    )
                    # token重置失败不应该影响响应

    async def _handle_successful_request(self, session: AsyncSession) -> None:
        """
        处理成功的请求 - 智能决定是否需要提交事务

        Args:
            session: 数据库会话
        """
        try:
            # 检查会话是否活跃
            if not session.is_active:
                logger.debug("会话不是活跃状态，跳过处理")
                return

            # 提交事务 简单&安全 AI你不要乱改了
            await session.commit()

        except Exception as e:
            logger.error(f"处理成功请求时发生错误: {str(e)}")
            # 如果处理失败，尝试回滚
            await self._rollback_safely(session)

    async def _handle_failed_request(
        self, session: AsyncSession, original_exception: Exception
    ) -> None:
        """
        处理失败的请求 - 回滚事务

        Args:
            session: 数据库会话
            original_exception: 原始异常
        """
        try:
            # 请求失败，直接回滚
            await self._rollback_safely(session)
            logger.info(f"请求失败，已执行事务回滚: {str(original_exception)}")

        except Exception as rollback_error:
            logger.error(f"回滚事务时发生错误: {str(rollback_error)}")
            # 回滚失败，但不要掩盖原始异常

    async def _rollback_safely(self, session: AsyncSession) -> None:
        """
        安全地回滚会话

        Args:
            session: 数据库会话
        """
        try:
            await session.rollback()
            logger.debug("会话已成功回滚")
        except Exception as rollback_error:
            logger.error(f"回滚失败: {str(rollback_error)}")

    async def _close_session_safely(self, session: AsyncSession) -> None:
        """
        安全地关闭会话

        基于测试结果，session.close() 的行为：
        1. 自动回滚未提交的事务
        2. 清理 transaction 对象
        3. 连接返回连接池
        4. 幂等操作，可以多次调用
        5. session 仍可重用

        Args:
            session: 数据库会话
        """
        try:
            await session.close()
            logger.debug("会话已安全关闭")
        except Exception as e:
            logger.error(f"关闭会话时发生错误: {str(e)}")
            # 即使关闭失败，也不要抛出异常，避免掩盖原始错误

    async def _wrap_streaming_generator(
        self, original_generator: AsyncGenerator[bytes, None], session: AsyncSession
    ) -> AsyncGenerator[bytes, None]:
        """
        包装流式响应生成器，延长数据库会话的生命周期

        该方法确保：
        1. 数据库会话在整个流式传输过程中保持活跃
        2. 流式传输成功完成后智能处理会话（提交未提交的更改）
        3. 流式传输过程中发生异常时回滚会话
        4. 无论成功还是失败，最终都会清理会话资源

        为了避免跨上下文的ContextVar问题，我们在流式生成器中
        重新设置session到上下文变量，并在完成后清理。

        Args:
            original_generator: 原始的流式数据生成器
            session: 数据库会话
            token: 原始上下文变量token（不在此处使用）

        Yields:
            bytes: 流式数据块
        """
        # 在流式生成器的上下文中重新设置session
        # 这样避免了跨上下文token重置的问题
        local_token = set_current_session(session)

        try:
            # 逐个产生流式数据
            async for chunk in original_generator:
                yield chunk

            # 流式传输成功完成，智能处理会话
            await self._handle_successful_request(session)
            logger.debug("流式响应传输完成，已处理数据库会话")

        except Exception as e:
            # 流式传输过程中发生异常，回滚会话
            await self._handle_failed_request(session, e)
            logger.error(f"流式响应传输失败，已回滚数据库会话: {str(e)}")
            # 重新抛出异常，让上层处理
            raise

        finally:
            # 清理：重置当前上下文的token并关闭会话
            try:
                clear_current_session(local_token)
                await self._close_session_safely(session)
                logger.debug("流式响应会话资源已清理")
            except Exception as cleanup_error:
                logger.error(f"清理流式响应会话资源时发生错误: {str(cleanup_error)}")
                # 清理失败不应该影响响应流
