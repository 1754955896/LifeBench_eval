"""
应用上下文中间件
负责提取和设置应用级别的上下文信息，如 task_id
"""

import asyncio
from typing import Callable, Optional
from contextvars import Token

from fastapi import Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.observation.logger import get_logger
from core.context.context import clear_current_app_info, set_current_app_info
from core.di.utils import get_bean_by_type
from component.app_info_provider import AppInfoProvider

logger = get_logger(__name__)


class AppContextMiddleware(BaseHTTPMiddleware):
    """
    应用上下文中间件

    专门负责从 HTTP 请求中提取和设置应用级别的上下文信息：
    - 与数据库会话中间件分离，职责单一
    - 支持多种提取策略（请求头、查询参数、URL路径、请求体等）
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.app_info_provider = get_bean_by_type(AppInfoProvider)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        app_context_token = None

        try:
            # 设置应用级别的上下文信息
            app_context_token = await self._set_app_context(request)

            # 处理请求
            response = await call_next(request)

            return response

        except Exception as e:
            logger.error(f"应用上下文中间件处理异常: {e}")
            raise

        finally:
            # 清理应用上下文
            if app_context_token:
                try:
                    clear_current_app_info(app_context_token)
                    logger.debug("已清理应用上下文")
                except Exception as cleanup_error:
                    logger.warning(f"清理应用上下文时发生错误: {cleanup_error}")

    async def _set_app_context(self, request: Request) -> Optional[Token]:
        """
        设置应用级别的上下文信息

        Args:
            request: FastAPI 请求对象

        Returns:
            Optional[Token]: 应用上下文token
        """
        try:
            # 使用 AppInfoProvider 提取应用级别的上下文信息
            app_info_provider = get_bean_by_type(AppInfoProvider)
            context_data = await app_info_provider.get_context_data_from_request(
                request
            )

            # 设置应用上下文
            if context_data:
                token = set_current_app_info(context_data)
                return token
            else:
                return None

        except Exception as e:
            logger.error(f"设置应用上下文时发生异常: {e}")
            # 即使上下文设置失败，也不应该影响请求的正常处理
            return None
