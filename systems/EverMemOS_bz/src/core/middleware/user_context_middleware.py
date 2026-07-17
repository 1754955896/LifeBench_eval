from core.authorize.enums import Role
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable

from core.context.context import set_current_user_info, clear_current_user_context
from component.auth_provider import AuthProvider
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)


class UserContextMiddleware(BaseHTTPMiddleware):
    """
    用户上下文中间件

    为每个 HTTP 请求提取用户信息并设置到上下文变量中，
    这样在整个请求处理过程中都可以通过context获取用户信息，
    无需显式传递request参数。
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.auth_provider = get_bean_by_type(AuthProvider)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        为每个请求设置用户上下文

        Args:
            request: FastAPI 请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: 响应对象
        """
        # 清除可能存在的用户上下文
        clear_current_user_context()

        # 设置用户上下文token
        token = None

        # 第一步：尝试获取和设置用户上下文
        try:
            # 尝试从请求中获取完整的用户数据
            # 这个方法现在会：
            # 1. 无认证数据 -> 返回匿名用户信息
            # 2. 认证失败 -> 抛出HTTPException(401)
            user_data = await self.auth_provider.get_optional_user_data_from_request(
                request
            )

            if user_data is not None:
                # 设置用户上下文（包括匿名用户）
                token = set_current_user_info(user_data)
                if user_data.get("role") == Role.ANONYMOUS.value:
                    logger.debug("已设置匿名用户上下文")
                else:
                    logger.debug(
                        "已设置用户上下文: 用户ID=%s, 角色=%s",
                        user_data.get("user_id"),
                        user_data.get("role"),
                    )
            else:
                user_data = {"user_id": None, "role": Role.ANONYMOUS.value}
                token = set_current_user_info(user_data)
                logger.debug("未获取到用户数据，设置匿名用户上下文")

        except HTTPException as e:
            # 如果是401认证失败，直接抛出，不要吞掉
            if e.status_code == 401:
                logger.debug("认证失败，直接返回401错误: %s", e.detail)
                raise e
            else:
                logger.error(
                    "设置用户上下文时发生HTTP异常: %s - %s", e.status_code, e.detail
                )
                # 其他HTTP异常不影响请求继续处理
        except Exception as e:
            logger.error("设置用户上下文时发生异常: %s", str(e))
            # 用户上下文设置失败不影响请求继续处理
            # 具体的认证检查由各个endpoint负责

        # 第二步：执行业务逻辑
        try:
            response = await call_next(request)
            return response

        except Exception as e:
            logger.error("业务逻辑处理异常: %s", str(e))
            # 业务逻辑异常，重新抛出让上层处理
            raise

        finally:
            # 清理用户上下文
            if token is not None:
                try:
                    clear_current_user_context(token)
                    logger.debug("已清理用户上下文")
                except Exception as reset_error:
                    logger.warning("清理用户上下文时发生错误: %s", str(reset_error))
