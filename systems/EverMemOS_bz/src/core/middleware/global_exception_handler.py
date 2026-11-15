"""
全局异常处理器

为FastAPI应用提供统一的异常处理机制，确保所有HTTP异常
（包括中间件抛出的异常）都能被正确处理并返回给客户端。
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from core.observation.logger import get_logger
from common_utils.datetime_utils import to_iso_format, get_now_with_timezone
from core.constants.errors import ErrorCode, ErrorStatus

logger = get_logger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    全局异常处理器

    统一处理所有异常，包括HTTPException和其他异常，
    确保它们被正确格式化并返回给客户端。

    Args:
        request: FastAPI请求对象
        exc: 异常对象

    Returns:
        JSONResponse: 格式化的错误响应
    """
    # 处理HTTP异常
    if isinstance(exc, HTTPException):
        logger.warning(
            "HTTP异常: %s %s - 状态码: %d, 详情: %s",
            request.method,
            str(request.url),
            exc.status_code,
            exc.detail,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": ErrorStatus.FAILED.value,
                "code": ErrorCode.HTTP_ERROR.value,
                "message": exc.detail,
                "timestamp": to_iso_format(get_now_with_timezone()),
                "path": str(request.url.path),
            },
        )

    # 处理其他异常
    logger.error(
        "未处理异常: %s %s - 异常类型: %s, 详情: %s",
        request.method,
        str(request.url),
        type(exc).__name__,
        str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": ErrorStatus.FAILED.value,
            "code": ErrorCode.SYSTEM_ERROR.value,
            "message": "内部服务器错误",
            "timestamp": to_iso_format(get_now_with_timezone()),
            "path": str(request.url.path),
        },
    )
