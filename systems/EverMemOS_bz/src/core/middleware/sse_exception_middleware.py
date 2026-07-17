"""
SSE (Server-Sent Events) 异常处理中间件

提供装饰器用于将HTTP异常和其他异常转换为SSE事件格式，
确保流式响应中的异常能够以标准格式返回给客户端。

该中间件属于基础设施层，处理HTTP协议相关的技术细节。
"""

import json
import logging
from typing import Any, AsyncGenerator, Callable
from functools import wraps

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def yield_sse_data(data: Any) -> str:
    """
    将数据格式化为SSE格式

    Args:
        data: 要发送的数据

    Returns:
        str: SSE格式的数据字符串
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_exception_handler(
    func: Callable[..., AsyncGenerator[str, None]]
) -> Callable[..., AsyncGenerator[str, None]]:
    """
    SSE流异常处理装饰器

    将HTTPException和其他异常转换为SSE事件格式，确保客户端能够
    以统一的方式处理流式响应中的错误。

    异常转换规则：
    - HTTPException -> {"type": "error", "data": {"code": status_code, "message": detail}}
    - 其他异常 -> {"type": "error", "data": {"code": 500, "message": "内部服务器错误: {error}"}}

    Usage:
        @sse_exception_handler
        async def my_sse_generator() -> AsyncGenerator[str, None]:
            # 生成SSE事件
            yield yield_sse_data({"type": "message", "content": "hello"})

    Args:
        func: 返回AsyncGenerator[str, None]的异步生成器函数

    Returns:
        装饰后的异步生成器函数
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> AsyncGenerator[str, None]:
        try:
            async for event in func(*args, **kwargs):
                yield event
        except HTTPException as e:
            # 将HTTPException转换为SSE错误事件
            error_data = {
                "type": "error",
                "data": {"code": e.status_code, "message": e.detail},
            }
            logger.error(f"SSE流中发生HTTP异常: {e.status_code} - {e.detail}")
            yield yield_sse_data(error_data)
        except Exception as e:
            # 将其他异常转换为SSE错误事件
            error_data = {
                "type": "error",
                "data": {"code": 500, "message": f"内部服务器错误: {str(e)}"},
            }
            logger.error(f"SSE流中发生未知异常: {e}", exc_info=True)
            yield yield_sse_data(error_data)

    return wrapper
