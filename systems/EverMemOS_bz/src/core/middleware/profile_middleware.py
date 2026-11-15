"""
性能分析中间件

为 HTTP 请求提供性能分析功能，基于 pyinstrument 库。

功能特性：
1. URL 参数触发：在请求 URL 添加 ?profile=true 即可启用性能分析
2. HTML 报告：返回可视化的性能分析 HTML 报告
3. 环境变量控制：通过 PROFILING_ENABLED 环境变量启用/禁用
4. 优雅降级：未安装 pyinstrument 时自动禁用，不影响正常请求

环境变量：
- PROFILING_ENABLED: 是否启用性能分析功能（默认: false）
- PROFILING: 同 PROFILING_ENABLED（备用环境变量名）

使用方法：
1. 设置环境变量: export PROFILING_ENABLED=true
2. 安装依赖: uv add pyinstrument
3. 访问接口时添加参数: http://localhost:8000/api/endpoint?profile=true
"""

import os
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.observation.logger import get_logger

logger = get_logger(__name__)


class ProfileMiddleware(BaseHTTPMiddleware):
    """
    性能分析中间件

    当请求 URL 包含 ?profile=true 参数时，启用性能分析并返回 HTML 格式的分析报告
    """

    def __init__(self, app: ASGIApp):
        """
        初始化性能分析中间件

        Args:
            app: ASGI 应用实例
        """
        super().__init__(app)

        # 从环境变量读取是否启用性能分析
        profiling_env = os.getenv(
            'PROFILING_ENABLED', os.getenv('PROFILING', 'true')
        ).lower()
        self._profiling_enabled = profiling_env in ('true', '1', 'yes')

        # 检查 pyinstrument 是否可用
        self._profiler_available = False
        if self._profiling_enabled:
            try:
                import pyinstrument

                self._profiler_available = True
                logger.info("✅ 性能分析中间件已启用")
                logger.info(
                    "提示: 在请求 URL 中添加 ?profile=true 参数即可启用性能分析"
                )
            except ImportError:
                logger.warning("⚠️ 未安装 pyinstrument，性能分析功能将被禁用")
                logger.warning("请执行: uv add pyinstrument")
                self._profiling_enabled = False
        else:
            logger.debug(
                "性能分析功能未启用 (设置环境变量 PROFILING_ENABLED=true 以启用)"
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理 HTTP 请求并在需要时进行性能分析

        Args:
            request: FastAPI 请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: 响应对象（正常响应或性能分析报告）
        """
        # 如果功能未启用，直接通过
        if not self._profiling_enabled or not self._profiler_available:
            return await call_next(request)

        # 检查是否需要进行性能分析
        profiling = request.query_params.get("profile", "").lower() in (
            "true",
            "1",
            "yes",
        )

        if not profiling:
            # 不需要性能分析，正常处理请求
            return await call_next(request)

        # 需要性能分析
        try:
            # 动态导入（虽然已经检查过可用性，但为了类型安全还是在使用时导入）
            from pyinstrument import Profiler

            # 创建并启动 profiler
            profiler = Profiler()
            profiler.start()

            logger.info("性能分析已启动: %s %s", request.method, request.url.path)

            try:
                # 执行请求（注意：原始响应会被丢弃，用 profiler 报告替换）
                await call_next(request)
            except Exception as e:
                # 即使请求失败，也要停止 profiler 并返回性能分析报告
                logger.error("性能分析期间请求失败: %s", str(e))
                # 继续生成性能分析报告

            # 停止 profiler
            profiler.stop()

            # 生成 HTML 报告
            html_output = profiler.output_html()

            logger.info("性能分析已完成: %s %s", request.method, request.url.path)

            # 返回 HTML 格式的性能分析报告
            return HTMLResponse(content=html_output, status_code=200)

        except Exception as e:
            logger.error("性能分析过程中发生错误: %s", str(e))
            # 性能分析失败时，重新执行正常请求
            return await call_next(request)
