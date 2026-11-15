"""
健康检查控制器

提供系统健康状态检查接口
"""

from datetime import datetime
from typing import Dict, Any
from core.interface.controller.base_controller import BaseController, get
from core.observation.logger import get_logger
from core.di.decorators import component

logger = get_logger(__name__)


@component(name="healthController")
class HealthController(BaseController):
    """
    健康检查控制器

    提供系统健康状态检查功能
    """

    def __init__(self):
        super().__init__(
            prefix="/api", tags=["Health"], default_auth="none"  # 健康检查不需要认证
        )

    @get("/health", summary="健康检查", description="检查系统健康状态")
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查接口

        Returns:
            Dict[str, Any]: 健康状态信息

        Raises:
            HTTPException: 当系统不健康时抛出500错误
        """
        try:
            # 记录健康检查请求
            logger.debug("健康检查请求")

            # 返回简单的健康状态
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "message": "系统运行正常",
            }
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            # 出现异常时抛出500错误
            from fastapi import HTTPException

            raise HTTPException(
                status_code=500,
                detail={
                    "status": "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    "message": f"系统检查异常: {str(e)}",
                },
            )
