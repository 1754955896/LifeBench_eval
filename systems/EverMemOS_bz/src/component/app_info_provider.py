from typing import Dict, Any
from abc import ABC, abstractmethod
import uuid
from fastapi import Request

from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


class AppInfoProvider(ABC):
    """应用信息提供者接口，负责从请求中提取应用级别的上下文信息"""

    @abstractmethod
    async def get_context_data_from_request(self, request: Request) -> Dict[str, Any]:
        """
        从请求中提取完整的上下文数据

        Args:
            request: FastAPI请求对象

        Returns:
            Dict[str, Any]: 包含所有上下文数据的字典
        """
        raise NotImplementedError


@component(name="app_info_provider")
class AppInfoProviderImpl(AppInfoProvider):
    """应用信息提供者实现，负责从请求中提取应用级别的上下文信息"""

    async def get_context_data_from_request(self, request: Request) -> Dict[str, Any]:
        """
        从请求中提取完整的上下文数据

        Args:
            request: FastAPI请求对象

        Returns:
            Dict[str, Any]: 包含所有上下文数据的字典，包括trace_id
        """
        # 创建新的app_info字典
        app_info = {}

        # 从请求头中获取trace_id，如果不存在则生成新的
        trace_id = request.headers.get('x-trace-id')
        if not trace_id:
            trace_id = str(uuid.uuid4())

        # 设置trace_id到app_info中
        app_info['trace_id'] = trace_id

        return app_info
