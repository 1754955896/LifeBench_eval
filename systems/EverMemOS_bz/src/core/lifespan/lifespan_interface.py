"""
FastAPI生命周期接口定义

简单的生命周期管理接口，支持顺序和name字段定义
"""

from abc import ABC, abstractmethod
from fastapi import FastAPI
from typing import Any

from core.observation.logger import get_logger

logger = get_logger(__name__)


class LifespanProvider(ABC):
    """生命周期提供者接口"""

    def __init__(self, name: str, order: int = 0):
        """
        初始化生命周期提供者

        Args:
            name (str): 提供者名称
            order (int): 执行顺序，数字越小越先执行
        """
        self.name = name
        self.order = order

    @abstractmethod
    async def startup(self, app: FastAPI) -> Any:
        """启动逻辑"""
        ...

    @abstractmethod
    async def shutdown(self, app: FastAPI) -> None:
        """关闭逻辑"""
        ...
