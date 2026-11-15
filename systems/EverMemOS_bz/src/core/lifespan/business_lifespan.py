"""
业务生命周期提供者实现
"""

from fastapi import FastAPI
from typing import Dict, Any

from core.observation.logger import get_logger
from core.di.utils import get_beans_by_type
from core.di.decorators import component
from core.interface.controller.base_controller import BaseController
from core.capability.app_capability import ApplicationCapability
from .lifespan_interface import LifespanProvider

logger = get_logger(__name__)


@component(name="business_lifespan_provider")
class BusinessLifespanProvider(LifespanProvider):
    """业务生命周期提供者"""

    def __init__(self, name: str = "business", order: int = 20):
        """
        初始化业务生命周期提供者

        Args:
            name (str): 提供者名称
            order (int): 执行顺序，业务逻辑通常在数据库之后启动
        """
        super().__init__(name, order)

    async def startup(self, app: FastAPI) -> Dict[str, Any]:
        """
        启动业务逻辑

        Args:
            app (FastAPI): FastAPI应用实例

        Returns:
            Dict[str, Any]: 业务初始化信息
        """
        logger.info("正在初始化业务逻辑...")

        # 1. 创建业务图结构
        graphs = self._register_graphs(app)

        # 2. 注册控制器
        controllers = self._register_controllers(app)

        # 3. 注册能力
        capabilities = self._register_capabilities(app)

        logger.info("业务应用初始化完成")

        return {
            'graphs': graphs,
            'controllers': controllers,
            'capabilities': capabilities,
        }

    async def shutdown(self, app: FastAPI) -> None:
        """
        关闭业务逻辑

        Args:
            app (FastAPI): FastAPI应用实例
        """
        logger.info("正在关闭业务逻辑...")

        # 清理app.state中的业务相关属性
        if hasattr(app.state, 'graphs'):
            delattr(app.state, 'graphs')

        logger.info("业务应用关闭完成")

    def _register_controllers(self, app: FastAPI) -> list:
        """注册所有控制器"""
        all_controllers = get_beans_by_type(BaseController)
        for controller in all_controllers:
            controller.register_to_app(app)
        logger.info("控制器注册完成，共注册 %d 个控制器", len(all_controllers))
        return all_controllers

    def _register_capabilities(self, app: FastAPI) -> list:
        """注册所有应用能力"""
        capability_beans = get_beans_by_type(ApplicationCapability)
        for capability in capability_beans:
            capability.enable(app)
        logger.info("应用能力注册完成，共注册 %d 个能力", len(capability_beans))
        return capability_beans

    def _create_graphs(self, checkpointer=None) -> dict:
        """创建所有业务图结构"""
        logger.info("正在创建业务图结构...")
        graphs = {}
        # 这里可以根据具体需求创建图结构
        logger.info("业务图结构创建完成，共创建 %d 个图", len(graphs))
        return graphs

    def _register_graphs(self, app: FastAPI) -> dict:
        """注册所有图结构到FastAPI应用"""
        checkpointer = getattr(app.state, 'checkpointer', None)
        if not checkpointer:
            logger.warning("未找到checkpointer，跳过图结构创建")
            return {}

        graphs = self._create_graphs(checkpointer)
        app.state.graphs = graphs
        logger.info("图结构注册完成，共注册 %d 个图", len(graphs))
        return graphs
