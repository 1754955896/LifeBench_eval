"""
应用模块

包含业务相关的特定逻辑，如控制器注册、图结构创建、能力加载等
"""

from fastapi import FastAPI
from core.di.utils import get_beans_by_type, get_bean_by_type
from core.capability.app_capability import ApplicationCapability
from core.observation.logger import get_logger
from core.interface.controller.base_controller import BaseController
from core.middleware.user_context_middleware import UserContextMiddleware
from core.middleware.app_context_middleware import AppContextMiddleware
from fastapi.middleware import Middleware

from base_app import create_base_app
from core.lifespan.lifespan_factory import LifespanFactory


# 推荐用法：模块顶部获取一次logger，后续直接使用（高性能）
logger = get_logger(__name__)


def register_controllers(fastapi_app: FastAPI):
    """
    注册所有控制器到FastAPI应用

    Args:
        fastapi_app (FastAPI): FastAPI应用实例
    """
    all_controllers = get_beans_by_type(BaseController)
    for controller in all_controllers:
        controller.register_to_app(fastapi_app)
    logger.info("控制器注册完成，共注册 %d 个控制器", len(all_controllers))


def create_graphs(checkpointer):
    """
    创建所有业务图结构

    Args:
        checkpointer: 检查点保存器

    Returns:
        dict: 包含所有图结构的字典
    """
    logger.info("正在创建业务图结构...")

    graphs = {}

    logger.info("业务图结构创建完成，共创建 %d 个图", len(graphs))
    return graphs


def register_capabilities(fastapi_app: FastAPI):
    """
    注册所有应用能力

    Args:
        fastapi_app (FastAPI): FastAPI应用实例
    """
    capability_beans = get_beans_by_type(ApplicationCapability)
    for capability in capability_beans:
        capability.enable(fastapi_app)
    logger.info("应用能力注册完成，共注册 %d 个能力", len(capability_beans))


def register_graphs(fastapi_app: FastAPI):
    """
    注册所有图结构到FastAPI应用

    Args:
        fastapi_app (FastAPI): FastAPI应用实例
    """
    checkpointer = fastapi_app.state.checkpointer
    graphs = create_graphs(checkpointer)
    fastapi_app.state.graphs = graphs
    logger.info("图结构注册完成，共注册 %d 个图", len(graphs))


# 注意：create_business_lifespan 现在从 core.lifespan 导入
# 这里保留原有的注册函数供新的业务组件使用


def create_business_app(
    cors_origins=None,
    cors_allow_credentials=True,
    cors_allow_methods=None,
    cors_allow_headers=None,
):
    """
    创建包含业务逻辑的完整应用

    Args:
        cors_origins (list[str], optional): CORS允许的源列表
        cors_allow_credentials (bool): 是否允许凭据
        cors_allow_methods (list[str], optional): 允许的HTTP方法
        cors_allow_headers (list[str], optional): 允许的HTTP头

    Returns:
        FastAPI: 配置好的FastAPI应用实例
    """
    # 使用DI获取lifespan工厂，自动创建包含所有提供者的生命周期
    lifespan_factory = get_bean_by_type(LifespanFactory)
    combined_lifespan = lifespan_factory.create_auto_lifespan()

    # 创建基础应用，传入组合的生命周期管理器
    fastapi_app = create_base_app(
        cors_origins=cors_origins,
        cors_allow_credentials=cors_allow_credentials,
        cors_allow_methods=cors_allow_methods,
        cors_allow_headers=cors_allow_headers,
        lifespan_context=combined_lifespan,
    )

    # 添加业务相关的中间件
    fastapi_app.user_middleware.append(Middleware(AppContextMiddleware))
    # 不直接对接用户
    # fastapi_app.user_middleware.append(Middleware(UserContextMiddleware))

    return fastapi_app


# 创建默认的业务应用实例
app = create_business_app()
