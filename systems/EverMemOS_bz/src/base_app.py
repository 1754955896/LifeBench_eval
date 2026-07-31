"""
基础应用模块

包含业务无关的FastAPI基础配置，如CORS、中间件、生命周期管理等
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.observation.logger import get_logger
from core.middleware.database_session_middleware import DatabaseSessionMiddleware
from core.middleware.global_exception_handler import global_exception_handler
from core.middleware.profile_middleware import ProfileMiddleware
from core.di.utils import get_bean_by_type
from component.database_connection_provider import DatabaseConnectionProvider

from core.lifespan.lifespan_factory import LifespanFactory

# 推荐用法：模块顶部获取一次logger，后续直接使用（高性能）
logger = get_logger(__name__)


def create_base_app(
    cors_origins=None,
    cors_allow_credentials=True,
    cors_allow_methods=None,
    cors_allow_headers=None,
    lifespan_context=None,
):
    """
    创建基础FastAPI应用

    Args:
        cors_origins (list[str], optional): CORS允许的源列表，默认为["*"]
        cors_allow_credentials (bool): 是否允许凭据，默认为True
        cors_allow_methods (list[str], optional): 允许的HTTP方法，默认为["*"]
        cors_allow_headers (list[str], optional): 允许的HTTP头，默认为["*"]
        lifespan_context (callable, optional): 生命周期上下文管理器，默认使用数据库生命周期

    Returns:
        FastAPI: 配置好的FastAPI应用实例
    """
    # 使用传入的生命周期管理器或默认的数据库生命周期
    if lifespan_context is None:
        lifespan_factory = get_bean_by_type(LifespanFactory)
        lifespan_context = lifespan_factory.create_lifespan_with_names(
            ["database_lifespan_provider"]
        )

    # 根据环境变量控制docs的显示
    # 只有在开发环境(ENV=dev)时才启用docs
    env = os.environ.get('ENV', 'prod').upper()
    enable_docs = env == 'DEV'

    # 创建FastAPI应用
    app = FastAPI(
        lifespan=lifespan_context,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    if enable_docs:
        logger.info("FastAPI文档已启用 (ENV=%s)", env)
    else:
        logger.info("FastAPI文档已禁用 (ENV=%s)", env)

    # 设置CORS默认值
    if cors_origins is None:
        cors_origins = ["*"]
    if cors_allow_methods is None:
        cors_allow_methods = ["*"]
    if cors_allow_headers is None:
        cors_allow_headers = ["*"]

    # 添加HTTP异常处理器，否则HTTPException不会被global_exception_handler处理
    app.add_exception_handler(HTTPException, global_exception_handler)

    # 添加全局异常处理器
    # 在middleware外面兜底
    app.add_exception_handler(Exception, global_exception_handler)

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_allow_credentials,
        allow_methods=cors_allow_methods,
        allow_headers=cors_allow_headers,
    )

    # 添加基础中间件
    # middleware 的顺序很重要，先添加的后执行
    # app.add_middleware(DatabaseSessionMiddleware)

    # 添加性能分析中间件（最后添加，最先执行）
    app.add_middleware(ProfileMiddleware)

    # 挂载lifespan管理方法到app实例
    _mount_lifespan_methods(app)

    return app


def _mount_lifespan_methods(app: FastAPI):
    """
    将lifespan管理方法挂载到FastAPI应用实例上

    挂载后可以直接使用:
    - app.start_lifespan(): 启动lifespan
    - app.exit_lifespan(): 退出lifespan

    Args:
        app (FastAPI): FastAPI应用实例
    """
    # 存储lifespan管理器的引用
    app.lifespan_manager = None

    async def start_lifespan():
        """启动应用的lifespan上下文管理器"""
        if app.lifespan_manager is not None:
            logger.warning("Lifespan已经启动，无需重复启动")
            return app.lifespan_manager

        # 获取lifespan上下文管理器
        lifespan_context = app.router.lifespan_context

        if lifespan_context:
            # 创建上下文管理器实例
            lifespan_manager = lifespan_context(app)

            # 手动进入上下文（相当于启动）
            await lifespan_manager.__aenter__()

            # 存储管理器引用
            app.lifespan_manager = lifespan_manager

            logger.info("应用Lifespan启动完成")
            return lifespan_manager
        else:
            logger.warning("该应用没有配置lifespan")
            return None

    async def exit_lifespan():
        """退出应用的lifespan上下文管理器"""
        if app.lifespan_manager is None:
            logger.warning("Lifespan尚未启动或已经退出")
            return

        try:
            # 手动退出上下文
            await app.lifespan_manager.__aexit__(None, None, None)
            logger.info("应用Lifespan退出完成")
        except (AttributeError, RuntimeError) as e:
            logger.error("退出Lifespan时出错: %s", str(e))
        finally:
            # 清理引用
            app.lifespan_manager = None

    # 将方法挂载到app实例上
    app.start_lifespan = start_lifespan
    app.exit_lifespan = exit_lifespan


async def manually_start_lifespan(app: FastAPI):
    """
    手动启动FastAPI应用的lifespan上下文管理器

    注意：推荐使用挂载到app实例上的便捷方法：
    - app.start_lifespan(): 启动lifespan
    - app.exit_lifespan(): 退出lifespan

    这个函数用于在不启动HTTP服务器的情况下初始化应用的生命周期，
    包括数据库连接、业务图结构等。适用于脚本、测试或其他需要应用
    上下文但不需要HTTP服务的场景。

    Args:
        app (FastAPI): FastAPI应用实例

    Returns:
        context_manager: 生命周期上下文管理器实例，可用于手动退出

    Example:
        ```python
        from app import app

        # 推荐方式：直接使用挂载的方法
        await app.start_lifespan()
        # 执行需要应用上下文的操作
        # ...
        await app.exit_lifespan()

        # 或者使用传统方式
        from base_app import manually_start_lifespan
        lifespan_manager = await manually_start_lifespan(app)
        # await lifespan_manager.__aexit__(None, None, None)
        ```
    """
    # 直接调用挂载的方法
    return await app.start_lifespan()


async def close_database_connection():
    """关闭数据库连接池"""
    try:
        db_provider = get_bean_by_type(DatabaseConnectionProvider)
        await db_provider.close()
    except (AttributeError, RuntimeError) as e:
        logger.error("关闭数据库连接时出错: %s", str(e))
