#!/usr/bin/env python3
"""
Task Worker - 异步任务处理器启动脚本

异步任务处理服务，负责：
- 后台任务队列处理
- 长时间运行的异步任务
- 定时任务和延迟任务
- 任务状态管理和监控

使用方法:
    arq task.WorkerSettings

环境变量:
    REDIS_HOST: Redis主机地址 (默认: localhost)
    REDIS_PORT: Redis端口 (默认: 6379)
    REDIS_DB: Redis数据库编号 (默认: 0)
    REDIS_PASSWORD: Redis密码 (可选)
    REDIS_SSL: 是否使用SSL (默认: false)
    REDIS_USERNAME: Redis用户名 (可选)
"""

import os
import logging

from arq.connections import RedisSettings

# 应用信息
APP_NAME = "Async Task Worker"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "异步任务处理服务"

# 这里环境变量还没加载，所以不能使用get_logger
logger = logging.getLogger(__name__)

# 添加src目录到Python路径
from import_parent_dir import add_parent_path

add_parent_path(0)

# 使用统一的环境加载工具
# 设置.env文件
from common_utils.load_env import setup_environment

setup_environment(check_env_var="REDIS_HOST")

# 显示应用启动信息
logger.info("🚀 启动 %s v%s", APP_NAME, APP_VERSION)
logger.info("⚙️ %s", APP_DESCRIPTION)

# 运行主函数
# 扫描 component & task
from application_startup import setup_all

setup_all()


# Worker启动和关闭回调函数
async def startup(_ctx):
    """Worker启动时的回调函数"""
    logger.info("🔄 正在初始化异步任务Worker...")

    # 在worker启动时初始化应用上下文
    from app import app

    # 将应用信息添加到FastAPI应用中（必须在start_lifespan之前）
    app.title = APP_NAME
    app.version = APP_VERSION
    app.description = APP_DESCRIPTION

    if hasattr(app, "start_lifespan"):
        await app.start_lifespan()
        logger.info("✅ 应用lifespan启动完成")
    else:
        logger.warning("⚠️ app实例没有start_lifespan方法")

    logger.info("🎯 %s 启动完成，准备处理任务", APP_NAME)


async def shutdown(_ctx):
    """Worker关闭时的回调函数"""
    logger.info("🛑 正在关闭 %s...", APP_NAME)

    # 在worker关闭时清理应用上下文
    from app import app

    if hasattr(app, "exit_lifespan"):
        await app.exit_lifespan()
        logger.info("✅ 应用lifespan关闭完成")
    else:
        logger.warning("⚠️ app实例没有exit_lifespan方法")

    logger.info("👋 %s 已停止", APP_NAME)


from core.asynctasks.task_manager import get_task_manager


class WorkerSettings:
    functions = get_task_manager().get_worker_functions()
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        database=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD", "123456"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        username=os.getenv("REDIS_USERNAME"),
    )
    health_check_interval = 30
    max_jobs = 10
    job_timeout = 300
    keep_result = 3600


#  arq task.WorkerSettings
