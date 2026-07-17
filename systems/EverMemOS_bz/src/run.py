#!/usr/bin/env python3
"""
Memsys Main Application - 主应用启动脚本

Memsys 记忆系统的主要业务应用，包含：
- 需求提取智能体
- 大纲生成和编辑智能体
- 全文写作和编辑智能体
- 文档管理和资源处理服务
"""
import argparse
import os
import asyncio
import sys
import uvicorn
import logging

# 这里环境变量还没加载，所以不能使用get_logger
logger = logging.getLogger(__name__)

# 添加src目录到Python路径
from import_parent_dir import add_parent_path

add_parent_path(0)

# 应用信息
APP_NAME = "Memory System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "记忆系统主应用"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description=f"启动 {APP_NAME} 服务")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="服务器监听主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="服务器监听端口 (默认: 8001)"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="指定要加载的环境变量文件 (默认: .env)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="启用Mock模式 (用于测试和开发)"
    )
    parser.add_argument(
        "--longjob", type=str, help="启动指定的长任务消费者 (例如: kafka_consumer)"
    )
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    if args.longjob:
        service_name = "longjob_" + args.longjob
    else:
        service_name = "web"

    # 使用统一的环境加载工具
    from common_utils.load_env import setup_environment

    # 设置环境（Python路径和.env文件）
    setup_environment(
        load_env_file_name=args.env_file,
        check_env_var="MONGODB_HOST",
        service_name=service_name,
    )

    # 检查是否启用Mock模式：优先检查命令行参数，其次检查环境变量
    from core.di.utils import enable_mock_mode

    if args.mock or (
        os.getenv("MOCK_MODE") and os.getenv("MOCK_MODE").lower() == "true"
    ):
        enable_mock_mode()
        logger.info("🚀 启用Mock模式")
    else:
        logger.info("🚀 禁用Mock模式")

    # 显示应用启动信息
    logger.info("🚀 启动 %s v%s", APP_NAME, APP_VERSION)
    logger.info("📝 %s", APP_DESCRIPTION)
    logger.info("🌟 启动参数:")
    logger.info("  📡 Host: %s", args.host)
    logger.info("  🔌 Port: %s", args.port)
    logger.info("  📄 Env File: %s", args.env_file)
    logger.info("  🎭 Mock Mode: %s", args.mock)
    logger.info("  🔧 LongJob Mode: %s", args.longjob if args.longjob else "Disabled")

    # 执行依赖注入和异步任务设置
    from application_startup import setup_all

    # 在模块加载时就执行依赖注入和异步任务设置
    setup_all()

    # 检查是否是 LongJob 模式
    if args.longjob:
        logger.info("🔧 启动 LongJob 模式: %s", args.longjob)
        from longjob_runner import run_longjob_mode

        asyncio.run(run_longjob_mode(args.longjob))
        sys.exit(0)

    from app import app

    # 将应用信息添加到FastAPI应用中
    app.title = APP_NAME
    app.version = APP_VERSION
    app.description = APP_DESCRIPTION

    # 使用命令行参数启动服务
    try:
        uvicorn_kwargs = {"host": args.host, "port": args.port}
        uvicorn.run(app, **uvicorn_kwargs)
    except KeyboardInterrupt:
        logger.info("👋 %s 已停止", APP_NAME)
    except (OSError, RuntimeError) as e:
        logger.error("❌ %s 启动失败: %s", APP_NAME, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
