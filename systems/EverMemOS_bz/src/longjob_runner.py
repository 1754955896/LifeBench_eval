"""
LongJob 运行器 - 用于启动和管理长任务

提供了运行单个长任务的功能，包括：
- 通过 DI 查找指定的长任务
- 优雅启动和关闭
- 信号处理
- 错误处理和日志记录
"""

import asyncio
import signal
from typing import Optional

from core.di.utils import get_bean
from core.longjob.interfaces import LongJobInterface
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def run_longjob_mode(longjob_name: str):
    """
    运行指定的长任务模式

    Args:
        longjob_name: 长任务名称
    """
    logger.info("🚀 启动 LongJob 模式: %s", longjob_name)

    longjob_instance: Optional[LongJobInterface] = None

    # 异步启动应用生命周期
    try:
        from app import app

        if hasattr(app, "start_lifespan"):
            await app.start_lifespan()
            logger.info("✅ 应用lifespan启动完成")
        else:
            logger.warning("⚠️ app实例没有start_lifespan方法")
    except Exception as e:
        logger.warning(f"⚠️ 启动应用lifespan时出错: {e}")
        # 不抛出异常，继续执行

    try:
        # 尝试从 DI 容器中获取指定的长任务
        try:
            longjob_instance = get_bean(longjob_name)
            logger.info(
                "✅ 找到长任务: %s (%s)", longjob_name, type(longjob_instance).__name__
            )
        except Exception as e:
            logger.error("❌ 无法找到长任务 '%s': %s", longjob_name, str(e))
            logger.info("💡 请确保长任务已正确注册到 DI 容器中")
            return

        # 检查是否是 LongJobInterface 的实现
        if not isinstance(longjob_instance, LongJobInterface):
            logger.error("❌ '%s' 不是 LongJobInterface 的实现", longjob_name)
            logger.info("💡 长任务必须继承 LongJobInterface 或其子类")
            return

        # 设置信号处理器用于优雅关闭
        shutdown_event = asyncio.Event()

        def signal_handler(signum, _):
            logger.info("🛑 收到停止信号 (%s)，开始优雅关闭...", signum)
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 启动长任务
        logger.info("🔄 启动长任务: %s", longjob_name)
        await longjob_instance.start()

        logger.info("✅ 长任务 '%s' 已启动，正在运行...", longjob_name)
        logger.info("💡 按 Ctrl+C 停止任务")

        # 等待关闭信号
        await shutdown_event.wait()

        # 优雅关闭
        logger.info("🔄 正在关闭长任务: %s", longjob_name)
        await longjob_instance.shutdown()

        logger.info("✅ 长任务 '%s' 已成功关闭", longjob_name)

    except KeyboardInterrupt:
        logger.info("🛑 收到键盘中断，开始关闭...")
        if longjob_instance:
            try:
                await longjob_instance.shutdown()
                logger.info("✅ 长任务已关闭")
            except Exception as e:
                logger.error("❌ 关闭长任务时出错: %s", str(e))
    except Exception as e:
        logger.error("❌ 运行长任务时出错: %s", str(e), exc_info=True)
        if longjob_instance:
            try:
                await longjob_instance.shutdown()
            except Exception as shutdown_error:
                logger.error("❌ 关闭长任务时出错: %s", str(shutdown_error))
