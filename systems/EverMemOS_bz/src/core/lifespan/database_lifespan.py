"""
数据库生命周期提供者实现
"""

from fastapi import FastAPI
from typing import Tuple, Any

from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type
from core.di.decorators import component
from component.database_connection_provider import DatabaseConnectionProvider
from .lifespan_interface import LifespanProvider

logger = get_logger(__name__)


# @component(name="database_lifespan_provider")
class DatabaseLifespanProvider(LifespanProvider):
    """数据库生命周期提供者"""

    def __init__(self, name: str = "database", order: int = 10):
        """
        初始化数据库生命周期提供者

        Args:
            name (str): 提供者名称
            order (int): 执行顺序，数据库通常需要优先启动
        """
        super().__init__(name, order)
        self._db_provider = None

    async def startup(self, app: FastAPI) -> Tuple[Any, Any, Any]:
        """
        启动数据库连接

        Args:
            app (FastAPI): FastAPI应用实例

        Returns:
            Tuple[Any, Any, Any]: (connection_pool, checkpointer, db_provider)
        """
        logger.info("正在初始化数据库连接...")

        try:
            # 获取数据库连接提供者
            self._db_provider = get_bean_by_type(DatabaseConnectionProvider)

            # 获取连接池和检查点保存器
            pool, checkpointer = (
                await self._db_provider.get_connection_and_checkpointer()
            )

            # 将连接池和checkpointer存储到app.state中，供业务逻辑使用
            app.state.connection_pool = pool
            app.state.checkpointer = checkpointer
            app.state.db_provider = self._db_provider

            logger.info("数据库连接初始化完成")

            # 返回连接信息
            return pool, checkpointer, self._db_provider

        except Exception as e:
            logger.error("数据库初始化过程中出错: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        关闭数据库连接

        Args:
            app (FastAPI): FastAPI应用实例
        """
        logger.info("正在关闭数据库连接...")

        if self._db_provider:
            try:
                await self._db_provider.close()
                logger.info("数据库连接关闭完成")
            except Exception as e:
                logger.error("关闭数据库连接时出错: %s", str(e))

        # 清理app.state中的数据库相关属性
        for attr in ['connection_pool', 'checkpointer', 'db_provider']:
            if hasattr(app.state, attr):
                delattr(app.state, attr)
