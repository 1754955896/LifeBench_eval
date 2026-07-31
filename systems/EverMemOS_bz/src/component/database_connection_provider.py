"""
数据库连接提供者

负责管理PostgreSQL连接池和LangGraph检查点保存器
"""

import os
from typing import Optional, Tuple
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


@component(name="database_connection_provider", primary=True)
class DatabaseConnectionProvider:
    """数据库连接提供者"""

    def __init__(self):
        """初始化数据库连接提供者"""
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("数据库连接字符串DATABASE_URL未配置")

        # 从环境变量读取时区配置，默认为上海时区
        self.timezone = os.getenv("TZ", "Asia/Shanghai")

        # 连接池配置
        self.max_size = int(os.getenv("CHECKPOINTER_DB_POOL_SIZE", "20"))

        # 初始化时不创建连接池，延迟到需要时创建
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._checkpointer: Optional[AsyncPostgresSaver] = None
        self._is_initialized = False

    async def _ensure_initialized(self):
        """确保连接池已初始化"""
        if self._is_initialized:
            return

        logger.info("正在初始化数据库连接池...")

        # 连接参数配置
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,  # 添加row_factory以匹配类型
            "options": f"-c timezone={self.timezone}",  # 设置连接时区
        }

        # 创建连接池
        self._connection_pool = AsyncConnectionPool(
            conninfo=self.database_url,
            max_size=self.max_size,
            open=False,  # 不在构造函数中打开
            kwargs=connection_kwargs,
        )

        logger.info("数据库连接池创建成功 %s", self.database_url)

        # 显式打开连接池
        await self._connection_pool.open()
        logger.info("数据库连接池初始化成功，时区设置为: %s", self.timezone)

        # 初始化checkpointer
        self._checkpointer = AsyncPostgresSaver(self._connection_pool)  # type: ignore
        await self._checkpointer.setup()
        logger.info("Checkpointer设置完成")

        self._is_initialized = True

    async def get_connection_pool(self) -> AsyncConnectionPool:
        """
        获取数据库连接池

        Returns:
            AsyncConnectionPool: 数据库连接池实例
        """
        await self._ensure_initialized()
        return self._connection_pool

    async def get_checkpointer(self) -> AsyncPostgresSaver:
        """
        获取LangGraph检查点保存器

        Returns:
            AsyncPostgresSaver: 检查点保存器实例
        """
        await self._ensure_initialized()
        return self._checkpointer

    async def get_connection_and_checkpointer(
        self,
    ) -> Tuple[AsyncConnectionPool, AsyncPostgresSaver]:
        """
        获取连接池和检查点保存器

        Returns:
            tuple: (连接池, 检查点保存器)
        """
        await self._ensure_initialized()
        return self._connection_pool, self._checkpointer

    async def close(self):
        """关闭数据库连接池"""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.info("数据库连接池已关闭")
            self._connection_pool = None
            self._checkpointer = None
            self._is_initialized = False

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._is_initialized
