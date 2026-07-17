import os
from typing import AsyncGenerator
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from core.di.decorators import component


@component(name="database_session_provider", primary=True)
class DatabaseSessionProvider:
    """数据库会话提供者"""

    def __init__(self):
        """初始化数据库会话提供者"""
        self.database_url = os.getenv("DATABASE_URL", "")

        # 从环境变量读取时区配置，默认为上海时区
        timezone = os.getenv("TZ", "Asia/Shanghai")

        # 将 postgresql:// 替换为 postgresql+asyncpg:// 以支持异步
        if self.database_url.startswith("postgresql://"):
            self.async_database_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        else:
            self.async_database_url = self.database_url

        # 创建异步引擎
        self.async_engine = create_async_engine(
            self.async_database_url,
            echo=False,  # 设置为True可以看到SQL日志
            future=True,
            pool_pre_ping=True,
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "300")),  # 5分钟回收连接
            pool_size=int(
                os.getenv("DB_POOL_SIZE", "40")
            ),  # 🔧 增加连接池大小（默认5 → 10）
            max_overflow=int(
                os.getenv("DB_MAX_OVERFLOW", "25")
            ),  # 🔧 增加最大溢出连接（默认10 → 15）
            connect_args={"server_settings": {"timezone": timezone}},
        )

        # 创建异步会话工厂
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine, class_=AsyncSession, expire_on_commit=False
        )

    def create_session(self) -> AsyncSession:
        """创建新的异步数据库会话"""
        return self.async_session_factory()

    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取异步数据库会话（上下文管理器形式）"""
        async with self.async_session_factory() as session:
            try:
                yield session
            finally:
                await session.close()
