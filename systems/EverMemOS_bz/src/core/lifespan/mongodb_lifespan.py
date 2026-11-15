"""
MongoDB 生命周期提供者实现
"""

from collections import defaultdict
from fastapi import FastAPI
from typing import Any
import os

from core.observation.logger import get_logger
from core.di.utils import get_bean
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider
from core.oxm.mongo.document_base import DocumentBase
from core.di.utils import get_all_subclasses
from component.mongodb_client_factory import MongoDBClientWrapper
from core.oxm.mongo.migration.manager import MigrationManager


logger = get_logger(__name__)


@component(name="mongodb_lifespan_provider")
class MongoDBLifespanProvider(LifespanProvider):
    """MongoDB 生命周期提供者"""

    def __init__(self, name: str = "mongodb", order: int = 15):
        """
        初始化 MongoDB 生命周期提供者

        Args:
            name (str): 提供者名称
            order (int): 执行顺序，MongoDB 在数据库连接之后启动
        """
        super().__init__(name, order)
        self._mongodb_factory = None
        self._mongodb_client = None

    async def startup(self, app: FastAPI) -> Any:
        """
        启动 MongoDB 连接和初始化

        Args:
            app (FastAPI): FastAPI应用实例

        Returns:
            Any: MongoDB 客户端信息
        """
        logger.info("正在初始化 MongoDB 连接...")

        try:

            # 获取 MongoDB 客户端工厂
            self._mongodb_factory = get_bean("mongodb_client_factory")

            # 获取默认客户端
            self._mongodb_client: MongoDBClientWrapper = (
                await self._mongodb_factory.get_default_client()
            )

            # 手动初始化 Beanie ODM
            all_subclasses_of_document_base = get_all_subclasses(DocumentBase)
            db_document_models = defaultdict(list)
            for subclass in all_subclasses_of_document_base:
                db_document_models[subclass.get_bind_database()].append(subclass)

            # 获取所有的DB名称
            db_names = list(db_document_models.keys())
            db_clients = {
                db_name: await self._mongodb_factory.get_named_client(db_name)
                for db_name in db_names
            }

            # 初始化 Beanie ODM
            for db_name, db_client in db_clients.items():
                await db_client.initialize_beanie(db_document_models[db_name])

            # 将 MongoDB 客户端存储到 app.state 中，供业务逻辑使用
            app.state.mongodb_clients = db_clients
            app.state.mongodb_factory = self._mongodb_factory

            logger.info("✅ MongoDB 连接初始化完成")

            # 在启动流程的最后执行可选的迁移步骤
            self._run_mongodb_migrations_if_enabled()

        except Exception as e:
            logger.error("❌ MongoDB 初始化过程中出错: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        关闭 MongoDB 连接

        Args:
            app (FastAPI): FastAPI应用实例
        """
        logger.info("正在关闭 MongoDB 连接...")

        if self._mongodb_factory:
            try:
                await self._mongodb_factory.close_all_clients()
                logger.info("✅ MongoDB 连接关闭完成")
            except Exception as e:
                logger.error("❌ 关闭 MongoDB 连接时出错: %s", str(e))

        # 清理 app.state 中的 MongoDB 相关属性
        for attr in ['mongodb_clients', 'mongodb_factory']:
            if hasattr(app.state, attr):
                delattr(app.state, attr)

    def _run_mongodb_migrations_if_enabled(self) -> None:
        """
        如果开启了启动时迁移开关，则执行 MongoDB 迁移。

        该步骤在启动流程的最后执行，避免影响核心连接与模型初始化流程。
        """
        if (
            str(os.getenv("MONGODB_RUN_MIGRATIONS_ON_STARTUP", "true")).lower()
            != "true"
        ):
            return

        distance_env = os.getenv("MONGODB_MIGRATIONS_DISTANCE")
        backward_flag = (
            str(os.getenv("MONGODB_MIGRATIONS_BACKWARD", "false")).lower() == "true"
        )
        no_tx_flag = (
            str(os.getenv("MONGODB_MIGRATIONS_NO_USE_TRANSACTION", "false")).lower()
            == "true"
        )

        try:
            migration_manager = MigrationManager(
                use_transaction=not no_tx_flag,
                distance=(
                    int(distance_env)
                    if distance_env and distance_env.isdigit()
                    else None
                ),
                backward=backward_flag,
            )
        except ValueError as e:
            logger.error("MongoDB 迁移参数无效: %s", str(e))
            return

        logger.info("检测到启动时迁移开关已开启，开始执行 MongoDB 迁移 ...")
        exit_code = migration_manager.run_migration()
        if exit_code != 0:
            logger.warning("MongoDB 迁移进程返回非零退出码: %s", exit_code)
