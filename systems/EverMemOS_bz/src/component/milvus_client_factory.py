"""
Milvus 客户端工厂

基于环境变量提供 Milvus 客户端连接功能。
"""

import os
import asyncio
from typing import Optional, Dict
from hashlib import md5

from pymilvus import MilvusClient
from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


def get_milvus_config(prefix: str = "") -> dict:
    """
    基于环境变量获取 Milvus 配置

    Args:
        prefix: 环境变量前缀，例如 prefix="A" 时，将读取 "A_MILVUS_HOST" 等
               如果不提供，则读取 "MILVUS_HOST" 等

    环境变量：
    - {PREFIX_}MILVUS_HOST: Milvus 主机，默认 localhost
    - {PREFIX_}MILVUS_PORT: Milvus 端口，默认 19530
    - {PREFIX_}MILVUS_USER: 用户名（可选）
    - {PREFIX_}MILVUS_PASSWORD: 密码（可选）
    - {PREFIX_}MILVUS_DB_NAME: 数据库名称（可选）

    Returns:
        dict: 配置字典
    """

    def _env(name: str, default: Optional[str] = None) -> str:
        if prefix:
            prefix_upper = prefix.upper()
            key = f"{prefix_upper}_{name}"
        else:
            key = name
        return os.getenv(key, default) if default is not None else os.getenv(key, "")

    host = _env("MILVUS_HOST", "localhost")
    port = int(_env("MILVUS_PORT", "19530"))

    config = {
        "uri": f"http://{host}:{port}",
        "user": _env("MILVUS_USER"),
        "password": _env("MILVUS_PASSWORD"),
        "db_name": _env("MILVUS_DB_NAME"),
    }

    logger.info("获取 Milvus 配置 [prefix=%s]:", prefix or "default")
    logger.info("  URI: %s", config["uri"])
    logger.info("  认证: %s", "Basic" if config["user"] else "无")
    logger.info("  数据库: %s", config["db_name"] or "默认")

    return config


@component(name="milvus_client_factory", primary=True)
class MilvusClientFactory:
    """
    Milvus 客户端工厂

    提供基于配置的 Milvus 客户端缓存和管理功能
    """

    def __init__(self):
        """初始化 Milvus 客户端工厂"""
        self._clients: Dict[str, MilvusClient] = {}
        self._lock = asyncio.Lock()
        self._default_config = None
        logger.info("MilvusClientFactory initialized")

    def get_client(
        self, uri: str, user: str = "", password: str = "", db_name: str = "", **kwargs
    ) -> MilvusClient:
        """
        获取 Milvus 客户端实例

        Args:
            uri: Milvus 连接地址，如 "http://localhost:19530"
            user: 用户名（可选）
            password: 密码（可选）
            db_name: 数据库名称（可选）
            alias: 连接别名，默认为 "default"
            **kwargs: 其他连接参数

        Returns:
            MilvusClient: Milvus 客户端实例
        """
        alias = kwargs.get("alias", None)

        client = MilvusClient(
            uri=uri, user=user, password=password, db_name=db_name, **kwargs
        )

        # 缓存客户端
        self._clients[alias] = client
        logger.info("Milvus 客户端已创建并缓存: %s (alias=%s)", uri, alias)

        return client

    def get_default_client(self) -> MilvusClient:
        """
        获取基于环境变量配置的默认 Milvus 客户端实例

        Returns:
            MilvusClient: Milvus 客户端实例
        """
        # 获取或创建默认配置
        if self._default_config is None:
            self._default_config = get_milvus_config()

        config = self._default_config
        return self.get_client(
            uri=config["uri"],
            user=config["user"],
            password=config["password"],
            db_name=config["db_name"],
            alias="default",  # 默认客户端使用 "default" 作为缓存键
        )

    def get_named_client(self, name: str) -> MilvusClient:
        """
        按名称获取 Milvus 客户端

        约定：name 作为环境变量前缀，从 "{name}_MILVUS_XXX" 读取配置。
        例如 name="A" 时，将尝试读取 "A_MILVUS_HOST"、"A_MILVUS_PORT" 等。

        Args:
            name: 前缀名称（即环境变量前缀）

        Returns:
            MilvusClient: Milvus 客户端实例
        """
        if name.lower() == "default":
            return self.get_default_client()

        # 获取带前缀的配置
        config = get_milvus_config(prefix=name)
        logger.info("📋 加载命名 Milvus 配置[name=%s]: %s", name, config["uri"])

        return self.get_client(
            uri=config["uri"],
            user=config["user"],
            password=config["password"],
            db_name=config["db_name"],
            alias=name,  # 使用 name 作为缓存键
        )

    def close_all_clients(self):
        """关闭所有客户端连接"""
        for _, client in self._clients.items():
            try:
                client.close()
            except Exception as e:
                logger.error("关闭 Milvus 客户端时出错: %s", e)

        self._clients.clear()
        logger.info("所有 Milvus 客户端已关闭")
