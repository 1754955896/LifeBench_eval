"""
生命周期工厂

提供动态获取和创建生命周期的工厂方法
"""

from typing import List, Optional
from core.di.utils import get_beans_by_type, get_bean
from core.di.decorators import component
from .lifespan_interface import LifespanProvider
from core.observation.logger import get_logger
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = get_logger(__name__)


def create_lifespan_with_providers(providers: list[LifespanProvider]):
    """
    创建包含多个提供者的生命周期管理器

    Args:
        providers (list[LifespanProvider]): 生命周期提供者列表

    Returns:
        callable: FastAPI生命周期上下文管理器
    """
    # 按order排序
    sorted_providers = sorted(providers, key=lambda x: x.order)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """FastAPI生命周期上下文管理器"""
        lifespan_data = {}

        try:
            # 启动所有提供者
            for provider in sorted_providers:
                logger.info(
                    "启动生命周期提供者: %s (order=%d)", provider.name, provider.order
                )
                result = await provider.startup(app)
                if result is not None:
                    lifespan_data[provider.name] = result
                logger.info("生命周期提供者启动完成: %s", provider.name)

            # 将数据存储到app.state，方便获取
            app.state.lifespan_data = lifespan_data

            yield  # 应用运行期间

        finally:
            # 按逆序关闭所有提供者
            for provider in reversed(sorted_providers):
                try:
                    logger.info("关闭生命周期提供者: %s", provider.name)
                    await provider.shutdown(app)
                    logger.info("生命周期提供者关闭完成: %s", provider.name)
                except Exception as e:
                    logger.error(
                        "关闭生命周期提供者失败: %s - %s", provider.name, str(e)
                    )

    return lifespan


@component(name="lifespan_factory")
class LifespanFactory:
    """生命周期工厂"""

    def create_auto_lifespan(self):
        """
        自动创建包含所有已注册提供者的生命周期

        Returns:
            callable: FastAPI生命周期上下文管理器
        """
        providers = get_beans_by_type(LifespanProvider)
        # 按order排序
        sorted_providers = sorted(providers, key=lambda x: x.order)
        return create_lifespan_with_providers(sorted_providers)

    def create_lifespan_with_names(self, provider_names: List[str]):
        """
        根据提供者名称创建生命周期

        Args:
            provider_names (List[str]): 提供者名称列表

        Returns:
            callable: FastAPI生命周期上下文管理器
        """
        providers = []
        for name in provider_names:
            provider = get_bean(name)
            if isinstance(provider, LifespanProvider):
                providers.append(provider)

        # 按order排序
        sorted_providers = sorted(providers, key=lambda x: x.order)
        return create_lifespan_with_providers(sorted_providers)

    def create_lifespan_with_orders(self, orders: List[int]):
        """
        根据order值创建生命周期

        Args:
            orders (List[int]): order值列表

        Returns:
            callable: FastAPI生命周期上下文管理器
        """
        all_providers = get_beans_by_type(LifespanProvider)
        filtered_providers = [p for p in all_providers if p.order in orders]

        # 按order排序
        sorted_providers = sorted(filtered_providers, key=lambda x: x.order)
        return create_lifespan_with_providers(sorted_providers)

    def list_available_providers(self) -> List[LifespanProvider]:
        """
        列出所有可用的生命周期提供者

        Returns:
            List[LifespanProvider]: 提供者列表（按order排序）
        """
        providers = get_beans_by_type(LifespanProvider)
        return sorted(providers, key=lambda x: x.order)
