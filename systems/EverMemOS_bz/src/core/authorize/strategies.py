from typing import Optional, Any, Dict
from .interfaces import AuthorizationStrategy
from .enums import Role

import asyncio


class DefaultAuthorizationStrategy(AuthorizationStrategy):
    """默认授权策略"""

    async def check_permission(
        self, user_info: Optional[Dict[str, Any]], required_role: Role, **kwargs
    ) -> bool:
        """
        默认的权限检查逻辑

        Args:
            user_info: 用户信息
            required_role: 需要的角色
            **kwargs: 额外参数

        Returns:
            bool: 是否有权限
        """
        # 匿名用户只能访问匿名资源
        if required_role == Role.ANONYMOUS:
            return True

        # 如果没有用户信息，则拒绝访问
        if not user_info:
            return False

        # 检查用户角色
        user_role = user_info.get('role', Role.USER)
        user_role = Role(user_role)

        # 角色权限检查
        if required_role == Role.USER:
            return user_role in [Role.USER, Role.ADMIN]
        elif required_role == Role.ADMIN:
            return user_role == Role.ADMIN
        elif required_role == Role.SIGNATURE:
            return user_role == Role.SIGNATURE

        return False


class RoleBasedAuthorizationStrategy(AuthorizationStrategy):
    """基于角色的授权策略"""

    def __init__(self):
        # 定义角色层级关系
        self.role_hierarchy = {
            Role.ANONYMOUS: 0,
            Role.USER: 1,
            Role.ADMIN: 2,
            Role.SIGNATURE: 1,  # SIGNATURE与USER同级，可以访问需要USER权限的资源
        }

    async def check_permission(
        self, user_info: Optional[Dict[str, Any]], required_role: Role, **kwargs
    ) -> bool:
        """
        基于角色的权限检查

        Args:
            user_info: 用户信息
            required_role: 需要的角色
            **kwargs: 额外参数

        Returns:
            bool: 是否有权限
        """
        # 匿名用户只能访问匿名资源
        if required_role == Role.ANONYMOUS:
            return True

        # 如果没有用户信息，则拒绝访问
        if not user_info:
            return False

        # 获取用户角色
        user_role_str = user_info.get('role', Role.USER.value)
        try:
            user_role = Role(user_role_str)
        except ValueError:
            # 如果角色无效，默认为USER
            user_role = Role.USER

        # 检查角色层级
        required_level = self.role_hierarchy.get(required_role, 0)
        user_level = self.role_hierarchy.get(user_role, 0)

        return user_level >= required_level


class CustomAuthorizationStrategy(AuthorizationStrategy):
    """自定义授权策略，允许用户自定义检查逻辑"""

    def __init__(self, custom_check_func):
        """
        初始化自定义策略

        Args:
            custom_check_func: 自定义检查函数，接收user_info和required_role参数
        """
        self.custom_check_func = custom_check_func

    async def check_permission(
        self, user_info: Optional[Dict[str, Any]], required_role: Role, **kwargs
    ) -> bool:
        """
        使用自定义函数进行权限检查

        Args:
            user_info: 用户信息
            required_role: 需要的角色
            **kwargs: 额外参数

        Returns:
            bool: 是否有权限
        """
        try:
            if asyncio.iscoroutinefunction(self.custom_check_func):
                return await self.custom_check_func(user_info, required_role, **kwargs)
            else:
                return self.custom_check_func(user_info, required_role, **kwargs)
        except (ValueError, TypeError, AttributeError):
            # 如果自定义检查失败，返回False
            return False
