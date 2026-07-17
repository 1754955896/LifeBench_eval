from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from .enums import Role


class AuthorizationStrategy(ABC):
    """授权策略接口"""

    @abstractmethod
    async def check_permission(
        self, user_info: Optional[Dict[str, Any]], required_role: Role, **kwargs
    ) -> bool:
        """
        检查用户权限

        Args:
            user_info: 用户信息，可能为None（匿名用户）
            required_role: 需要的角色
            **kwargs: 额外的参数

        Returns:
            bool: 是否有权限
        """
        pass


class AuthorizationContext:
    """授权上下文，包含授权检查所需的信息"""

    def __init__(
        self,
        user_info: Optional[Dict[str, Any]] = None,
        required_role: Role = Role.ANONYMOUS,
        strategy: Optional[AuthorizationStrategy] = None,
        **kwargs,
    ):
        self.user_info = user_info
        self.required_role = required_role
        self.strategy = strategy
        self.extra_kwargs = kwargs

    def need_auth(self) -> bool:
        """
        检查是否需要授权
        """
        return self.required_role != Role.ANONYMOUS
