"""
授权模块

提供基于角色的授权系统，支持匿名、用户、管理员角色，
以及自定义授权策略。
"""

from .enums import Role
from .interfaces import AuthorizationStrategy, AuthorizationContext
from .strategies import (
    DefaultAuthorizationStrategy,
    RoleBasedAuthorizationStrategy,
    CustomAuthorizationStrategy,
)
from .decorators import (
    authorize,
    require_anonymous,
    require_user,
    require_admin,
    custom_authorize,
    check_and_apply_default_auth,
)

__all__ = [
    # 枚举
    'Role',
    # 接口
    'AuthorizationStrategy',
    'AuthorizationContext',
    # 策略实现
    'DefaultAuthorizationStrategy',
    'RoleBasedAuthorizationStrategy',
    'CustomAuthorizationStrategy',
    # 装饰器
    'authorize',
    'require_anonymous',
    'require_user',
    'require_admin',
    'custom_authorize',
    'check_and_apply_default_auth',
]
