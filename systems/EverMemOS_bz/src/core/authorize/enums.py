from enum import Enum


class Role(Enum):
    """用户角色枚举"""

    ANONYMOUS = "anonymous"  # 匿名用户
    USER = "user"  # 普通用户
    ADMIN = "admin"  # 超级管理员
    SIGNATURE = "signature"  # HMAC签名验证用户
