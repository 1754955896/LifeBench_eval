"""
记忆范围枚举

定义记忆检索时的范围类型
"""
from enum import Enum


class MemoryScope(str, Enum):
    """
    记忆范围枚举
    
    用于定义记忆检索时的范围:
    - ALL: 所有类型的记忆(个人 + 群组)
    - PERSONAL: 仅个人记忆(user_id != "")
    - GROUP: 仅群组记忆(user_id == "")
    """
    
    ALL = "all"
    PERSONAL = "personal"
    GROUP = "group"

