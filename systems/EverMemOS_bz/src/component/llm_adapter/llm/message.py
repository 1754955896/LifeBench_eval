from enum import Enum
from dataclasses import dataclass
from typing import Dict


class MessageRole(Enum):
    """消息角色枚举"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """聊天消息数据类"""

    role: MessageRole
    content: str

    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {"role": self.role.value, "content": self.content}
