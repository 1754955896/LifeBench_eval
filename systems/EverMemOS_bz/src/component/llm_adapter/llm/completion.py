from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel
from component.llm_adapter.llm.message import ChatMessage


@dataclass
class ChatCompletionRequest:
    """聊天完成请求数据类"""

    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    thinking_budget: Optional[int] = None  # 添加thinking_budget参数支持
    stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = {
            "messages": [msg.to_dict() for msg in self.messages],
            "stream": self.stream,
        }

        # 只添加非None的字段
        for field_name in [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "thinking_budget",
        ]:
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = value

        return data


class ChatCompletionResponse(BaseModel, extra="allow"):
    """聊天完成响应数据类，兼容多余字段"""

    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatCompletionResponse':
        """从字典创建响应对象"""
        return cls(**data)
