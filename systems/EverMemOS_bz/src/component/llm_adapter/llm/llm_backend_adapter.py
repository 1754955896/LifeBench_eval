from abc import ABC, abstractmethod
from typing import Union, AsyncGenerator, List
from component.llm_adapter.llm.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class LLMBackendAdapter(ABC):
    """LLM后端适配器抽象基类"""

    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """执行聊天完成"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        pass
