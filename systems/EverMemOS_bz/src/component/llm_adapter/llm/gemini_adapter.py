import asyncio
import time
import logging
from typing import Dict, Any, List, Union, AsyncGenerator
import os
from google.genai.client import Client
from core.di.decorators import service
from google.genai.types import (
    GenerateContentConfig,
    ContentDict,
    HarmCategory,
    HarmBlockThreshold,
)
from google.genai.types import ThinkingConfig
from component.llm_adapter.llm.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from component.llm_adapter.llm.message import MessageRole
from component.llm_adapter.llm.llm_backend_adapter import LLMBackendAdapter

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from core.constants.errors import ErrorMessage

logger = logging.getLogger(__name__)


class GeminiAdapter(LLMBackendAdapter):
    """Google Gemini API适配器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.max_retries = config.get("max_retries", 3)

        if not self.api_key:
            raise ValueError(ErrorMessage.CONFIGURATION_MISSING.value)

        # 使用新的 google.genai API
        self.client = Client(api_key=self.api_key)
        self.model_name = self.config.get("default_model", "gemini-2.5-flash")

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """执行聊天完成（转换为Gemini格式）"""
        if not request.model:
            request.model = self.model_name

        contents = self._convert_messages_to_gemini_format(request.messages)

        # 构建GenerationConfig
        generation_config_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_tokens,
        }

        # 如果提供了thinking_budget参数，创建ThinkingConfig
        thinking_config = None
        if request.thinking_budget is not None:
            thinking_config = ThinkingConfig(thinking_budget=request.thinking_budget)
            generation_config_params["thinking_config"] = thinking_config

        generation_config = GenerateContentConfig(**generation_config_params)

        for attempt in range(self.max_retries):
            try:
                if request.stream:
                    return self._stream_chat_completion(
                        contents=contents, generation_config=generation_config
                    )
                else:
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=generation_config,
                    )
                    return self._convert_gemini_response(response, request.model)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"An unexpected error occurred in GeminiAdapter: {e}"
                    ) from e
                await asyncio.sleep(2**attempt)

        raise RuntimeError(
            "Gemini chat completion request failed after multiple retries."
        )

    def _convert_messages_to_gemini_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[ContentDict]:
        """将消息列表转换为Gemini格式"""
        contents = []
        for msg in messages:
            if type(msg) == HumanMessage:
                contents.append(ContentDict(role="user", parts=[{"text": msg.content}]))
            elif type(msg) == AIMessage:
                contents.append(
                    ContentDict(role="model", parts=[{"text": msg.content}])
                )
            elif type(msg) == SystemMessage:
                contents.append(
                    ContentDict(role="model", parts=[{"text": msg.content}])
                )
            elif MessageRole(msg.role) == MessageRole.SYSTEM:
                contents.append(
                    ContentDict(role="model", parts=[{"text": msg.content}])
                )
            elif MessageRole(msg.role) == MessageRole.USER:
                contents.append(ContentDict(role="user", parts=[{"text": msg.content}]))
            elif MessageRole(msg.role) == MessageRole.ASSISTANT:
                contents.append(
                    ContentDict(role="model", parts=[{"text": msg.content}])
                )
        return contents

    def _convert_gemini_response(self, response, model: str) -> ChatCompletionResponse:
        """转换Gemini响应为OpenAI格式"""
        # Token信息由审计系统统一处理，这里不需要提取
        result = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",  # Gemini不提供ID，我们自己生成一个
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.text},
                    "finish_reason": "stop",  # Gemini API v1 不直接提供 finish_reason
                }
            ],
            usage={},  # 空的usage，token信息由审计系统处理
        )

        # 将原始Gemini响应对象附加到结果上，供审计系统使用
        result._original_gemini_response = response

        return result

    async def _stream_chat_completion(
        self, contents: List[ContentDict], generation_config: GenerateContentConfig
    ) -> AsyncGenerator[str, None]:
        """流式聊天完成"""
        response_stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name, contents=contents, config=generation_config
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self.config.get("models", [])

    async def close(self):
        """关闭客户端（Gemini库不需要）"""
        pass
