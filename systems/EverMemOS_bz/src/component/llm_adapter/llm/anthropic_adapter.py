import json
import time
import asyncio
from typing import Dict, Any, List, Union, AsyncGenerator
import os
import httpx
from core.di.decorators import service

from component.llm_adapter.llm.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from component.llm_adapter.llm.message import MessageRole
from component.llm_adapter.llm.llm_backend_adapter import LLMBackendAdapter
from core.constants.errors import ErrorMessage


class AnthropicAdapter(LLMBackendAdapter):
    """Anthropic Claude API适配器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url")
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)

        if not self.api_key or not self.base_url:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=self.timeout,
        )

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """执行聊天完成（转换为Anthropic格式）"""
        if not request.model:
            request.model = self.config.get("default_model")

        system_message = ""
        messages = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                messages.append(msg.to_dict())

        data = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,  # Anthropic requires max_tokens
            "stream": request.stream,
        }

        if system_message:
            data["system"] = system_message
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p

        # 添加thinking配置支持
        if request.thinking_budget is not None and request.thinking_budget > 0:
            # 检查模型是否支持thinking功能
            thinking_supported_models = [
                "claude-3-5-sonnet-20241022",
                "claude-3-7-sonnet-20241022",
                "claude-sonnet-4-20250514",
            ]
            if any(
                model_name in request.model for model_name in thinking_supported_models
            ):
                data["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": request.thinking_budget,
                }

        for attempt in range(self.max_retries):
            try:
                if request.stream:
                    return self._stream_chat_completion(data)
                else:
                    response = await self.client.post("/v1/messages", json=data)
                    response.raise_for_status()
                    return self._convert_anthropic_response(
                        response.json(), request.model
                    )
            except httpx.HTTPStatusError as e:
                if attempt == self.max_retries - 1 or e.response.status_code < 500:
                    raise RuntimeError(
                        f"Anthropic chat completion request failed: {e.response.text}"
                    ) from e
                await asyncio.sleep(2**attempt)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"An unexpected error occurred in AnthropicAdapter: {e}"
                    ) from e
                await asyncio.sleep(2**attempt)

    def _convert_anthropic_response(
        self, response_data: Dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """转换Anthropic响应为OpenAI格式"""
        return ChatCompletionResponse(
            id=response_data.get("id", ""),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_data.get("content", [{}])[0].get(
                            "text", ""
                        ),
                    },
                    "finish_reason": response_data.get("stop_reason"),
                }
            ],
            usage=response_data.get("usage"),
        )

    async def _stream_chat_completion(
        self, data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """流式聊天完成"""
        async with self.client.stream("POST", "/v1/messages", json=data) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    line_data = line[len("data: ") :]
                    if not line_data:
                        continue
                    try:
                        chunk = json.loads(line_data)
                        if chunk.get("type") == "content_block_delta":
                            yield chunk.get("delta", {}).get("text", "")
                    except json.JSONDecodeError:
                        continue

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self.config.get("models", [])

    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()
