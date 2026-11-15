import asyncio
import os
from typing import Dict, Any, List, Union, AsyncGenerator, Optional
from google.genai.client import Client
from google.genai.types import GenerateContentConfig, ContentDict
from google.genai.types import ThinkingConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from core.di.decorators import component
from component.config_provider import ConfigProvider
from core.constants.errors import ErrorMessage


@component(name="gemini_client", primary=True)
class GeminiClient:
    """Google Gemini API 客户端 - 直接返回原始响应"""

    def __init__(self, config_provider: ConfigProvider):
        """
        初始化Gemini客户端

        Args:
            config_provider: 配置提供者，用于加载llm_backends配置
        """
        self.config_provider = config_provider
        self._llm_config: Dict[str, Any] = self.config_provider.get_config(
            "llm_backends"
        )

        # 获取Gemini后端配置
        gemini_backends = self._llm_config.get("llm_backends", {})
        if "gemini" not in gemini_backends:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        self._config = gemini_backends["gemini"]

        # 获取API密钥，优先级：配置文件 > 环境变量
        self.api_key = self._config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.default_model = self._config.get("default_model") or self._config.get(
            "model", "gemini-2.5-flash"
        )
        self.max_retries = self._config.get("max_retries", 3)

        if not self.api_key:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        # 使用新的 google.genai API
        self.client = Client(api_key=self.api_key)

    async def generate_content(
        self,
        messages: Union[List[Dict[str, Any]], List[BaseMessage], str],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_mime_type: Optional[str] = None,
    ) -> Union[Any, AsyncGenerator[str, None]]:
        """
        生成内容 - 直接返回Gemini原始响应

        Args:
            messages: 消息列表，支持多种格式：
                     - List[Dict]: 标准消息格式 [{"role": "user", "content": "..."}]
                     - List[BaseMessage]: LangChain消息对象
                     - str: 单条文本消息
            model: 模型名称，如果为None则使用默认模型
            temperature: 温度参数
            top_p: top_p参数
            max_tokens: 最大输出token数
            thinking_budget: 思考预算（仅某些模型支持）
            stream: 是否流式输出
            tools: 工具列表，用于function calling和grounding
            response_mime_type: 响应MIME类型，如"application/json"

        Returns:
            如果stream=False，返回Gemini原始响应对象
            如果stream=True，返回异步生成器
        """
        if not model:
            model = self.default_model

        # 转换消息格式
        contents = self._convert_messages_to_gemini_format(messages)

        # 构建GenerationConfig
        generation_config_params = {"temperature": temperature, "top_p": top_p}

        if max_tokens is not None:
            generation_config_params["max_output_tokens"] = max_tokens

        # 如果提供了thinking_budget参数，创建ThinkingConfig
        if thinking_budget is not None:
            thinking_config = ThinkingConfig(thinking_budget=thinking_budget)
            generation_config_params["thinking_config"] = thinking_config

        # 支持响应MIME类型
        if response_mime_type is not None:
            generation_config_params["response_mime_type"] = response_mime_type

        # 支持工具（tools应该在config中）
        if tools is not None:
            generation_config_params["tools"] = tools

        generation_config = GenerateContentConfig(**generation_config_params)

        for attempt in range(self.max_retries):
            try:
                if stream:
                    return self._stream_generate_content(
                        model=model,
                        contents=contents,
                        generation_config=generation_config,
                    )
                else:
                    # 直接返回Gemini原始响应（tools已在config中）
                    response = await self.client.aio.models.generate_content(
                        model=model, contents=contents, config=generation_config
                    )
                    return response
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"An unexpected error occurred in GeminiClient: {e}"
                    ) from e
                await asyncio.sleep(2**attempt)

        raise RuntimeError("Gemini content generation failed after multiple retries.")

    def _convert_messages_to_gemini_format(
        self, messages: Union[List[Dict[str, Any]], List[BaseMessage], str]
    ) -> List[ContentDict]:
        """
        将消息列表转换为Gemini格式 - 兼容多种输入格式

        Args:
            messages: 支持以下格式：
                     - str: 单条文本消息，自动转为user角色
                     - List[Dict]: 标准消息格式 [{"role": "user", "content": "..."}]
                     - List[BaseMessage]: LangChain消息对象列表

        Returns:
            List[ContentDict]: Gemini API格式的消息列表
        """
        contents = []

        # 处理字符串输入
        if isinstance(messages, str):
            contents.append(ContentDict(role="user", parts=[{"text": messages}]))
            return contents

        # 处理列表输入
        if not isinstance(messages, list):
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        for msg in messages:
            # 处理LangChain消息对象
            if isinstance(msg, BaseMessage):
                if isinstance(msg, HumanMessage):
                    contents.append(
                        ContentDict(role="user", parts=[{"text": msg.content}])
                    )
                elif isinstance(msg, AIMessage):
                    contents.append(
                        ContentDict(role="model", parts=[{"text": msg.content}])
                    )
                elif isinstance(msg, SystemMessage):
                    # Gemini将system消息作为model角色处理
                    contents.append(
                        ContentDict(role="model", parts=[{"text": msg.content}])
                    )
                else:
                    # 其他类型的消息，尝试获取content属性
                    content = getattr(msg, 'content', str(msg))
                    contents.append(ContentDict(role="user", parts=[{"text": content}]))
                continue

            # 处理字典格式消息
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # 处理可能的嵌套content结构
                if isinstance(content, list):
                    # 如果content是列表，提取文本部分
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif "text" in part:
                                text_parts.append(part["text"])
                        else:
                            text_parts.append(str(part))
                    content = " ".join(text_parts)

                # 转换角色映射
                gemini_role = self._map_role_to_gemini(role)
                contents.append(
                    ContentDict(role=gemini_role, parts=[{"text": str(content)}])
                )
                continue

            # 处理其他类型，尝试转换为字符串
            try:
                # 检查是否有role和content属性
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = getattr(msg, 'role')
                    content = getattr(msg, 'content')
                    gemini_role = self._map_role_to_gemini(role)
                    contents.append(
                        ContentDict(role=gemini_role, parts=[{"text": str(content)}])
                    )
                else:
                    # 作为用户消息处理
                    contents.append(
                        ContentDict(role="user", parts=[{"text": str(msg)}])
                    )
            except Exception:
                # 最后的备用方案
                contents.append(ContentDict(role="user", parts=[{"text": str(msg)}]))

        return contents

    def _map_role_to_gemini(self, role: str) -> str:
        """
        将标准角色映射到Gemini格式

        Args:
            role: 原始角色名称

        Returns:
            str: Gemini格式的角色名称
        """
        role_lower = str(role).lower()

        if role_lower in ["user", "human"]:
            return "user"
        elif role_lower in ["assistant", "ai", "model", "bot"]:
            return "model"
        elif role_lower in ["system"]:
            # Gemini将system消息作为model角色处理
            return "model"
        else:
            # 默认作为用户消息
            return "user"

    async def _stream_generate_content(
        self,
        model: str,
        contents: List[ContentDict],
        generation_config: GenerateContentConfig,
    ) -> AsyncGenerator[str, None]:
        """流式内容生成"""
        # tools已经在generation_config中传递
        response_stream = await self.client.aio.models.generate_content_stream(
            model=model, contents=contents, config=generation_config
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def reload_config(self):
        """重新加载配置"""
        self._llm_config = self.config_provider.get_config("llm_backends")

        # 获取Gemini后端配置
        gemini_backends = self._llm_config.get("llm_backends", {})
        if "gemini" not in gemini_backends:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        self._config = gemini_backends["gemini"]

        # 更新配置
        self.api_key = self._config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.default_model = self._config.get("default_model") or self._config.get(
            "model", "gemini-2.5-flash"
        )
        self.max_retries = self._config.get("max_retries", 3)

        if not self.api_key:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        # 重新创建客户端
        self.client = Client(api_key=self.api_key)

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self._config.get("models", [])

    async def close(self):
        """关闭客户端（Gemini库不需要）"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
