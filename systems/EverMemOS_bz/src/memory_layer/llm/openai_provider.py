"""
OpenAI LLM provider implementation using OpenRouter.

This provider uses OpenRouter API to access OpenAI models.
"""

from math import log
import os
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import aiohttp
from typing import Optional
import asyncio
import random

from .protocol import LLMProvider, LLMError
from core.observation.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider using OpenRouter API.

    This provider uses OpenRouter to access OpenAI models with environment variable configuration.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 100 * 1024,
        enable_stats: bool = False,  # 新增：可选的统计功能，默认关闭
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            base_url: OpenRouter base URL (defaults to OpenRouter endpoint)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            enable_stats: Enable usage statistics accumulation (default: False)
            **kwargs: Additional arguments (ignored for now)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_stats = enable_stats  # 新增

        # Disable thinking mode if LLM_DISABLE_THINKING env var is set
        self._disable_thinking = os.getenv("LLM_DISABLE_THINKING", "").lower() in ("1", "true", "yes")

        # Use OpenRouter API key and base URL
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or "https://openrouter.ai/api/v1"

        # 新增：可选的单次调用统计（默认不开启，不影响现有使用）
        if self.enable_stats:
            self.current_call_stats = None  # 存储当前调用的统计信息

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt
            temperature: Override temperature for this request
            max_tokens: Override max tokens for this request

        Returns:
            Generated response text

        Raises:
            LLMError: If generation fails
        """
        # 使用 time.perf_counter() 获得更精确的时间测量
        start_time = time.perf_counter()
        # Prepare request data
        if os.getenv("LLM_OPENROUTER_PROVIDER", "default") != "default":
            provider_str = os.getenv('LLM_OPENROUTER_PROVIDER')
            provider_list = [p.strip() for p in provider_str.split(',')]
            openrouter_provider = {"order": provider_list, "allow_fallbacks": False}
        else:
            openrouter_provider = None
        # Prepare request data
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "provider": openrouter_provider,
            "response_format": response_format,
        }
        if extra_body:
            data["extra_body"] = extra_body

        # Check if thinking should be disabled (for models that support it)
        # 阿里云 qwen3 系列用 enable_thinking:false（thinking:{"type":"disabled"} 会 400）
        if self._disable_thinking:
            data.setdefault("extra_body", {})
            if "enable_thinking" not in data["extra_body"]:
                data["extra_body"]["enable_thinking"] = False
        # print(data)
        # print(data["extra_body"])
        # Add max_tokens if specified
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens

        # 使用异步的 aiohttp 替代同步的 urllib
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        max_retries = 5
        for retry_num in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=600)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions", json=data, headers=headers
                    ) as response:
                        chunks = []
                        async for chunk in response.content.iter_any():
                            chunks.append(chunk)
                        test = b"".join(chunks).decode()
                        response_data = json.loads(test)
                        # print(response_data)
                        # 处理错误响应
                        if response.status != 200:
                            # 代理返回的 error 可能是字符串（非 dict），加固解析
                            err = response_data.get('error', {}) if isinstance(response_data, dict) else {}
                            if isinstance(err, dict):
                                error_msg = err.get('message', f"HTTP {response.status}")
                            else:
                                error_msg = str(err) or f"HTTP {response.status}"
                            logger.error(
                                f"❌ [OpenAI-{self.model}] HTTP错误 {response.status}:"
                            )
                            logger.error(f"   💬 错误信息: {error_msg}")
                            # Debug: 429 Too Many Requests 断点调试
                            if response.status == 429:
                                logger.warning(
                                    f"429 Too Many Requests, waiting for 10 seconds"
                                )
                                await asyncio.sleep(random.randint(5, 20))

                            raise LLMError(f"HTTP Error {response.status}: {error_msg}")

                        # 使用 time.perf_counter() 获得更精确的时间测量
                        end_time = time.perf_counter()

                        # 提取finish_reason
                        finish_reason = response_data.get('choices', [{}])[0].get(
                            'finish_reason', ''
                        )
                        if finish_reason == 'stop':
                            logger.debug(
                                f"[OpenAI-{self.model}] 完成原因: {finish_reason}"
                            )
                        else:
                            logger.warning(
                                f"[OpenAI-{self.model}] 完成原因: {finish_reason}"
                            )

                        # 提取token使用信息
                        usage = response_data.get('usage', {})
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        completion_tokens = usage.get('completion_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)

                        # 打印详细的使用信息

                        logger.debug(f"[OpenAI-{self.model}] API调用完成:")
                        logger.debug(
                            f"[OpenAI-{self.model}] 耗时: {end_time - start_time:.2f}s"
                        )
                        # 如果耗时太长
                        if end_time - start_time > 180:
                            logger.warning(
                                f"[OpenAI-{self.model}] 耗时太长: {end_time - start_time:.2f}s"
                            )
                        logger.debug(
                            f"[OpenAI-{self.model}] Prompt Tokens: {prompt_tokens:,}"
                        )
                        logger.debug(
                            f"[OpenAI-{self.model}] Completion Tokens: {completion_tokens:,}"
                        )
                        logger.debug(
                            f"[OpenAI-{self.model}] 总Token数: {total_tokens:,}"
                        )

                        # 新增：记录当前调用的统计信息（如果开启统计）
                        if self.enable_stats:
                            self.current_call_stats = {
                                'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens,
                                'total_tokens': total_tokens,
                                'duration': end_time - start_time,
                                'timestamp': time.time(),
                            }

                        return response_data['choices'][0]['message']['content']

            except aiohttp.ClientError as e:
                error_time = time.perf_counter()
                logger.error("aiohttp.ClientError: %s", e)
                # logger.error(f"❌ [OpenAI-{self.model}] 请求失败:")
                logger.error(f"   ⏱️  耗时: {error_time - start_time:.2f}s")
                logger.error(f"   💬 错误信息: {str(e)}")
                logger.error(f"retry_num: {retry_num}")
                # raise LLMError(f"Request failed: {str(e)}")
                if retry_num == max_retries - 1:
                    raise LLMError(f"Request failed: {str(e)}")
            except Exception as e:
                error_time = time.perf_counter()
                logger.error("Exception: %s", e)
                logger.error(f"   ⏱️  耗时: {error_time - start_time:.2f}s")
                logger.error(f"   💬 错误信息: {str(e)}")
                logger.error(f"retry_num: {retry_num}")
                if retry_num == max_retries - 1:
                    raise LLMError(f"Request failed: {str(e)}")

    async def test_connection(self) -> bool:
        """
        Test the connection to the OpenRouter API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"🔗 [OpenAI-{self.model}] 测试API连接...")
            # Try a simple generation to test connection
            test_response = await self.generate("Hello", temperature=0.1)
            success = len(test_response) > 0
            if success:
                logger.info(f"✅ [OpenAI-{self.model}] API连接测试成功")
            else:
                logger.error(f"❌ [OpenAI-{self.model}] API连接测试失败: 空响应")
            return success
        except Exception as e:
            logger.error(f"❌ [OpenAI-{self.model}] API连接测试失败: {e}")
            return False

    def get_current_call_stats(self) -> Optional[dict]:
        if self.enable_stats:
            return self.current_call_stats
        return None

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"OpenAIProvider(model={self.model}, base_url={self.base_url})"
