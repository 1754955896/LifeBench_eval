"""
LLM配置管理

提供简单的LLM配置管理
"""

import os
from typing import Optional
from .openai_provider import OpenAIProvider


def create_provider(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs,
) -> OpenAIProvider:
    """
    创建OpenAI提供者

    Args:
        model: 模型名称
        api_key: API密钥，如果为None则使用环境变量
        base_url: Base URL，如果为None则使用默认值
        temperature: 温度
        max_tokens: 最大令牌数
        **kwargs: 其他参数

    Returns:
        配置好的OpenAIProvider实例
    """
    return OpenAIProvider(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def create_cheap_provider() -> OpenAIProvider:
    """创建便宜的提供者（使用gpt-4o-mini）"""
    return create_provider(model="gpt-4o-mini", temperature=0.3, max_tokens=1024)


def create_high_quality_provider() -> OpenAIProvider:
    """创建高质量提供者（使用gpt-4o）"""
    return create_provider(model="gpt-4o", temperature=0.7, max_tokens=4096)
