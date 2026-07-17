"""
Agentic Retrieval 工具函数

提供 LLM 引导的多轮检索所需的工具：
1. Sufficiency Check: 判断检索结果是否充分
2. Multi-Query Generation: 生成多个互补的改进查询
3. Document Formatting: 格式化文档供 LLM 使用
"""

import json
import asyncio
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ==================== Prompt 模板 ====================

SUFFICIENCY_CHECK_PROMPT = """你是一个记忆检索评估专家。请评估当前检索到的记忆是否足以回答用户的查询。

用户查询：
{query}

检索到的记忆：
{retrieved_docs}

请判断这些记忆是否足以回答用户的查询。

输出格式（JSON）：
{{
    "is_sufficient": true/false,
    "reasoning": "你的判断理由",
    "missing_information": ["缺失的信息1", "缺失的信息2"]
}}

要求：
1. 如果记忆中包含了回答查询所需的关键信息，判断为充分（true）
2. 如果缺少关键信息，判断为不充分（false），并列出缺失的信息
3. reasoning 要简洁明确
4. missing_information 只在不充分时填写，否则为空数组
"""


MULTI_QUERY_GENERATION_PROMPT = """你是一个查询优化专家。用户的原始查询未能检索到充分的信息，请生成多个互补的改进查询。

原始查询：
{original_query}

当前检索到的记忆：
{retrieved_docs}

缺失的信息：
{missing_info}

请生成 2-3 个互补的查询，帮助找到缺失的信息。这些查询应该：
1. 聚焦于不同的缺失信息点
2. 使用不同的表达方式
3. 避免与原始查询完全相同
4. 保持简洁明确

输出格式（JSON）：
{{
    "queries": [
        "改进查询1",
        "改进查询2",
        "改进查询3"
    ],
    "reasoning": "查询生成策略说明"
}}

要求：
1. queries 数组包含 2-3 个查询
2. 每个查询长度 5-200 字
3. reasoning 说明生成策略
"""


# ==================== 配置类 ====================

@dataclass
class AgenticConfig:
    """Agentic 检索配置"""
    
    # Round 1 配置
    round1_emb_top_n: int = 50  # Embedding 候选数
    round1_bm25_top_n: int = 50  # BM25 候选数
    round1_top_n: int = 20  # RRF 融合后返回数
    round1_rerank_top_n: int = 5  # Rerank 后用于 LLM 判断
    
    # LLM 配置
    llm_temperature: float = 0.0  # 判断用低温度
    llm_max_tokens: int = 500
    
    # Round 2 配置
    enable_multi_query: bool = True  # 是否启用多查询
    num_queries: int = 3  # 期望生成查询数量
    round2_per_query_top_n: int = 50  # 每个查询召回数
    
    # 融合配置
    combined_total: int = 40  # 合并后总数
    final_top_n: int = 20  # 最终返回数
    
    # Rerank 配置
    use_reranker: bool = True
    reranker_instruction: str = "根据查询与记忆的相关性进行排序"
    reranker_batch_size: int = 10
    reranker_timeout: float = 30.0
    
    # 降级策略
    fallback_on_error: bool = True  # LLM 失败时回退
    timeout: float = 60.0  # 整体超时（秒）


# ==================== 工具函数 ====================

def format_documents_for_llm(
    results: List[Tuple[Any, float]],
    max_docs: int = 10,
) -> str:
    """
    格式化检索结果供 LLM 使用
    
    Args:
        results: 检索结果列表 [(candidate, score), ...]
        max_docs: 最多包含的文档数
    
    Returns:
        格式化的文档字符串
    """
    formatted_docs = []
    
    for i, (candidate, score) in enumerate(results[:max_docs], 1):
        # 提取记忆内容
        timestamp = getattr(candidate, 'timestamp', 'N/A')
        if hasattr(timestamp, 'strftime'):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = str(timestamp)
        
        # 优先使用 episode（MemCell 的核心内容）
        content = getattr(candidate, 'episode', None)
        if not content:
            content = getattr(candidate, 'summary', None)
        if not content:
            content = getattr(candidate, 'subject', 'N/A')
        
        # 构建文档条目
        doc_entry = f"[记忆 {i}]\n"
        doc_entry += f"时间: {timestamp_str}\n"
        doc_entry += f"内容: {content}\n"
        doc_entry += f"相关性得分: {score:.4f}\n"
        
        formatted_docs.append(doc_entry)
    
    return "\n".join(formatted_docs) if formatted_docs else "无检索结果"


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    解析 LLM 返回的 JSON 响应
    
    Args:
        response: LLM 原始响应字符串
    
    Returns:
        解析后的字典
    
    Raises:
        ValueError: JSON 解析失败
    """
    try:
        # 提取 JSON 部分（可能包含额外文本）
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)
        
        return result
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {response[:500]}")
        raise ValueError(f"JSON parse error: {e}")


def parse_sufficiency_response(response: str) -> Tuple[bool, str, List[str]]:
    """
    解析充分性判断的响应
    
    Args:
        response: LLM 原始响应
    
    Returns:
        (is_sufficient, reasoning, missing_information)
    """
    try:
        result = parse_json_response(response)
        
        # 验证必需字段
        if "is_sufficient" not in result:
            raise ValueError("Missing 'is_sufficient' field")
        
        is_sufficient = bool(result["is_sufficient"])
        reasoning = result.get("reasoning", "No reasoning provided")
        missing_info = result.get("missing_information", [])
        
        # 验证类型
        if not isinstance(missing_info, list):
            missing_info = []
        
        return is_sufficient, reasoning, missing_info
    
    except Exception as e:
        logger.error(f"Failed to parse sufficiency response: {e}")
        # 保守回退：假设充分
        return True, f"Parse error: {str(e)}", []


def parse_multi_query_response(response: str, original_query: str) -> Tuple[List[str], str]:
    """
    解析多查询生成的响应
    
    Args:
        response: LLM 原始响应
        original_query: 原始查询（用于回退）
    
    Returns:
        (queries_list, reasoning)
    """
    try:
        result = parse_json_response(response)
        
        # 验证必需字段
        if "queries" not in result or not isinstance(result["queries"], list):
            raise ValueError("Missing or invalid 'queries' field")
        
        queries = result["queries"]
        reasoning = result.get("reasoning", "No reasoning provided")
        
        # 过滤和验证查询
        valid_queries = []
        for q in queries:
            if isinstance(q, str) and 5 <= len(q) <= 300:
                # 避免与原查询完全相同
                if q.lower().strip() != original_query.lower().strip():
                    valid_queries.append(q.strip())
        
        # 至少返回 1 个查询
        if not valid_queries:
            logger.warning("No valid queries generated, using original")
            return [original_query], "Fallback: used original query"
        
        # 限制最多 3 个查询
        valid_queries = valid_queries[:3]
        
        logger.info(f"Generated {len(valid_queries)} valid queries")
        return valid_queries, reasoning
    
    except Exception as e:
        logger.error(f"Failed to parse multi-query response: {e}")
        # 回退：返回原始查询
        return [original_query], f"Parse error: {str(e)}"


# ==================== 核心 LLM 工具函数 ====================

async def check_sufficiency(
    query: str,
    results: List[Tuple[Any, float]],
    llm_provider,
    max_docs: int = 5
) -> Tuple[bool, str, List[str]]:
    """
    检查检索结果是否充分
    
    使用 LLM 判断当前检索到的记忆是否足以回答用户查询。
    如果不充分，返回缺失的信息列表。
    
    Args:
        query: 用户查询
        results: 检索结果（Top N）
        llm_provider: LLM Provider (Memory Layer)
        max_docs: 最多评估的文档数
    
    Returns:
        (is_sufficient, reasoning, missing_information)
        - is_sufficient: True 表示充分，False 表示不充分
        - reasoning: LLM 的判断理由
        - missing_information: 缺失的信息列表（仅在不充分时有值）
    
    Example:
        >>> is_sufficient, reasoning, missing = await check_sufficiency(
        ...     query="用户喜欢吃什么？",
        ...     results=[(mem1, 0.92), (mem2, 0.85)],
        ...     llm_provider=llm
        ... )
        >>> print(is_sufficient)  # False
        >>> print(missing)  # ["用户的具体菜系偏好", "口味喜好"]
    """
    try:
        # 1. 格式化文档
        retrieved_docs = format_documents_for_llm(results, max_docs=max_docs)
        
        # 2. 构建 Prompt
        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            query=query,
            retrieved_docs=retrieved_docs
        )
        
        # 3. 调用 LLM
        logger.debug(f"Calling LLM for sufficiency check on query: {query[:50]}...")
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.0,  # 低温度，判断更稳定
            max_tokens=500,
        )
        
        # 4. 解析响应
        is_sufficient, reasoning, missing_info = parse_sufficiency_response(result_text)
        
        logger.info(f"Sufficiency check result: {is_sufficient}")
        logger.debug(f"Reasoning: {reasoning}")
        
        return is_sufficient, reasoning, missing_info
    
    except asyncio.TimeoutError:
        logger.error("Sufficiency check timeout")
        # 超时回退：假设充分（避免无限重试）
        return True, "Timeout: LLM took too long", []
    
    except Exception as e:
        logger.error(f"Sufficiency check failed: {e}", exc_info=True)
        # 保守回退：假设充分
        return True, f"Error: {str(e)}", []


async def generate_multi_queries(
    original_query: str,
    results: List[Tuple[Any, float]],
    missing_info: List[str],
    llm_provider,
    max_docs: int = 5,
    num_queries: int = 3
) -> Tuple[List[str], str]:
    """
    生成多个互补的改进查询
    
    基于原始查询、当前检索结果和缺失信息，生成多个互补的查询。
    这些查询用于多查询检索，帮助找到缺失的信息。
    
    Args:
        original_query: 原始查询
        results: Round 1 检索结果
        missing_info: 缺失的信息列表
        llm_provider: LLM Provider
        max_docs: 最多使用的文档数
        num_queries: 期望生成的查询数量（实际可能 1-3 个）
    
    Returns:
        (queries_list, reasoning)
        - queries_list: 生成的查询列表（1-3 个）
        - reasoning: LLM 的生成策略说明
    
    Example:
        >>> queries, reasoning = await generate_multi_queries(
        ...     original_query="用户喜欢吃什么？",
        ...     results=[(mem1, 0.9)],
        ...     missing_info=["菜系偏好", "口味"],
        ...     llm_provider=llm
        ... )
        >>> print(queries)
        ['用户最喜欢的菜系是什么？', '用户喜欢什么口味？', '用户有什么饮食习惯？']
    """
    try:
        # 1. 格式化文档和缺失信息
        retrieved_docs = format_documents_for_llm(results, max_docs=max_docs)
        missing_info_str = ", ".join(missing_info) if missing_info else "N/A"
        
        # 2. 构建 Prompt
        prompt = MULTI_QUERY_GENERATION_PROMPT.format(
            original_query=original_query,
            retrieved_docs=retrieved_docs,
            missing_info=missing_info_str
        )
        
        # 3. 调用 LLM
        logger.debug(f"Generating multi-queries for: {original_query[:50]}...")
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.4,  # 稍高温度，增加查询多样性
            max_tokens=300,
        )
        
        # 4. 解析响应
        queries, reasoning = parse_multi_query_response(result_text, original_query)
        
        logger.info(f"Generated {len(queries)} queries")
        for i, q in enumerate(queries, 1):
            logger.debug(f"  Query {i}: {q[:80]}{'...' if len(q) > 80 else ''}")
        
        return queries, reasoning
    
    except asyncio.TimeoutError:
        logger.error("Multi-query generation timeout")
        # 超时回退：使用原始查询
        return [original_query], "Timeout: used original query"
    
    except Exception as e:
        logger.error(f"Multi-query generation failed: {e}", exc_info=True)
        # 回退到原始查询
        return [original_query], f"Error: {str(e)}"

