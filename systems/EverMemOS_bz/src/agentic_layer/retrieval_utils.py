"""检索工具函数

提供多种检索策略的实现：
- Embedding 向量检索
- BM25 关键词检索
- RRF 融合检索
- Agentic 检索（LLM 引导的多轮检索）
"""

import re
import time
import jieba
import numpy as np
import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from core.nlp.stopwords_utils import filter_stopwords as filter_chinese_stopwords
from .vectorize_service import get_vectorize_service

logger = logging.getLogger(__name__)


def build_bm25_index(candidates):
    """构建 BM25 索引（支持中英文）"""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        return None, None, None, None
    
    # 确保 NLTK 数据已下载
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    # 提取文本并分词（支持中英文）
    tokenized_docs = []
    for mem in candidates:
        text = getattr(mem, "episode", None) or getattr(mem, "summary", "") or ""
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        if has_chinese:
            tokens = list(jieba.cut(text))
            processed_tokens = filter_chinese_stopwords(tokens)
        else:
            tokens = word_tokenize(text.lower())
            processed_tokens = [
                stemmer.stem(token)
                for token in tokens
                if token.isalpha() and len(token) >= 2 and token not in stop_words
            ]
        
        tokenized_docs.append(processed_tokens)
    
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs, stemmer, stop_words


async def search_with_bm25(
    query: str,
    bm25,
    candidates,
    stemmer,
    stop_words,
    top_k: int = 50
) -> List[Tuple]:
    """BM25 检索（支持中英文）"""
    if bm25 is None:
        return []
    
    try:
        from nltk.tokenize import word_tokenize
    except ImportError:
        return []
    
    # 分词查询（支持中英文）
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
    
    if has_chinese:
        tokens = list(jieba.cut(query))
        tokenized_query = filter_chinese_stopwords(tokens)
    else:
        tokens = word_tokenize(query.lower())
        tokenized_query = [
            stemmer.stem(token)
            for token in tokens
            if token.isalpha() and len(token) >= 2 and token not in stop_words
        ]
    
    if not tokenized_query:
        return []
    
    # 计算 BM25 分数
    scores = bm25.get_scores(tokenized_query)
    
    # 排序并返回 Top-K
    results = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    return results


def reciprocal_rank_fusion(
    results1: List[Tuple],
    results2: List[Tuple],
    k: int = 60
) -> List[Tuple]:
    """RRF 融合两个检索结果"""
    doc_rrf_scores = {}
    doc_map = {}
    
    # 处理第一个结果集
    for rank, (doc, score) in enumerate(results1, start=1):
        doc_id = doc.get('id')
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 处理第二个结果集
    for rank, (doc, score) in enumerate(results2, start=1):
        doc_id = doc.get('id')
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 转换为列表并排序
    fused_results = [
        (doc_map[doc_id], rrf_score)
        for doc_id, rrf_score in doc_rrf_scores.items()
    ]
    fused_results.sort(key=lambda x: x[1], reverse=True)
    
    return fused_results


async def lightweight_retrieval(
    query: str,
    candidates,
    emb_top_n: int = 50,
    bm25_top_n: int = 50,
    final_top_n: int = 20
) -> Tuple:
    """轻量级检索（Embedding + BM25 + RRF 融合）"""
    start_time = time.time()
    
    metadata = {
        "retrieval_mode": "lightweight",
        "emb_count": 0,
        "bm25_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }
    
    if not candidates:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    
    # 构建 BM25 索引
    bm25, tokenized_docs, stemmer, stop_words = build_bm25_index(candidates)
    
    # Embedding 检索
    emb_results = []
    try:
        vectorize_service = get_vectorize_service()
        query_vec = await vectorize_service.get_embedding(query)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm > 0:
            scores = []
            for mem in candidates:
                try:
                    doc_vec = np.array(mem.extend.get("embedding", []))
                    if len(doc_vec) > 0:
                        doc_norm = np.linalg.norm(doc_vec)
                        if doc_norm > 0:
                            sim = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
                            scores.append((mem, float(sim)))
                except:
                    continue
            
            emb_results = sorted(scores, key=lambda x: x[1], reverse=True)[:emb_top_n]
    except Exception as e:
        pass
    
    metadata["emb_count"] = len(emb_results)
    
    # BM25 检索
    bm25_results = []
    if bm25 is not None:
        bm25_results = await search_with_bm25(
            query, bm25, candidates, stemmer, stop_words, top_k=bm25_top_n
        )
    
    metadata["bm25_count"] = len(bm25_results)
    
    # RRF 融合
    if not emb_results and not bm25_results:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    elif not emb_results:
        final_results = bm25_results[:final_top_n]
    elif not bm25_results:
        final_results = emb_results[:final_top_n]
    else:
        fused_results = reciprocal_rank_fusion(emb_results, bm25_results, k=60)
        final_results = fused_results[:final_top_n]
    
    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000
    
    return final_results, metadata


def multi_rrf_fusion(
    results_list: List[List[Tuple]],
    k: int = 60
) -> List[Tuple]:
    """
    使用 RRF 融合多个查询的检索结果（多查询融合）
    
    与双路 RRF 类似，但支持融合任意数量的检索结果。
    每个结果集贡献的分数：1 / (k + rank)
    
    原理：
    - 在多个查询中都排名靠前的文档 → 分数累积高 → 最终排名靠前
    - 这是一种"投票机制"：多个查询都认为相关的文档更可能真正相关
    
    Args:
        results_list: 多个检索结果列表 [
            [(doc1, score), (doc2, score), ...],  # Query 1 结果
            [(doc3, score), (doc1, score), ...],  # Query 2 结果
            [(doc4, score), (doc2, score), ...],  # Query 3 结果
        ]
        k: RRF 常数（默认 60）
    
    Returns:
        融合后的结果 [(doc, rrf_score), ...]，按 RRF 分数降序排列
    
    Example:
        Query 1 结果: [(doc_A, 0.9), (doc_B, 0.8), (doc_C, 0.7)]
        Query 2 结果: [(doc_B, 0.88), (doc_D, 0.82), (doc_A, 0.75)]
        Query 3 结果: [(doc_A, 0.92), (doc_E, 0.85), (doc_B, 0.80)]
        
        RRF 分数计算：
        doc_A: 1/(60+1) + 1/(60+3) + 1/(60+1) = 0.0323  ← 在 Q1,Q2,Q3 都出现
        doc_B: 1/(60+2) + 1/(60+1) + 1/(60+3) = 0.0323  ← 在 Q1,Q2,Q3 都出现
        doc_C: 1/(60+3) + 0        + 0        = 0.0159  ← 只在 Q1 出现
        doc_D: 0        + 1/(60+2) + 0        = 0.0161  ← 只在 Q2 出现
        doc_E: 0        + 0        + 1/(60+2) = 0.0161  ← 只在 Q3 出现
        
        融合结果: doc_A 和 doc_B 排名最高（被多个查询认可）
    """
    if not results_list:
        return []
    
    # 如果只有一个结果集，直接返回
    if len(results_list) == 1:
        return results_list[0]
    
    # 使用文档的内存地址作为唯一标识
    doc_rrf_scores = {}  # {doc_id: rrf_score}
    doc_map = {}         # {doc_id: doc}
    
    # 遍历每个查询的检索结果
    for query_results in results_list:
        for rank, (doc, score) in enumerate(query_results, start=1):
            doc_id = id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            # 累加 RRF 分数
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 按 RRF 分数排序
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 转换回 (doc, score) 格式
    fused_results = [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]
    
    return fused_results


async def multi_query_retrieval(
    queries: List[str],
    candidates,
    emb_top_n: int = 50,
    bm25_top_n: int = 50,
    final_top_n: int = 40,
    rrf_k: int = 60
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    多查询并行检索 + RRF 融合
    
    对每个查询执行混合检索（Embedding + BM25），然后用 RRF 融合所有结果。
    这种策略可以捕获不同角度的相关信息，提升召回率。
    
    流程：
    1. 并行执行所有查询的混合检索
    2. 使用多查询 RRF 融合结果
    3. 返回 Top-N 文档
    
    Args:
        queries: 查询列表（2-3 个）
        candidates: 候选记忆列表
        emb_top_n: 每个查询的 Embedding 候选数
        bm25_top_n: 每个查询的 BM25 候选数
        final_top_n: 融合后返回的文档数
        rrf_k: RRF 参数
    
    Returns:
        (results, metadata)
        - results: 融合后的 Top-N 结果
        - metadata: 包含性能指标和统计信息
    
    Example:
        >>> queries = [
        ...     "用户最喜欢的菜系是什么？",
        ...     "用户喜欢什么口味？",
        ...     "用户有什么饮食习惯？"
        ... ]
        >>> results, metadata = await multi_query_retrieval(queries, candidates)
        >>> print(len(results))  # 40
        >>> print(metadata["num_queries"])  # 3
    """
    start_time = time.time()
    
    metadata = {
        "retrieval_mode": "multi_query",
        "num_queries": len(queries),
        "per_query_results": [],
        "total_docs_before_fusion": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }
    
    if not queries or not candidates:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    
    logger.info(f"Executing {len(queries)} queries in parallel...")
    
    # 并行执行所有查询的混合检索
    tasks = [
        lightweight_retrieval(q, candidates, emb_top_n, bm25_top_n, final_top_n)
        for q in queries
    ]
    
    multi_query_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 收集有效结果
    valid_results = []
    for i, result in enumerate(multi_query_results, 1):
        if isinstance(result, Exception):
            logger.error(f"Query {i} failed: {result}")
            continue
        
        results, query_metadata = result
        if results:
            valid_results.append(results)
            metadata["per_query_results"].append({
                "query_index": i,
                "count": len(results),
                "latency_ms": query_metadata.get("total_latency_ms", 0),
            })
            logger.debug(f"Query {i}: Retrieved {len(results)} documents")
    
    if not valid_results:
        logger.warning("All queries failed")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    
    # 统计融合前的总文档数
    metadata["total_docs_before_fusion"] = sum(len(r) for r in valid_results)
    
    # 使用多查询 RRF 融合
    logger.info(f"Fusing {len(valid_results)} query results...")
    fused_results = multi_rrf_fusion(valid_results, k=rrf_k)
    
    # 截取 Top-N
    final_results = fused_results[:final_top_n]
    
    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000
    
    logger.info(f"Multi-query retrieval: {metadata['total_docs_before_fusion']} → {len(final_results)} docs")
    
    return final_results, metadata


async def rerank_candidates(
    query: str,
    candidates: List[Tuple],
    top_n: int,
    rerank_service
) -> List[Tuple]:
    """
    对候选结果进行 Rerank
    
    使用 Rerank 服务对检索结果进行重排序，提升精度。
    
    Args:
        query: 用户查询
        candidates: 候选结果 [(doc, score), ...]
        top_n: 返回的 Top-N 数量
        rerank_service: Rerank 服务实例
    
    Returns:
        重排序后的 Top-N 结果 [(doc, new_score), ...]
    
    Note:
        - 如果 Rerank 失败，回退到原始排序
        - 使用批量处理避免 API 限流
    """
    if not candidates:
        return []
    
    try:
        logger.debug(f"Reranking {len(candidates)} candidates for query: {query[:50]}...")
        
        # 🔥 转换格式：将 [(doc, score)] 转为 rerank 服务期望的格式
        # rerank_service._rerank_all_hits 期望 List[Dict[str, Any]]
        candidates_for_rerank = []
        for idx, (doc, score) in enumerate(candidates):
            # 构建 hit 字典，包含足够的信息用于 rerank
            hit = {
                "index": idx,
                "score": score,
            }
            
            # 如果 doc 是 dict，直接合并
            if isinstance(doc, dict):
                hit.update(doc)
            else:
                # 如果 doc 是对象，提取关键字段
                hit["episode"] = getattr(doc, "episode", "")
                hit["summary"] = getattr(doc, "summary", "")
                hit["subject"] = getattr(doc, "subject", "")
                
                # 尝试提取 event_log（如果存在）
                if hasattr(doc, "event_log"):
                    event_log = doc.event_log
                    if isinstance(event_log, dict):
                        hit["event_log"] = event_log
                    elif event_log:
                        # 如果是对象，转为字典
                        hit["event_log"] = {
                            "atomic_fact": getattr(event_log, "atomic_fact", []),
                            "time": getattr(event_log, "time", ""),
                        }
            
            candidates_for_rerank.append(hit)
        
        # 调用 rerank 服务（rerank_memories 返回重排后的 hits，每个 hit 含 index + _rerank_score）
        reranked_hits = await rerank_service.rerank_memories(
            query,
            candidates_for_rerank,
        )

        # 转换格式：从 rerank 返回的格式转为 (doc, score) 格式
        if reranked_hits:
            reranked_results = []
            for hit in reranked_hits[:top_n]:
                if isinstance(hit, dict):
                    idx = hit.get("index", 0)
                    new_score = hit.get("_rerank_score", 0.0)
                else:
                    logger.warning(f"Unexpected rerank result type: {type(hit)}")
                    continue

                if 0 <= idx < len(candidates):
                    doc = candidates[idx][0]
                    reranked_results.append((doc, new_score))
            
            logger.debug(f"Rerank complete: {len(reranked_results)} results")
            return reranked_results if reranked_results else candidates[:top_n]
        else:
            logger.warning("Rerank returned empty results, using original")
            return candidates[:top_n]
    
    except Exception as e:
        logger.error(f"Rerank failed: {e}, using original ranking", exc_info=True)
        return candidates[:top_n]


async def agentic_retrieval(
    query: str,
    candidates,
    llm_provider,
    config: Optional[Any] = None,
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    Agentic 多轮检索（LLM 引导）
    
    使用 LLM 判断检索充分性，并在必要时进行多轮检索。
    
    流程：
    1. Round 1: 混合检索 → Top 20
    2. Rerank → Top 5 → LLM 判断充分性
    3. 如果充分：返回原始 Top 20
    4. 如果不充分：
       - LLM 生成多个改进查询（2-3 个）
       - Round 2: 并行检索所有查询
       - 使用 RRF 融合 → 去重合并到 40 个
       - Rerank → 返回最终 Top 20
    
    Args:
        query: 用户查询
        candidates: 候选记忆列表
        llm_provider: LLM Provider (Memory Layer)
        config: Agentic 配置（可选）
    
    Returns:
        (final_results, metadata)
        - final_results: 最终检索结果 [(doc, score), ...]
        - metadata: 包含详细的检索过程信息
    
    Example:
        >>> from agentic_layer.agentic_utils import AgenticConfig
        >>> config = AgenticConfig(use_reranker=True)
        >>> results, metadata = await agentic_retrieval(
        ...     query="用户喜欢吃什么？",
        ...     candidates=memcells,
        ...     llm_provider=llm,
        ...     config=config
        ... )
        >>> print(metadata["is_sufficient"])  # False
        >>> print(metadata["refined_queries"])  # ["用户最喜欢的菜系？", ...]
    """
    # 导入配置和工具
    from .agentic_utils import (
        AgenticConfig,
        check_sufficiency,
        generate_multi_queries
    )
    from .rerank_service import get_rerank_service
    
    # 使用默认配置或提供的配置
    if config is None:
        config = AgenticConfig()
    
    start_time = time.time()
    
    metadata = {
        "retrieval_mode": "agentic",
        "is_multi_round": False,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "missing_info": None,
        "refined_queries": None,
        "round2_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }
    
    logger.info(f"{'='*60}")
    logger.info(f"Agentic Retrieval: {query[:60]}...")
    logger.info(f"{'='*60}")
    
    # ========== Round 1: 混合检索 Top 20 ==========
    logger.info("Round 1: Hybrid search for Top 20...")
    
    try:
        round1_results, round1_metadata = await lightweight_retrieval(
            query=query,
            candidates=candidates,
            emb_top_n=config.round1_emb_top_n,
            bm25_top_n=config.round1_bm25_top_n,
            final_top_n=config.round1_top_n
        )
        
        metadata["round1_count"] = len(round1_results)
        metadata["round1_latency_ms"] = round1_metadata.get("total_latency_ms", 0)
        
        logger.info(f"Round 1: Retrieved {len(round1_results)} documents")
        
        if not round1_results:
            logger.warning("Round 1 returned no results")
            metadata["total_latency_ms"] = (time.time() - start_time) * 1000
            return [], metadata
    
    except Exception as e:
        logger.error(f"Round 1 failed: {e}")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    
    # ========== Rerank Top 20 → Top 5 用于 Sufficiency Check ==========
    if config.use_reranker:
        logger.info("Reranking Top 20 to get Top 5 for sufficiency check...")
        
        try:
            rerank_service = get_rerank_service()
            reranked_top5 = await rerank_candidates(
                query=query,
                candidates=round1_results,
                top_n=config.round1_rerank_top_n,
                rerank_service=rerank_service
            )
            
            metadata["round1_reranked_count"] = len(reranked_top5)
            logger.info(f"Rerank: Got Top {len(reranked_top5)} for sufficiency check")
        
        except Exception as e:
            logger.error(f"Rerank failed: {e}, using original Top 5")
            reranked_top5 = round1_results[:config.round1_rerank_top_n]
            metadata["round1_reranked_count"] = len(reranked_top5)
    else:
        # 不使用 reranker，直接取前 5 个
        reranked_top5 = round1_results[:config.round1_rerank_top_n]
        metadata["round1_reranked_count"] = len(reranked_top5)
        logger.info("No Rerank: Using original Top 5 for sufficiency check")
    
    if not reranked_top5:
        logger.warning("No results for sufficiency check, returning Round 1 results")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_results, metadata
    
    # ========== LLM Sufficiency Check ==========
    logger.info("LLM: Checking sufficiency on Top 5...")
    
    try:
        is_sufficient, reasoning, missing_info = await check_sufficiency(
            query=query,
            results=reranked_top5,
            llm_provider=llm_provider,
            max_docs=config.round1_rerank_top_n
        )
        
        metadata["is_sufficient"] = is_sufficient
        metadata["reasoning"] = reasoning
        metadata["missing_info"] = missing_info
        
        logger.info(f"LLM Result: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}")
        logger.info(f"LLM Reasoning: {reasoning}")
        
    except Exception as e:
        logger.error(f"Sufficiency check failed: {e}, assuming sufficient")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_results, metadata
    
    # ========== 如果充分：返回原始 Round 1 的 Top 20 ==========
    if is_sufficient:
        logger.info("Decision: Sufficient! Using Round 1 Top 20 results")
        
        final_results = round1_results
        metadata["final_count"] = len(final_results)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        
        logger.info(f"Complete: Latency {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata
    
    # ========== 如果不充分：进入 Round 2 ==========
    metadata["is_multi_round"] = True
    logger.info("Decision: Insufficient, entering Round 2")
    if missing_info:
        logger.info(f"Missing: {', '.join(missing_info)}")
    
    # ========== LLM 生成多个改进查询 ==========
    if config.enable_multi_query:
        logger.info("LLM: Generating multiple refined queries...")
        
        try:
            refined_queries, query_strategy = await generate_multi_queries(
                original_query=query,
                results=reranked_top5,
                missing_info=missing_info,
                llm_provider=llm_provider,
                max_docs=config.round1_rerank_top_n,
                num_queries=config.num_queries
            )
            
            metadata["refined_queries"] = refined_queries
            metadata["query_strategy"] = query_strategy
            metadata["num_queries"] = len(refined_queries)
            
            logger.info(f"Generated {len(refined_queries)} queries")
            for i, q in enumerate(refined_queries, 1):
                logger.debug(f"  Query {i}: {q[:80]}...")
        
        except Exception as e:
            logger.error(f"Query generation failed: {e}, using original query")
            refined_queries = [query]
            metadata["refined_queries"] = refined_queries
            metadata["num_queries"] = 1
    else:
        # 单查询模式（向后兼容）
        refined_queries = [query]
        metadata["refined_queries"] = refined_queries
        metadata["num_queries"] = 1
    
    # ========== Round 2: 并行执行多个查询检索 ==========
    logger.info(f"Round 2: Executing {len(refined_queries)} queries in parallel...")
    
    try:
        round2_results, round2_metadata = await multi_query_retrieval(
            queries=refined_queries,
            candidates=candidates,
            emb_top_n=config.round1_emb_top_n,
            bm25_top_n=config.round1_bm25_top_n,
            final_top_n=config.round2_per_query_top_n,
            rrf_k=60
        )
        
        metadata["round2_count"] = len(round2_results)
        metadata["round2_latency_ms"] = round2_metadata.get("total_latency_ms", 0)
        metadata["multi_query_total_docs"] = round2_metadata.get("total_docs_before_fusion", 0)
        
        logger.info(f"Round 2: Retrieved {len(round2_results)} unique documents")
    
    except Exception as e:
        logger.error(f"Round 2 failed: {e}, using Round 1 results")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_results, metadata
    
    # ========== 合并：确保总共 40 个文档 ==========
    logger.info("Merge: Combining Round 1 and Round 2...")
    
    # 去重：使用文档 ID 去重
    round1_ids = {id(doc) for doc, _ in round1_results}
    round2_unique = [(doc, score) for doc, score in round2_results if id(doc) not in round1_ids]
    
    # 合并：Round1 Top20 + Round2 去重后的文档（确保总数<=40）
    combined_results = round1_results.copy()
    needed_from_round2 = config.combined_total - len(combined_results)
    combined_results.extend(round2_unique[:needed_from_round2])
    
    logger.info(f"Merge: Round1={len(round1_results)}, Round2_unique={len(round2_unique[:needed_from_round2])}, Total={len(combined_results)}")
    
    # ========== Rerank 合并后的文档 ==========
    if config.use_reranker and len(combined_results) > 0:
        logger.info(f"Rerank: Reranking {len(combined_results)} documents...")
        
        try:
            rerank_service = get_rerank_service()
            final_results = await rerank_candidates(
                query=query,  # 使用原始查询进行 rerank
                candidates=combined_results,
                top_n=config.final_top_n,
                rerank_service=rerank_service
            )
            
            logger.info(f"Rerank: Final Top {len(final_results)} selected")
        
        except Exception as e:
            logger.error(f"Final rerank failed: {e}, using top {config.final_top_n}")
            final_results = combined_results[:config.final_top_n]
    else:
        # 不使用 Reranker，直接返回 Top N
        final_results = combined_results[:config.final_top_n]
        logger.info(f"No Rerank: Returning Top {len(final_results)}")
    
    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000
    
    logger.info(f"Complete: Final {len(final_results)} docs | Latency {metadata['total_latency_ms']:.0f}ms")
    logger.info(f"{'='*60}\n")
    
    return final_results, metadata

