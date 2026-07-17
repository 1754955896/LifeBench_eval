#!/usr/bin/env python3
"""
修复历史 EpisodicMemory 文档中缺失的向量字段。

运行方式（推荐通过 bootstrap 运行，自动加载应用上下文与依赖注入）：
  python src/bootstrap.py src/scripts/data_fix/fix_episodic_memory_missing_vector.py --limit 1000 --batch 200 --concurrency 8

参数：
  --limit         最多处理的文档数量（默认 1000）
  --batch         每次从数据库拉取的文档数量（默认 200，越大越快但更占内存）
  --concurrency   并发度（默认 8）
"""

import argparse
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from agentic_layer.vectorize_service import get_vectorize_service
from common_utils.datetime_utils import from_iso_format, to_iso_format


logger = get_logger(__name__)

# 目标向量模型：不等于该模型的记录也需要重刷
TARGET_VECTOR_MODEL = "Qwen/Qwen3-Embedding-4B"


async def _fetch_candidates(
    size: int,
    created_before: Optional[Any],
    created_gte: Optional[Any],
    created_lte: Optional[Any],
) -> List[EpisodicMemory]:
    """
    查询缺失向量的情景记忆候选文档。

    返回以下两类文档：
    1) episode 不为空且 vector 不存在/为 None/为空数组 的文档
    2) vector_model 不等于目标模型（TARGET_VECTOR_MODEL） 的文档（即需要重刷）
    """
    and_filters: List[Dict[str, Any]] = [
        {"episode": {"$exists": True, "$ne": ""}},
        {
            "$or": [
                {"vector": {"$exists": False}},
                {"vector": None},
                {"vector": []},
                {"vector_model": {"$ne": TARGET_VECTOR_MODEL}},
                {"vector_model": {"$exists": False}},
                {"vector_model": None},
                {"vector_model": ""},
            ]
        },
    ]

    # created_at 过滤条件（范围 + 翻页锚点）
    created_at_filter: Dict[str, Any] = {}
    if created_gte is not None:
        created_at_filter["$gte"] = created_gte
    if created_lte is not None:
        created_at_filter["$lte"] = created_lte
    # 翻页锚点：优先处理最近创建的数据，其次按更早的数据继续翻页
    if created_before is not None:
        created_at_filter["$lt"] = created_before
    if created_at_filter:
        and_filters.append({"created_at": created_at_filter})

    query: Dict[str, Any] = {"$and": and_filters}

    cursor = EpisodicMemory.find(query).sort("-created_at").limit(size)  # 最近优先

    results = await cursor.to_list()
    return results


async def _process_one(
    document: EpisodicMemory, semaphore: asyncio.Semaphore
) -> Tuple[Optional[str], Optional[str]]:
    """
    处理单个文档：向量化 episode 并回写 vector 与 vector_model。

    返回 (doc_id, error)；成功时 error 为 None。
    """
    async with semaphore:
        try:
            if not document.episode:
                return str(document.id), "episode 为空，跳过"

            vectorize_service = get_vectorize_service()
            embedding = await vectorize_service.get_embedding(document.episode)
            vector_list = embedding.tolist()  # 与仓库逻辑保持一致
            model_name = vectorize_service.get_model_name()

            # 精确按 _id 更新，避免覆盖其他字段
            await EpisodicMemory.find({"_id": document.id}).update(
                {"$set": {"vector": vector_list, "vector_model": model_name}}
            )

            return str(document.id), None
        except Exception as exc:  # noqa: BLE001 非关键错误，记录后继续
            return str(document.id), str(exc)


async def run_fix(
    limit: int = 1000,
    batch: int = 200,
    concurrency: int = 10,
    start_created_at: Optional[Any] = None,
    end_created_at: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    执行修复任务。

    Args:
        limit:    最多处理的文档数量
        batch:    每次批量从数据库拉取的文档数量
        concurrency: 并发度（协程并发）

    Returns:
        统计信息字典
    """
    if limit <= 0:
        limit = 1
    if batch <= 0:
        batch = 1
    if concurrency <= 0:
        concurrency = 1

    semaphore = asyncio.Semaphore(concurrency)

    processed_total = 0
    succeeded = 0
    errors: List[Tuple[str, str]] = []
    created_before: Optional[Any] = None
    # 通过函数参数传入的范围过滤
    created_gte: Optional[Any] = start_created_at
    created_lte: Optional[Any] = end_created_at

    logger.info(
        "🔍 开始扫描需修复文档（limit=%d, batch=%d, concurrency=%d）",
        limit,
        batch,
        concurrency,
    )

    while processed_total < limit:
        fetch_size = min(batch, limit - processed_total)
        candidates = await _fetch_candidates(
            size=fetch_size,
            created_before=created_before,
            created_gte=created_gte,
            created_lte=created_lte,
        )

        if not candidates:
            break

        # 下一页锚点：本批次中最早的 created_at
        try:
            created_before = candidates[-1].created_at
            try:
                logger.info("⏱️ 当前处理到 created_at=%s", to_iso_format(created_before))
            except Exception:  # noqa: BLE001
                logger.info("⏱️ 当前处理到 created_at=%s", str(created_before))
        except AttributeError:
            # 如果模型无该字段或异常，退化为按 skip 逻辑（不更新锚点）
            pass

        logger.info(
            "📦 拉取到候选 %d 条（已累计处理=%d/%d）",
            len(candidates),
            processed_total,
            limit,
        )

        tasks: List[asyncio.Task] = []
        for doc in candidates:
            task = asyncio.create_task(_process_one(doc, semaphore))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=False)

        for doc_id, err in results:
            if err is None:
                succeeded += 1
            else:
                errors.append((doc_id or "unknown", err))

        processed_total += len(candidates)

    failed = len(errors)
    if failed:
        for doc_id, err_msg in errors[:20]:  # 避免日志过多
            logger.error("❌ 修复失败 doc=%s, error=%s", doc_id, err_msg)
        if failed > 20:
            logger.error("… 还有 %d 条错误未逐条打印", failed - 20)

    logger.info(
        "✅ 修复完成 | total=%d, succeeded=%d, failed=%d",
        processed_total,
        succeeded,
        failed,
    )
    return {
        "total": processed_total,
        "succeeded": succeeded,
        "failed": failed,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修复历史 EpisodicMemory 缺失向量数据")
    parser.add_argument(
        "--limit", type=int, default=1000, help="最多处理的文档数量（默认 1000）"
    )
    parser.add_argument(
        "--batch", type=int, default=200, help="每次从数据库拉取的文档数量（默认 200）"
    )
    parser.add_argument("--concurrency", type=int, default=8, help="并发度（默认 8）")
    parser.add_argument(
        "--start-created-at",
        dest="start_created_at",
        type=str,
        default=None,
        help="仅处理 created_at 大于等于该时间的文档（ISO格式，例如 2025-09-16T20:20:06+08:00）",
    )
    parser.add_argument(
        "--end-created-at",
        dest="end_created_at",
        type=str,
        default=None,
        help="仅处理 created_at 小于等于该时间的文档（ISO格式，例如 2025-09-30T23:59:59+08:00）",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    # 通过 bootstrap 运行时，应用上下文已加载；此处直接执行异步任务
    # 解析时间范围参数（ISO -> 带时区 datetime）
    start_dt = from_iso_format(args.start_created_at) if args.start_created_at else None
    end_dt = from_iso_format(args.end_created_at) if args.end_created_at else None

    if start_dt or end_dt:
        try:
            start_str = to_iso_format(start_dt) if start_dt else "(未指定)"
            end_str = to_iso_format(end_dt) if end_dt else "(未指定)"
        except Exception:  # noqa: BLE001
            start_str = str(start_dt) if start_dt else "(未指定)"
            end_str = str(end_dt) if end_dt else "(未指定)"
        logger.info("⛳ 使用 created_at 过滤范围: [%s, %s]", start_str, end_str)

    asyncio.run(
        run_fix(
            limit=args.limit,
            batch=args.batch,
            concurrency=args.concurrency,
            start_created_at=start_dt,
            end_created_at=end_dt,
        )
    )


if __name__ == "__main__":
    main()
