import traceback
from datetime import timedelta
from typing import Optional, AsyncIterator, Dict, Any

from core.observation.logger import get_logger
from core.di.utils import get_bean, get_bean_by_type
from elasticsearch.helpers import async_streaming_bulk


logger = get_logger(__name__)


async def sync_episodic_memory_docs(
    batch_size: int, limit: Optional[int], days: Optional[int]
) -> None:
    """
    同步情景记忆文档到 Elasticsearch。

    Args:
        batch_size: 批处理大小
        limit: 最多处理的文档数量
        days: 仅处理最近 N 天创建的文档；为 None 处理全部
    """
    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
        EpisodicMemoryRawRepository,
    )
    from infra_layer.adapters.out.search.elasticsearch.converter.episodic_memory_converter import (
        EpisodicMemoryConverter,
    )
    from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
        EpisodicMemoryDoc,
    )

    from common_utils.datetime_utils import get_now_with_timezone

    mongo_repo = get_bean_by_type(EpisodicMemoryRawRepository)
    index_name = EpisodicMemoryDoc._index._name  # type: ignore[attr-defined]

    query_filter = {}
    if days is not None:
        now = get_now_with_timezone()
        start_time = now - timedelta(days=days)
        query_filter["created_at"] = {"$gte": start_time}
        logger.info("只处理过去 %s 天创建的文档（从 %s 开始）", days, start_time)

    logger.info("开始同步情景记忆文档到ES...")

    total_processed = 0
    success_count = 0
    error_count = 0

    # 获取 ES 异步客户端与索引名
    try:
        es_factory = get_bean("elasticsearch_client_factory")
        client_wrapper = await es_factory.get_default_client()
        async_client = client_wrapper.async_client
    except Exception as e:  # noqa: BLE001
        logger.error("获取 Elasticsearch 客户端失败: %s", e)
        raise

    async def generate_actions() -> AsyncIterator[Dict[str, Any]]:
        nonlocal total_processed
        skip = 0
        while True:
            query = mongo_repo.model.find(query_filter).sort("created_at")
            mongo_docs = await query.skip(skip).limit(batch_size).to_list()

            if not mongo_docs:
                logger.info("没有更多文档需要处理")
                break

            first_doc_time = (
                mongo_docs[0].created_at
                if hasattr(mongo_docs[0], "created_at")
                else "未知"
            )
            last_doc_time = (
                mongo_docs[-1].created_at
                if hasattr(mongo_docs[-1], "created_at")
                else "未知"
            )
            logger.info(
                "准备批量写入第 %s - %s 个文档，时间范围: %s ~ %s",
                skip + 1,
                skip + len(mongo_docs),
                first_doc_time,
                last_doc_time,
            )

            for mongo_doc in mongo_docs:
                es_doc = EpisodicMemoryConverter.from_mongo(mongo_doc)
                src = es_doc.to_dict()
                doc_id = es_doc.meta.id

                yield {
                    "retry_on_conflict": 3,
                    "_op_type": "update",
                    "_index": index_name,
                    "doc_as_upsert": True,
                    "_id": doc_id,
                    "doc": src,
                }

                total_processed += 1
                if limit and total_processed >= limit:
                    logger.info("已达到处理限制 %s，停止继续生成 actions", limit)
                    return

            skip += batch_size
            if len(mongo_docs) < batch_size:
                logger.info("已处理完所有文档")
                break

    try:
        # 使用 streaming bulk 批量 upsert
        async for ok, info in async_streaming_bulk(
            async_client, generate_actions(), chunk_size=batch_size
        ):
            if ok:
                success_count += 1
            else:
                error_count += 1
                logger.error("批量写入失败: %s", info)

        # 刷新索引
        await async_client.indices.refresh(index=index_name)

        logger.info(
            "同步完成! 总处理: %s, 成功: %s, 失败: %s",
            total_processed,
            success_count,
            error_count,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("同步过程中发生错误: %s", exc)
        traceback.print_exc()
        raise
