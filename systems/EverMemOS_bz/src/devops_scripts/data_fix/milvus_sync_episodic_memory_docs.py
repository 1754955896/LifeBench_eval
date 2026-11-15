"""
同步情景记忆文档到 Milvus

从 MongoDB 批量获取情景记忆文档，转换后批量插入到 Milvus。
注重效率，采用批量获取、批量转换、批量插入的策略。

技术实现：
- 批量从 MongoDB 读取文档（batch_size 控制）
- 使用 EpisodicMemoryMilvusConverter 进行格式转换
- 批量插入到 Milvus Collection
- 支持增量同步（基于 days 参数）
- 支持幂等操作（使用 upsert 语义）
"""

import traceback
from datetime import timedelta
from typing import Optional, List, Dict, Any

from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type


logger = get_logger(__name__)


async def sync_episodic_memory_docs(
    batch_size: int, limit: Optional[int], days: Optional[int]
) -> None:
    """
    同步情景记忆文档到 Milvus。

    实现策略：
    1. 从 MongoDB 批量获取文档（batch_size 条）
    2. 批量转换为 Milvus 实体格式
    3. 批量插入到 Milvus（使用 upsert 语义，支持幂等）
    4. 循环处理直到所有文档处理完毕

    Args:
        batch_size: 批处理大小，建议 500-1000
        limit: 最多处理的文档数量，None 表示处理全部
        days: 仅处理最近 N 天创建的文档，None 表示处理全部
    """
    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
        EpisodicMemoryRawRepository,
    )
    from infra_layer.adapters.out.search.milvus.converter.episodic_memory_milvus_converter import (
        EpisodicMemoryMilvusConverter,
    )
    from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
        EpisodicMemoryCollection,
    )
    from common_utils.datetime_utils import get_now_with_timezone

    # 获取 MongoDB Repository
    mongo_repo = get_bean_by_type(EpisodicMemoryRawRepository)

    # 构建查询过滤条件
    query_filter: Dict[str, Any] = {}
    if days is not None:
        now = get_now_with_timezone()
        start_time = now - timedelta(days=days)
        query_filter["created_at"] = {"$gte": start_time}
        logger.info("只处理过去 %s 天创建的文档（从 %s 开始）", days, start_time)

    logger.info("开始同步情景记忆文档到 Milvus...")

    # 统计计数器
    total_processed = 0
    success_count = 0
    error_count = 0

    # 获取 Milvus Collection
    try:
        # 直接使用 EpisodicMemoryCollection 的 async_collection() 方法
        collection = EpisodicMemoryCollection.async_collection()
        collection_name = collection.collection.name
        logger.info("使用 Milvus Collection: %s", collection_name)
    except Exception as e:  # noqa: BLE001
        logger.error("获取 Milvus Collection 失败: %s", e)
        raise

    # 批量处理主循环
    try:
        skip = 0
        while True:
            # 从 MongoDB 批量获取文档
            query = mongo_repo.model.find(query_filter).sort("created_at")
            mongo_docs = await query.skip(skip).limit(batch_size).to_list()

            if not mongo_docs:
                logger.info("没有更多文档需要处理")
                break

            # 记录当前批次的时间范围
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

            # 批量转换为 Milvus 实体
            milvus_entities: List[Dict[str, Any]] = []
            batch_errors = 0

            for mongo_doc in mongo_docs:
                try:
                    # 转换单个文档
                    milvus_entity = EpisodicMemoryMilvusConverter.from_mongo(mongo_doc)

                    # 验证必要字段
                    if not milvus_entity.get("id"):
                        logger.warning("文档缺少 id 字段，跳过: %s", mongo_doc.id)
                        batch_errors += 1
                        continue

                    if not milvus_entity.get("vector"):
                        logger.warning(
                            "文档缺少 vector 字段，跳过: id=%s", milvus_entity.get("id")
                        )
                        batch_errors += 1
                        continue

                    milvus_entities.append(milvus_entity)

                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "转换文档失败: id=%s, error=%s",
                        getattr(mongo_doc, 'id', 'unknown'),
                        e,
                    )
                    batch_errors += 1
                    continue

            # 批量插入到 Milvus
            if milvus_entities:
                try:
                    # Milvus insert 方法接受列表格式的数据
                    # 需要将实体字典列表转换为列表的列表格式
                    insert_data = milvus_entities

                    _ = await collection.insert(insert_data)

                    # 统计成功数量
                    inserted_count = len(milvus_entities)
                    success_count += inserted_count
                    logger.info("批量插入成功: %d 条记录", inserted_count)

                except Exception as e:  # noqa: BLE001
                    logger.error("批量插入 Milvus 失败: %s", e)
                    traceback.print_exc()
                    error_count += len(milvus_entities)

            # 更新统计
            total_processed += len(mongo_docs)
            error_count += batch_errors

            # 检查是否达到限制
            if limit and total_processed >= limit:
                logger.info("已达到处理限制 %s，停止处理", limit)
                break

            # 继续下一批
            skip += batch_size
            if len(mongo_docs) < batch_size:
                logger.info("已处理完所有文档")
                break

        # 刷新 Collection 以确保数据持久化
        try:
            await collection.flush()
            logger.info("Milvus Collection 刷新完成")
        except Exception as e:  # noqa: BLE001
            logger.warning("Milvus Collection 刷新失败: %s", e)

        # 输出统计信息
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
