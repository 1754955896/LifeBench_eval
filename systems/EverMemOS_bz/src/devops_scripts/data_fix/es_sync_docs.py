import argparse
import asyncio
import traceback
from elasticsearch.dsl import AsyncDocument

from core.observation.logger import get_logger
from core.oxm.es.migration.utils import find_document_class_by_index_name


logger = get_logger(__name__)


async def run(
    index_name: str, batch_size: int, limit_: int | None, days: int | None
) -> None:
    """同步 MongoDB 数据到 Elasticsearch 指定索引。"""
    try:
        document_class: type[AsyncDocument] = find_document_class_by_index_name(
            index_name
        )
        logger.info(
            "找到文档类: %s.%s", document_class.__module__, document_class.__name__
        )

        doc_alias = getattr(getattr(document_class, "_index", object()), "_name", "")
        logger.info("索引别名: %s", doc_alias)

        if "episodic-memory" in str(doc_alias):
            from devops_scripts.data_fix.es_sync_episodic_memory_docs import (
                sync_episodic_memory_docs,
            )

            await sync_episodic_memory_docs(
                batch_size=batch_size, limit=limit_, days=days
            )
        else:
            raise ValueError(f"不支持的索引类型: {doc_alias}")
    except Exception as exc:  # noqa: BLE001
        logger.error("同步文档失败: %s", exc)
        traceback.print_exc()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="同步 MongoDB 数据到 Elasticsearch")
    parser.add_argument(
        "--index-name", "-i", required=True, help="索引别名，如: episodic-memory"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=500, help="批处理大小，默认500"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="限制处理的文档数量，默认全部"
    )
    parser.add_argument(
        "--days", "-d", type=int, default=None, help="只处理过去N天创建的文档，默认全部"
    )
    args = parser.parse_args(argv)

    asyncio.run(run(args.index_name, args.batch_size, args.limit, args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
