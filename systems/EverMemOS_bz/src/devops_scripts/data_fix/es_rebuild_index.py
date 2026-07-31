import argparse
import asyncio
import traceback

from elasticsearch.dsl import AsyncDocument

from component.elasticsearch_client_factory import ElasticsearchClientFactory
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger
from core.oxm.es.migration.utils import find_document_class_by_index_name, rebuild_index


logger = get_logger(__name__)


async def run(index_name: str, close_old: bool, delete_old: bool) -> None:
    try:
        document_class: type[AsyncDocument] = find_document_class_by_index_name(
            index_name
        )
        logger.info(
            "找到文档类: %s.%s", document_class.__module__, document_class.__name__
        )
        logger.info(
            "索引别名: %s",
            getattr(getattr(document_class, "_index", object()), "_name", ""),
        )

        es_factory = get_bean_by_type(ElasticsearchClientFactory)
        es_connect = es_factory.get_default_connection()

        await rebuild_index(
            document_class, es_connect, close_old=close_old, delete_old=delete_old
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("重建索引失败: %s", exc)
        traceback.print_exc()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重建并切换 Elasticsearch 索引别名")
    parser.add_argument(
        "--index-name", "-i", required=True, help="索引别名，如: episodic-memory"
    )
    parser.add_argument("--close-old", "-c", action="store_true", help="是否关闭旧索引")
    parser.add_argument(
        "--delete-old", "-x", action="store_true", help="是否删除旧索引"
    )
    args = parser.parse_args(argv)

    asyncio.run(run(args.index_name, args.close_old, args.delete_old))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
