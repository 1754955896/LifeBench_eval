"""
同步 MongoDB 数据到 Milvus

主入口脚本，根据 Collection 名称调用相应的同步实现。
支持命令行参数配置批量大小、处理限制和时间范围。

运行方式（推荐通过 bootstrap 运行，自动加载应用上下文与依赖注入）：
  python src/bootstrap.py src/devops_scripts/data_fix/milvus_sync_docs.py --collection-name episodic_memory --batch-size 500

参数：
  --collection-name, -c  Milvus Collection 名称（必需），如: episodic_memory
  --batch-size, -b       批处理大小（默认 500）
  --limit, -l            限制处理的文档数量（默认全部）
  --days, -d             只处理过去 N 天创建的文档（默认全部）
"""

import argparse
import asyncio
import traceback

from core.observation.logger import get_logger


logger = get_logger(__name__)


async def run(
    collection_name: str, batch_size: int, limit_: int | None, days: int | None
) -> None:
    """
    同步 MongoDB 数据到 Milvus 指定 Collection。

    根据 Collection 名称路由到具体的同步实现。

    Args:
        collection_name: Milvus Collection 名称，如: episodic_memory
        batch_size: 批处理大小，默认 500
        limit_: 限制处理的文档数量，None 表示处理全部
        days: 只处理过去 N 天创建的文档，None 表示处理全部

    Raises:
        ValueError: 当 Collection 名称不支持时
        Exception: 当同步过程中发生错误时
    """
    try:
        logger.info("开始同步到 Milvus Collection: %s", collection_name)

        # 根据 Collection 名称路由到具体实现
        if collection_name == "episodic_memory":
            from devops_scripts.data_fix.milvus_sync_episodic_memory_docs import (
                sync_episodic_memory_docs,
            )

            await sync_episodic_memory_docs(
                batch_size=batch_size, limit=limit_, days=days
            )
        else:
            raise ValueError(f"不支持的 Collection 类型: {collection_name}")

    except Exception as exc:  # noqa: BLE001
        logger.error("同步文档失败: %s", exc)
        traceback.print_exc()
        raise


def main(argv: list[str] | None = None) -> int:
    """
    命令行入口函数。

    解析命令行参数并调用同步函数。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv

    Returns:
        int: 退出码，0 表示成功

    Examples:
        # 同步所有 episodic_memory 文档
        python milvus_sync_docs.py --collection-name episodic_memory

        # 只同步最近 7 天的文档，批量大小 1000
        python milvus_sync_docs.py --collection-name episodic_memory --batch-size 1000 --days 7

        # 限制只处理 10000 条文档
        python milvus_sync_docs.py --collection-name episodic_memory --limit 10000
    """
    parser = argparse.ArgumentParser(
        description="同步 MongoDB 数据到 Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --collection-name episodic_memory
  %(prog)s --collection-name episodic_memory --batch-size 1000 --days 7
  %(prog)s --collection-name episodic_memory --limit 10000
        """,
    )

    parser.add_argument(
        "--collection-name",
        "-c",
        required=True,
        help="Milvus Collection 名称，如: episodic_memory",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=500, help="批处理大小，默认 500"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="限制处理的文档数量，默认全部"
    )
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=None,
        help="只处理过去 N 天创建的文档，默认全部",
    )

    args = parser.parse_args(argv)

    # 运行异步同步任务
    asyncio.run(run(args.collection_name, args.batch_size, args.limit, args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
