"""
Milvus 重建脚本（调用 core 通用工具）

基于 MilvusCollectionBase 提供的方法实现：
- 根据别名找到对应的 Collection 管理类
- 调用 create_new_collection() 创建新集合（自动创建索引并 load）
- 执行数据迁移（支持分批处理，避免内存溢出）
- 调用 switch_alias() 切换别名到新集合
- 可选删除旧 Collection

注意：本脚本默认会迁移数据（每批3000条）。
如不需要迁移数据，可通过 --no-migrate-data 选项禁用。
"""

import argparse
import sys
import traceback
from typing import Optional, List

from pymilvus import Collection

from core.observation.logger import get_logger
from core.oxm.milvus.migration.utils import rebuild_collection


logger = get_logger(__name__)


def migrate_data_callback(
    old_collection: Collection, new_collection: Collection, batch_size: int = 3000
) -> None:
    """
    数据迁移回调函数（分批处理，避免内存溢出）

    Args:
        old_collection: 旧集合实例
        new_collection: 新集合实例
        batch_size: 每批处理的数据量，默认3000条
    """
    logger.info(
        "开始迁移数据: %s -> %s (批大小: %d)",
        old_collection.name,
        new_collection.name,
        batch_size,
    )

    # 分批查询和插入的配置
    last_id = ""  # 初始ID为空字符串，表示从头开始查询
    total_migrated = 0  # 已迁移的总数

    while True:
        # 使用id作为offset进行分页查询
        # 注意：id是字符串类型（来自MongoDB的ObjectId）
        if last_id:
            expr = f'id > "{last_id}"'
        else:
            expr = ""  # 第一次查询，不需要过滤条件

        logger.info(
            "查询第 %d 批数据，起始ID: %s",
            total_migrated // batch_size + 1,
            last_id if last_id else "开始",
        )

        query_result = old_collection.query(
            expr=expr, output_fields=["*"], limit=batch_size
        )

        # 如果查询结果为空，说明已经没有更多数据了
        if not query_result:
            logger.info("没有更多数据，迁移完成")
            break

        # 获取当前批次的数量
        batch_count = len(query_result)
        logger.info("查询到 %d 条数据，开始插入新集合...", batch_count)

        # 插入到新集合
        new_collection.insert(query_result)
        new_collection.flush()

        # 更新统计信息
        total_migrated += batch_count
        logger.info("已迁移 %d 条数据", total_migrated)

        # 更新last_id为当前批次的最大ID（字符串比较）
        # 注意：query_result是列表，每个元素是一个字典
        if query_result:
            # 找到当前批次的最大ID（按字符串排序）
            current_ids = [
                record.get("id", record.get("pk", "")) for record in query_result
            ]
            current_max_id = max(current_ids)
            last_id = current_max_id
            logger.info("当前批次最大ID: %s", last_id)

        # 如果查询到的数据量小于batch_size，说明已经是最后一批了
        if batch_count < batch_size:
            logger.info("最后一批数据，迁移完成")
            break

    logger.info("数据迁移完成: 共 %d 条", total_migrated)


def run(alias: str, drop_old: bool, migrate_data: bool, batch_size: int) -> None:
    """
    执行重建逻辑（委托给 core 工具）

    Args:
        alias: Collection 别名
        drop_old: 是否删除旧集合
        migrate_data: 是否迁移数据
        batch_size: 每批处理的数据量
    """
    try:
        # 根据是否需要迁移数据，决定是否传入回调函数
        if migrate_data:
            # 使用lambda包装migrate_data_callback，传入batch_size参数
            populate_fn = lambda old_col, new_col: migrate_data_callback(
                old_col, new_col, batch_size
            )
        else:
            populate_fn = None

        result = rebuild_collection(
            alias=alias, drop_old=drop_old, populate_fn=populate_fn
        )

        logger.info(
            "Milvus 重建完成: alias=%s, src=%s -> dest=%s, dropped_old=%s",
            result.alias,
            result.source_collection,
            result.dest_collection,
            result.dropped_old,
        )
    except Exception as exc:
        logger.error("Milvus 重建失败: %s", exc)
        traceback.print_exc()
        raise


def main(argv: Optional[List[str]] = None) -> int:
    """
    主函数：解析命令行参数并执行重建

    Args:
        argv: 命令行参数列表

    Returns:
        退出码（0 表示成功）
    """
    parser = argparse.ArgumentParser(
        description="重建并切换 Milvus Collection 别名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 重建集合并迁移数据（默认每批3000条）
  python milvus_rebuild_collection.py -a episodic_memory
  
  # 重建集合但不迁移数据
  python milvus_rebuild_collection.py -a episodic_memory --no-migrate-data
  
  # 重建集合、迁移数据并指定批大小
  python milvus_rebuild_collection.py -a episodic_memory --batch-size 5000
  
  # 重建集合、迁移数据并删除旧集合
  python milvus_rebuild_collection.py -a episodic_memory --drop-old
        """,
    )

    parser.add_argument(
        "--alias", "-a", required=True, help="Collection 别名，如: episodic_memory"
    )
    parser.add_argument(
        "--drop-old", "-x", action="store_true", help="是否删除旧集合（默认保留）"
    )
    parser.add_argument(
        "--no-migrate-data", action="store_true", help="不迁移数据（默认会迁移数据）"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=3000,
        help="每批迁移的数据量（默认3000条）",
    )

    args = parser.parse_args(argv)

    run(
        alias=args.alias,
        drop_old=args.drop_old,
        migrate_data=not args.no_migrate_data,  # 默认迁移数据
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
