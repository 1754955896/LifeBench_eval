#!/usr/bin/env python3
"""
Add Timestamp Shard

为MemCell集合添加基于timestamp的时间戳分片配置
创建时间: 2025-09-11T23:37:54.703305
"""

import asyncio
import logging
from typing import Optional

from pymongo.errors import OperationFailure

from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell

logger = logging.getLogger(__name__)


async def enable_timestamp_sharding(session=None):
    """
    启用MemCell集合的timestamp分片
    """
    try:
        # 获取MongoDB集合和客户端
        collection = MemCell.get_pymongo_collection()
        db = collection.database
        client = db.client
        admin_db = client.admin

        logger.info("🔧 开始配置timestamp分片...")

        # 1. 检查是否为分片集群
        try:
            shard_status = await admin_db.command('listShards')
            if not shard_status.get('shards'):
                logger.warning("⚠️  当前不是分片集群环境，跳过分片配置")
                return
            logger.info(f"✅ 检测到分片集群，共 {len(shard_status['shards'])} 个分片")
        except OperationFailure as e:
            logger.warning(f"⚠️  无法检查分片状态: {e}，可能不是分片环境")
            return

        # 2. 启用数据库分片
        try:
            await admin_db.command('enableSharding', db.name)
            logger.info(f"✅ 数据库 '{db.name}' 分片已启用")
        except OperationFailure as e:
            if "already enabled" in str(e).lower():
                logger.info(f"📝 数据库 '{db.name}' 分片已存在")
            else:
                logger.error(f"❌ 启用数据库分片失败: {e}")
                raise

        # 3. 设置集合分片键 - timestamp
        collection_name = f"{db.name}.memcells"
        try:
            await admin_db.command(
                'shardCollection', collection_name, key={"timestamp": 1}
            )
            logger.info("✅ MemCell集合timestamp分片键设置完成")
        except OperationFailure as e:
            if "already sharded" in str(e).lower():
                logger.info("📝 MemCell集合分片已存在")
            else:
                logger.error(f"❌ 设置集合分片失败: {e}")
                raise

        # 4. 创建预分片（可选，提高初始性能）
        try:
            from datetime import datetime, timedelta

            # 创建未来12个月的预分片点
            base_date = datetime.now().replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            split_points = []

            for i in range(1, 13):  # 未来12个月
                split_date = base_date + timedelta(days=30 * i)
                split_points.append({"timestamp": split_date})

            # 执行预分片
            for point in split_points:
                try:
                    await admin_db.command('split', collection_name, middle=point)
                    logger.debug(f"📅 创建分片点: {point['timestamp']}")
                except OperationFailure as e:
                    if "already exists" not in str(e).lower():
                        logger.debug(f"预分片点创建失败: {e}")

            logger.info(f"✅ 创建了 {len(split_points)} 个预分片点")

        except Exception as e:
            logger.warning(f"⚠️  预分片创建失败: {e}")

        # 5. 验证分片配置
        try:
            shard_info = await db.command('collStats', 'memcells')

            if shard_info.get('sharded'):
                logger.info("✅ MemCell集合分片配置验证成功")
                logger.info(f"📊 分片键: {shard_info.get('shardKey', {})}")
            else:
                logger.warning("⚠️  分片配置验证失败")

        except Exception as e:
            logger.warning(f"⚠️  分片验证失败: {e}")

        logger.info("🎉 timestamp分片配置完成")

    except Exception as e:
        logger.error(f"❌ 分片配置过程中发生错误: {e}")
        raise


async def disable_timestamp_sharding(session=None):
    """
    警告：禁用分片是危险操作，通常不推荐在生产环境执行
    """
    logger.warning("⚠️  禁用分片是危险操作，需要管理员手动处理")
    logger.info("📝 请手动执行以下MongoDB命令来禁用分片:")
    logger.info("   1. 停止均衡器: sh.stopBalancer()")
    logger.info("   2. 等待均衡完成: sh.waitForBalancer()")
    logger.info("   3. 移除分片配置需要重新创建集合")


async def main():
    """主函数"""
    # 执行分片配置
    await enable_timestamp_sharding()


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
