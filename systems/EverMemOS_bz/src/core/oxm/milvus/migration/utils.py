"""
Milvus 集合重建与别名切换通用工具

设计目标：
- 将与基础设施相关、可复用的 Milvus 重建逻辑沉淀在 core 层
- 业务或脚本层仅负责获取 client、传入 alias 和选项

注意：
- 本工具仅做结构重建与别名切换，不做数据迁移
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Type

from pymilvus import Collection

from core.observation.logger import get_logger
from core.oxm.milvus.milvus_collection_base import (
    MilvusCollectionBase,
    MilvusCollectionWithSuffix,
)
from core.di.utils import get_all_subclasses


logger = get_logger(__name__)


@dataclass
class RebuildResult:
    """Milvus 集合重建结果。"""

    alias: str
    source_collection: str
    dest_collection: str
    dropped_old: bool


def find_collection_manager_by_alias(alias: str) -> Type[MilvusCollectionBase]:
    """
    根据别名查找对应的 Collection 管理类

    Args:
        alias: Collection 别名

    Returns:
        对应的 MilvusCollectionBase 子类

    Raises:
        ValueError: 找不到对应的集合类
    """
    all_doc_classes = get_all_subclasses(MilvusCollectionBase)

    # 遍历所有子类，找到 alias 匹配的
    for doc_class in all_doc_classes:
        # 跳过抽象类
        # pylint: disable=protected-access  # 框架内部使用，访问子类的配置属性
        if (
            not hasattr(doc_class, '_COLLECTION_NAME')
            or doc_class._COLLECTION_NAME is None
        ):
            continue

        # 检查是否是 MilvusCollectionWithSuffix 类型
        if issubclass(doc_class, MilvusCollectionWithSuffix):
            # 临时实例化以获取 alias（需要解析 suffix）
            try:
                # 尝试从 alias 解析 suffix
                base_name = (
                    doc_class._COLLECTION_NAME
                )  # pylint: disable=protected-access
                if alias.startswith(base_name):
                    return doc_class
            except (
                Exception
            ):  # pylint: disable=broad-except  # 忽略实例化失败，继续尝试下一个类
                continue
        else:
            # 对于 MilvusCollectionBase，直接比较 _COLLECTION_NAME
            if doc_class._COLLECTION_NAME == alias:  # pylint: disable=protected-access
                return doc_class

    raise ValueError(f"找不到别名 '{alias}' 对应的集合类")


def rebuild_collection(
    alias: str,
    drop_old: bool = False,
    populate_fn: Optional[Callable[[Collection, Collection], None]] = None,
) -> RebuildResult:
    """
    基于别名进行 Milvus 集合重建流程：
    1) 根据别名找到对应的 Collection 管理类
    2) 调用 create_new_collection() 创建新集合（自动创建索引并 load）
    3) 调用可选的数据填充回调（由调用方实现）
    4) 调用 switch_alias() 切换别名，按需删除旧集合

    Args:
        alias: 集合别名
        drop_old: 是否删除旧集合
        populate_fn: 可选回调，用于在索引创建完成后、别名切换前执行数据填充。
            函数签名为 (old_collection: Collection, new_collection: Collection) -> None

    Returns:
        RebuildResult: 重建结果信息

    Raises:
        ValueError: 找不到对应的集合类
        MilvusException: Milvus 操作失败
    """
    logger.info("开始重建 Collection: alias=%s, drop_old=%s", alias, drop_old)

    # 1. 根据别名找到对应的 Collection 管理类
    collection_class = find_collection_manager_by_alias(alias)
    logger.info("找到集合类: %s", collection_class.__name__)

    # 2. 实例化管理器（从 alias 解析 suffix）
    if issubclass(collection_class, MilvusCollectionWithSuffix):
        # 从 alias 解析 suffix
        base_name = (
            collection_class._COLLECTION_NAME
        )  # pylint: disable=protected-access
        suffix = None
        if alias != base_name and alias.startswith(base_name + "_"):
            suffix = alias[len(base_name) + 1 :]
        manager = collection_class(suffix=suffix)
    else:
        raise NotImplementedError("不支持的集合类型: %s", collection_class.__name__)

    # 确保原集合已加载
    manager.ensure_loaded()
    old_collection = manager.collection()
    old_real_name = old_collection.name
    logger.info("原集合真实名: %s", old_real_name)

    # 3. 创建新集合（自动创建索引并 load）
    logger.info("开始创建新集合...")
    new_collection = manager.create_new_collection()
    new_real_name = new_collection.name
    logger.info("新集合创建完成: %s", new_real_name)

    # 4. 调用数据填充回调（如果提供）
    if populate_fn:
        logger.info("开始执行数据填充回调...")
        try:
            populate_fn(old_collection, new_collection)
            logger.info("数据填充完成")
        except Exception as e:
            logger.error("数据填充失败: %s", e)
            raise

    # 5. 切换别名到新集合，并可选删除旧集合
    logger.info("切换别名 '%s' 到新集合 '%s'...", alias, new_real_name)
    manager.switch_alias(new_collection, drop_old=drop_old)

    logger.info(
        "重建完成: alias=%s, src=%s -> dest=%s, dropped_old=%s",
        alias,
        old_real_name,
        new_real_name,
        drop_old,
    )

    return RebuildResult(
        alias=alias,
        source_collection=old_real_name,
        dest_collection=new_real_name,
        dropped_old=drop_old,
    )
