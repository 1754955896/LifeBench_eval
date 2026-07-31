"""
Elasticsearch 索引迁移工具

提供通用的 Elasticsearch 索引重建和迁移功能。
"""

import time
import traceback
from typing import Type, Any
from elasticsearch import NotFoundError, RequestError
from elasticsearch.dsl import AsyncDocument
from core.observation.logger import get_logger
from core.di.utils import get_all_subclasses
from core.oxm.es.doc_base import DocBase, get_index_ns

logger = get_logger(__name__)


def find_document_class_by_index_name(index_name: str) -> Type[AsyncDocument]:
    """
    通过索引别名查找文档类

    使用 `get_index_ns()` 进行命名空间拼接，确保与 `AliasDoc` 的别名规则一致。

    Args:
        index_name: 索引别名（例如: "episodic-memory"）

    Returns:
        匹配到的 ES 文档类

    Raises:
        ValueError: 找不到或找到多个匹配的文档类
    """
    ns = get_index_ns()
    expected_alias = f"{index_name}-{ns}" if ns else index_name

    all_doc_classes = get_all_subclasses(DocBase)

    matched_classes: list[Type[AsyncDocument]] = []
    for cls in all_doc_classes:
        if hasattr(cls, "_index") and hasattr(cls._index, "_name"):
            # hardcode一下
            if (
                cls._index._name == expected_alias
                and cls.__name__ != "GeneratedAliasSupportDoc"
            ):
                matched_classes.append(cls)

    if not matched_classes:
        available_indexes = [
            cls._index._name
            for cls in all_doc_classes
            if hasattr(cls, "_index") and hasattr(cls._index, "_name")
        ]
        logger.error("找不到索引别名 '%s' 对应的文档类", index_name)
        logger.info("可用的索引别名: %s", ", ".join(available_indexes))
        raise ValueError(f"找不到索引别名 '{index_name}' 对应的文档类")

    if len(matched_classes) > 1:
        logger.error(
            "找到多个索引别名为 '%s' 的文档类: %s",
            index_name,
            [cls.__module__ + "." + cls.__name__ for cls in matched_classes],
        )
        raise ValueError(
            f"找到多个索引别名为 '{index_name}' 的文档类: {', '.join([f'{cls.__module__}.{cls.__name__}' for cls in matched_classes])}"
        )

    document_class = matched_classes[0]

    # 基本校验
    if not hasattr(document_class, "PATTERN"):
        raise ValueError(f"文档类 {document_class.__name__} 必须有 PATTERN 属性")
    if not hasattr(document_class, "dest"):
        raise ValueError(f"文档类 {document_class.__name__} 必须有 dest() 方法")

    return document_class


async def rebuild_index(
    document_class: Type[AsyncDocument],
    es_connect: Any,
    close_old: bool = False,
    delete_old: bool = False,
) -> None:
    """
    重建 Elasticsearch 索引

    基于现有索引创建新索引，并更新别名指向。支持关闭或删除旧索引。

    Args:
        document_class: ES文档类，必须有 PATTERN 属性和 dest() 方法
        es_connect: Elasticsearch 连接对象
        close_old: 是否关闭旧索引
        delete_old: 是否删除旧索引

    Returns:
        None

    Raises:
        ValueError: 如果文档类缺少必要的属性或方法
    """
    # 验证文档类
    if not hasattr(document_class, 'PATTERN'):
        raise ValueError("文档类 %s 必须有 PATTERN 属性" % document_class.__name__)
    if not hasattr(document_class, 'dest'):
        raise ValueError("文档类 %s 必须有 dest() 方法" % document_class.__name__)

    # 获取索引信息
    index_settings = getattr(document_class, '_index', None)
    if not index_settings or not hasattr(index_settings, '_name'):
        raise ValueError("文档类 %s 缺少必要的索引设置" % document_class.__name__)

    alias_name = index_settings._name
    pattern = document_class.PATTERN
    dest_index = document_class.dest()

    logger.info("开始重建索引: %s", alias_name)
    logger.info("源索引模式: %s", pattern)
    logger.info("目标索引: %s", dest_index)

    # 检查目标索引是否已存在
    if await es_connect.indices.exists(index=dest_index):
        logger.warning("目标索引 %s 已存在，跳过重建", dest_index)
        return

    # 初始化新索引
    await document_class.init(index=dest_index)
    logger.info("已创建新索引: %s", dest_index)

    # 开始重建索引任务
    reindex_body = {"source": {"index": alias_name}, "dest": {"index": dest_index}}

    logger.info("开始数据迁移...")
    result = await es_connect.reindex(body=reindex_body, wait_for_completion=False)
    task_id = result["task"]
    logger.info("重建任务ID: %s", task_id)

    # 等待任务完成
    await wait_for_task_completion(es_connect, task_id)

    # 更新别名
    await update_aliases(es_connect, alias_name, dest_index, close_old, delete_old)

    logger.info("索引重建完成: %s", alias_name)


async def wait_for_task_completion(es_connect: Any, task_id: str) -> None:
    """
    等待 Elasticsearch 任务完成

    Args:
        es_connect: Elasticsearch 连接对象
        task_id: 任务ID
    """
    logger.info("等待重建任务完成...")

    while True:
        try:
            task_result = await es_connect.tasks.get(task_id=task_id)

            if task_result.get("completed", False):
                logger.info("重建任务已完成")
                break

            # 显示进度信息
            status = task_result.get("task", {}).get("status", {})
            if status:
                created = status.get("created", 0)
                total = status.get("total", 0)
                if total > 0:
                    progress = (created / total) * 100
                    logger.info("重建进度: %d/%d (%.1f%%)", created, total, progress)

            time.sleep(5)  # 每5秒检查一次

        except (NotFoundError, RequestError) as e:
            traceback.print_exc()
            logger.error("检查任务状态失败: %s", e)
            time.sleep(10)  # 出错时等待更长时间


async def update_aliases(
    es_connect: Any,
    alias_name: str,
    dest_index: str,
    close_old: bool = False,
    delete_old: bool = False,
) -> None:
    """
    更新 Elasticsearch 索引别名

    Args:
        es_connect: Elasticsearch 连接对象
        alias_name: 别名
        dest_index: 目标索引
        close_old: 是否关闭旧索引
        delete_old: 是否删除旧索引
    """
    logger.info("更新索引别名...")

    # 获取当前别名指向的索引
    try:
        existing_indices = list(
            (await es_connect.indices.get_alias(name=alias_name)).keys()
        )
        logger.info("当前别名 %s 指向的索引: %s", alias_name, existing_indices)
    except NotFoundError:
        existing_indices = []
        logger.info("别名 %s 不存在，将创建新别名", alias_name)

    # 刷新新索引
    await es_connect.indices.refresh(index=dest_index)

    # 构建别名更新操作
    actions = []

    # 移除旧的别名关联
    for old_index in existing_indices:
        actions.append({"remove": {"alias": alias_name, "index": old_index}})

    # 添加新的别名关联
    actions.append(
        {"add": {"alias": alias_name, "index": dest_index, "is_write_index": True}}
    )

    # 执行别名更新
    await es_connect.indices.update_aliases(body={"actions": actions})
    logger.info("已更新别名 %s 指向 %s", alias_name, dest_index)

    # 处理旧索引
    for old_index in existing_indices:
        if close_old:
            logger.info("关闭旧索引: %s", old_index)
            await es_connect.indices.close(index=old_index)
        elif delete_old:
            logger.info("删除旧索引: %s", old_index)
            await es_connect.indices.delete(index=old_index)
