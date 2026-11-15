import typing
from typing import Type, Any, Dict, Set

import os
from fnmatch import fnmatch
from datetime import datetime
from elasticsearch.dsl import MetaField, AsyncDocument, field as e_field
from common_utils.datetime_utils import get_now_with_timezone, to_timezone


def get_index_ns() -> str:
    return os.getenv("SELF_ES_INDEX_NS") or ""


class DocBase(AsyncDocument):
    """Elasticsearch文档基类"""

    class Meta:
        abstract = True


class AliasSupportDoc(DocBase):
    """支持别名模式的文档类，增强了日期字段的时区处理"""

    class CustomMeta:
        # 指定用于自动填充 meta.id 的字段名（如：MongoDB 主键字段），未设置则不启用
        id_source_field: typing.Optional[str] = None
        # 缓存Date类型字段名的集合，用于快速判断（动态设置，不需要预先定义）
        # date_fields: typing.Optional[Set[str]] = None

    @classmethod
    def _init_date_fields_cache(cls) -> Set[str]:
        """
        初始化Date字段缓存，收集类中所有Date类型的字段名

        Returns:
            Date类型字段名的集合
        """
        # 从 CustomMeta 中获取缓存的 date_fields
        custom_meta = getattr(cls, 'CustomMeta', None)
        if custom_meta is not None:
            existing_cache = getattr(custom_meta, 'date_fields', None)
            if existing_cache is not None:
                return existing_cache

        date_fields = set()
        # 遍历类的所有属性，找出Date类型的字段
        for attr_name in dir(cls):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, e_field.Date):
                    date_fields.add(attr_name)
            except (AttributeError, TypeError):
                # 忽略无法获取或不是字段的属性
                continue

        # 动态设置到 CustomMeta 中
        if custom_meta is not None:
            setattr(custom_meta, 'date_fields', date_fields)

        return date_fields

    def _process_date_field(self, field_name: str, field_value: Any) -> Any:
        """
        处理日期字段，确保有时区信息

        Args:
            field_name: 字段名
            field_value: 字段值

        Returns:
            处理后的字段值
        """
        # 使用缓存的Date字段集合进行快速判断
        # 通过调用_init_date_fields_cache确保获得非None的Set[str]
        date_fields = self.__class__._init_date_fields_cache()
        if field_name in date_fields and isinstance(field_value, datetime):
            return to_timezone(field_value)
        return field_value

    def __setattr__(self, name: str, value: Any) -> None:
        """重写字段设置方法，对日期字段进行时区处理"""
        # 对日期字段进行时区处理
        processed_value = self._process_date_field(name, value)

        # 调用父类的__setattr__
        super().__setattr__(name, processed_value)

    def __init__(self, meta: Dict[str, Any] = None, **kwargs: Any):
        """重写构造函数：仅基于显式 ID_SOURCE_FIELD 设置 meta.id，缺失即报错"""

        # 初始化Date字段缓存（首次调用时会真正初始化，之后直接使用缓存）
        self.__class__._init_date_fields_cache()

        raw_kwargs = dict(kwargs)

        # 处理kwargs中的日期字段
        processed_kwargs = {}
        for field_name, field_value in raw_kwargs.items():
            processed_kwargs[field_name] = self._process_date_field(
                field_name, field_value
            )

        # 基于 ID_SOURCE_FIELD 严格设置 meta.id（无启发式），并兼容从ES构造（meta带_id）
        # 从 CustomMeta 中获取 id_source_field 配置
        custom_meta_class = getattr(self.__class__, 'CustomMeta', None)

        id_source_field = (
            getattr(custom_meta_class, 'id_source_field', None)
            if custom_meta_class
            else None
        )
        merged_meta: Dict[str, Any] = {} if meta is None else dict(meta)
        # 提取已提供的meta id（来自ES加载场景）
        given_meta_id = None
        if "id" in merged_meta and merged_meta["id"] not in (None, ""):
            given_meta_id = merged_meta["id"]
        if "_id" in merged_meta and merged_meta["_id"] not in (None, ""):
            if given_meta_id is not None and given_meta_id != merged_meta["_id"]:
                raise ValueError("meta.id conflicts between 'id' and '_id'")
            given_meta_id = merged_meta["_id"]

        if given_meta_id is not None:
            # 如果显式提供了meta id，则与ID_SOURCE_FIELD（若存在）进行一致性校验
            if id_source_field and id_source_field in processed_kwargs:
                source_value = processed_kwargs[id_source_field]
                if source_value not in (None, "") and source_value != given_meta_id:
                    raise ValueError(
                        "meta.id conflicts with value from ID_SOURCE_FIELD"
                    )
            # 规整meta字段
            merged_meta["id"] = given_meta_id
            merged_meta["_id"] = given_meta_id
        elif id_source_field:
            # 未提供meta id，则要求从ID_SOURCE_FIELD获取
            if id_source_field not in processed_kwargs or processed_kwargs[
                id_source_field
            ] in (None, ""):
                raise ValueError(
                    f"{self.__class__.__name__} requires non-empty '{id_source_field}' to set meta.id"
                )
            source_value = processed_kwargs[id_source_field]
            merged_meta["id"] = source_value
            merged_meta["_id"] = source_value

        # 调用父类构造函数
        super().__init__(merged_meta or None, **processed_kwargs)

    @classmethod
    def _matches(cls, hit):
        # override _matches to match indices in a pattern instead of just ALIAS
        # hit is the raw dict as returned by elasticsearch
        return fnmatch(hit["_index"], cls.PATTERN)

    @classmethod
    def dest(cls):
        now = get_now_with_timezone()
        return f"{cls._index._name}-{now.strftime('%Y%m%d%H%M%S%f')}"


def AliasDoc(doc_name: str, number_of_shards: int = 2) -> Type[AsyncDocument]:
    """
    创建支持别名模式的ES文档类

    自动处理日期字段的时区：
    - 对于int类型的时间戳，不进行处理
    - 对于datetime对象，如果没有时区信息，会自动添加系统当前时区
    - 确保所有日期字段都有时区信息，避免时区相关的问题

    Args:
        doc_name: 文档名称
        build_analyzers: 可选的分析器列表
        number_of_shards: 分片数量

    Returns:
        增强的文档类
    """

    if get_index_ns():
        doc_name = f"{doc_name}-{get_index_ns()}"

    class GeneratedAliasSupportDoc(AliasSupportDoc):
        PATTERN = f"{doc_name}-*"

        class Index:
            name = doc_name
            settings = {
                "number_of_shards": number_of_shards,
                "number_of_replicas": 1,
                "refresh_interval": "60s",
                "max_ngram_diff": 50,
                "max_shingle_diff": 10,
            }

        class Meta:
            dynamic = MetaField("strict")

    return GeneratedAliasSupportDoc
