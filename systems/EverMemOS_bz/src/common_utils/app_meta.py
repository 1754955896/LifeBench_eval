"""
应用元数据管理模块

提供对应用元数据的读写操作，包括服务名称等信息。
"""

from typing import Dict, Optional

# 应用元数据存储
_APP_META_DATA: Dict = {}


def set_service_name(name: str) -> None:
    """
    设置服务名称

    Args:
        name: 服务名称
    """
    _APP_META_DATA['service_name'] = name


def get_service_name() -> Optional[str]:
    """
    获取服务名称

    Returns:
        str: 服务名称，如果未设置则返回None
    """
    return _APP_META_DATA.get('service_name')


def set_meta_data(key: str, value: any) -> None:
    """
    设置元数据

    Args:
        key: 元数据键
        value: 元数据值
    """
    _APP_META_DATA[key] = value


def get_meta_data(key: str) -> Optional[any]:
    """
    获取元数据

    Args:
        key: 元数据键

    Returns:
        any: 元数据值，如果不存在则返回None
    """
    return _APP_META_DATA.get(key)


def get_all_meta_data() -> Dict:
    """
    获取所有元数据

    Returns:
        Dict: 所有元数据的副本
    """
    return _APP_META_DATA.copy()
