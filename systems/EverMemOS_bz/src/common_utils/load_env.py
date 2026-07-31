#!/usr/bin/env python3
"""
统一的环境加载工具

提供Python路径设置和.env文件加载功能，确保项目在不同位置运行时都能正确加载环境变量。
"""

import logging
import os
import sys
from typing import Optional
from dotenv import load_dotenv
import time

from common_utils.app_meta import set_service_name
from common_utils.project_path import PROJECT_DIR

# 这里环境变量还没加载，所以不能使用get_logger
logger = logging.getLogger(__name__)

# 项目元数据已迁移到app_meta模块中

"""
- setup_pythonpath不需要。
  - 对于python run.py 和python src/run.py不需要这个，src会被加进来。
  - 如果真的没有src，setup_pythonpath需要导入load_env.py，然后又要依赖pythonpath，实际也加载不进来。
  - vscode启动不需要，可以launch.json配置。
  - 线上用的python run.py也不需要。
    - 入口点 web 确实会让src找不到，这个要解决一下。
"""


def load_env_file(
    env_file_name: str = ".env", check_env_var: Optional[str] = None
) -> bool:
    """
    加载.env文件

    Args:
        env_file_name: .env文件名
        check_env_var: 检查的环境变量名，用于判断环境是否已加载

    Returns:
        bool: 是否成功加载环境变量
    """
    # 基于load_env.py的位置计算.env文件路径
    # .env文件在src的父目录

    env_file_path = PROJECT_DIR / env_file_name

    if not env_file_path.exists():
        logger.warning(".env文件不存在: %s", env_file_path)
        return False

    try:
        load_dotenv(env_file_path)
        logger.debug("成功加载.env文件: %s", env_file_path)
    except (IOError, OSError) as e:
        logger.error("加载.env文件失败: %s", e)
        return False

    if check_env_var and os.getenv(check_env_var):
        logger.info("%s 已设置，已加载环境变量", check_env_var)
        return True
    else:
        if check_env_var:
            logger.error("请确保%s环境变量已设置", check_env_var)
        return False


def reset_timezone():
    """
    重置时区
    """
    timezone = os.environ.get("TZ") or "Asia/Shanghai"
    os.environ["TZ"] = timezone
    # tzset() is not available on Windows, only call it if available
    if hasattr(time, 'tzset'):
        time.tzset()


def sync_pythonpath_with_syspath():
    """
    同步 PYTHONPATH 和 sys.path，确保 sys.path 中的路径都在 PYTHONPATH 中

    注意：
    1. 只同步非标准库路径
    2. 排除 .venv 和类似的虚拟环境路径
    3. 保持原有 PYTHONPATH 的优先级
    """
    import sys
    import os
    from pathlib import Path

    # 获取当前 PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "").split(":")
    pythonpath = [p for p in pythonpath if p]  # 移除空字符串

    # 需要排除的路径模式
    exclude_patterns = [
        ".venv",
        "site-packages",
        "dist-packages",
        "lib/python",
        "__pycache__",
    ]

    # 从 sys.path 中获取需要添加的路径
    new_paths = []
    for path in sys.path:
        # 跳过空路径
        if not path:
            continue

        # 转换为 Path 对象以便处理
        path_obj = Path(path)

        # 跳过不存在的路径
        if not path_obj.exists():
            continue

        # 跳过需要排除的路径
        if any(pattern in str(path_obj) for pattern in exclude_patterns):
            continue

        # 转换为字符串并规范化
        path_str = str(path_obj.resolve())

        # 如果路径不在当前 PYTHONPATH 中，添加到新路径列表
        if path_str not in pythonpath:
            new_paths.append(path_str)

    # 如果有新的路径需要添加
    if new_paths:
        # 将新路径添加到现有 PYTHONPATH 后面
        all_paths = pythonpath + new_paths
        # 更新环境变量
        os.environ["PYTHONPATH"] = ":".join(all_paths)
        logger.debug("已更新 PYTHONPATH: %s", os.environ["PYTHONPATH"])


def setup_environment(
    load_env_file_name: str = ".env",
    check_env_var: Optional[str] = None,
    service_name: Optional[str] = None,
) -> bool:
    """
    统一的环境设置函数

    Args:
        load_env_file_name: .env文件名
        check_env_var: 检查的环境变量名，用于判断环境是否已加载
        service_name: 当前启动服务的名称，将被存储在APP_META_DATA中

    Returns:
        bool: 是否成功设置环境
    """
    # 加载.env文件
    success = load_env_file(
        env_file_name=load_env_file_name, check_env_var=check_env_var
    )

    # 同步 PYTHONPATH 和 sys.path
    sync_pythonpath_with_syspath()

    # 重置时区
    reset_timezone()

    # 设置服务名称
    if service_name:
        set_service_name(service_name)
        logger.debug("已设置服务名称: %s", service_name)

    if not success:
        logger.error("环境设置失败，程序退出")
        sys.exit(1)

    return success
