import os
import yaml
import json
from typing import Dict, Any, Optional

from core.di.decorators import component
from common_utils.project_path import CURRENT_DIR


@component(name="config_provider")
class ConfigProvider:
    """配置提供者"""

    def __init__(self):
        """初始化配置提供者"""
        self.config_dir = CURRENT_DIR / "config"
        self._cache: Dict[str, Any] = {}

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        获取配置

        Args:
            config_name: 配置文件名（不含扩展名）

        Returns:
            Dict[str, Any]: 配置数据
        """
        if config_name in self._cache:
            return self._cache[config_name]

        config_file = self.config_dir / f"{config_name}.yaml"
        if not config_file.exists():
            config_file = self.config_dir / f"{config_name}.yml"
        if not config_file.exists():
            config_file = self.config_dir / f"{config_name}.json"

        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_name}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            self._cache[config_name] = config_data
            return config_data

        except Exception as e:
            raise RuntimeError(f"加载配置文件失败 {config_name}: {e}")

    def get_raw_config(self, config_name: str) -> str:
        """
        获取原始配置文本内容

        Args:
            config_name: 配置文件名（带后缀）

        Returns:
            str: 配置文件的原始文本内容
        """
        # 检查缓存
        cache_key = f"raw_{config_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 构建配置文件路径（config_name已包含后缀）
        config_file = self.config_dir / config_name

        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_name}")

        try:
            # 直接读取文本文件内容
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # 缓存原始文本内容
            self._cache[cache_key] = raw_content
            return raw_content

        except Exception as e:
            raise RuntimeError(f"读取配置文件失败 {config_name}: {e}")

    def get_available_configs(self) -> list:
        """
        获取config目录下的所有文件列表

        Returns:
            list: 文件名列表
        """
        configs = []
        for file in self.config_dir.iterdir():
            if file.is_file():
                configs.append(file.name)

        return sorted(configs)
