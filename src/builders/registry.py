"""
Builder registry - provides builder registration and creation.
"""

import importlib
import logging
from typing import Dict, Type, Optional

from src.builders.base_builder import BaseBuilder

logger = logging.getLogger(__name__)

_BUILDER_REGISTRY: Dict[str, Type[BaseBuilder]] = {}

_BUILDER_MODULES: Dict[str, str] = {
    "mem0": "src.builders.mem0_builder",
    "hindsight": "src.builders.hindsight_builder",
    "cognee": "src.builders.cognee_builder",
    "memu_server": "src.builders.memu_server_builder",
    "graphiti": "src.builders.graphiti_builder",
    "graphiti_local": "src.builders.graphiti_local_builder",
    "graphrag": "src.builders.graphrag_builder",
    "tencentdb": "src.builders.tencentdb_builder",
    "evermemos": "src.builders.evermemos_builder",
    "evermemos_native": "src.builders.evermemos_native_builder",
    "mindmemos": "src.builders.mindmemos_builder",
}


def register_builder(name: str):
    """
    Decorator for registering builders.

    Usage:
        @register_builder("mem0")
        class Mem0Builder(BaseBuilder):
            ...
    """

    def decorator(cls: Type[BaseBuilder]):
        _BUILDER_REGISTRY[name] = cls
        logger.debug(f"Registered builder: {name}")
        return cls

    return decorator


def _ensure_builder_loaded(name: str):
    """
    Ensure specified builder is loaded (lazy loading strategy).

    Args:
        name: Builder name

    Raises:
        ValueError: If builder doesn't exist
        RuntimeError: If module loaded but not registered
    """
    if name in _BUILDER_REGISTRY:
        return

    if name not in _BUILDER_MODULES:
        raise ValueError(
            f"Unknown builder: {name}. "
            f"Available builders: {list(_BUILDER_MODULES.keys())}"
        )

    module_path = _BUILDER_MODULES[name]
    importlib.import_module(module_path)

    if name not in _BUILDER_REGISTRY:
        raise RuntimeError(
            f"Builder '{name}' module loaded but not registered. "
            f"Check if @register_builder('{name}') decorator is present."
        )


def create_builder(name: str, config: dict, project_root: Optional[str] = None) -> BaseBuilder:
    """
    Create builder instance.

    Args:
        name: Builder name
        config: Config dict
        project_root: Project root path (optional)

    Returns:
        Builder instance
    """
    _ensure_builder_loaded(name)

    try:
        return _BUILDER_REGISTRY[name](config, project_root=project_root)
    except TypeError:
        try:
            return _BUILDER_REGISTRY[name](config)
        except TypeError:
            return _BUILDER_REGISTRY[name](config, None)


def list_builders():
    """List all available builders."""
    return list(_BUILDER_MODULES.keys())


def register_builder_module(name: str, module_path: str):
    """Register a builder module path for lazy loading."""
    _BUILDER_MODULES[name] = module_path