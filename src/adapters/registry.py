"""
Adapter registry - provide adapter registration and creation.
Uses lazy loading strategy.
"""
import importlib
from typing import Dict, List, Type, Optional

from src.adapters.base import BaseAdapter

_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}

_ADAPTER_MODULES: Dict[str, str] = {
    "example": "src.adapters.example_adapter",
    "mem0": "src.adapters.mem0_adapter",
    "hindsight": "src.adapters.hindsight_adapter",
    "memos_cloud": "src.adapters.memos_cloud_adapter",
    "memos_cloud_blocking": "src.adapters.memos_cloud_nonblocking_adapter",
    "cognee": "src.adapters.cognee_adapter",
    "memu_server": "src.adapters.memu_server_adapter",
    "graphiti_local": "src.adapters.graphiti_local_adapter",
    "graphrag": "src.adapters.graphrag_adapter",
    "evermemos": "src.adapters.evermemos_adapter",
    "evermemos_native": "src.adapters.evermemos_native_adapter",
    "mindmemos": "src.adapters.mindmemos_adapter",
    "datatest": "src.adapters.datatest_adapter",
}


def register_adapter(name: str):
    """
    Decorator for registering adapters.

    Usage:
        @register_adapter("lifemem")
        class LifeMemAdapter(BaseAdapter):
            ...
    """

    def decorator(cls: Type[BaseAdapter]):
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def _ensure_adapter_loaded(name: str):
    """
    Ensure specified adapter is loaded (lazy loading strategy).

    Args:
        name: Adapter name

    Raises:
        ValueError: If adapter doesn't exist
        RuntimeError: If module loaded but not registered
    """
    if name in _ADAPTER_REGISTRY:
        return

    if name not in _ADAPTER_MODULES:
        raise ValueError(
            f"Unknown adapter: {name}. "
            f"Available adapters: {list(_ADAPTER_MODULES.keys())}"
        )

    module_path = _ADAPTER_MODULES[name]
    importlib.import_module(module_path)

    if name not in _ADAPTER_REGISTRY:
        raise RuntimeError(
            f"Adapter '{name}' module loaded but not registered. "
            f"Check if @register_adapter('{name}') decorator is present."
        )


def create_adapter(
    name: str, config: dict, output_dir=None, stats_collector=None
) -> BaseAdapter:
    """
    Create adapter instance.

    Args:
        name: Adapter name
        config: Config dict
        output_dir: Output directory (optional)
        stats_collector: Optional stats collector

    Returns:
        Adapter instance
    """
    _ensure_adapter_loaded(name)

    try:
        return _ADAPTER_REGISTRY[name](
            config, output_dir=output_dir, stats_collector=stats_collector
        )
    except TypeError:
        try:
            return _ADAPTER_REGISTRY[name](config, output_dir=output_dir)
        except TypeError:
            return _ADAPTER_REGISTRY[name](config)


def list_adapters() -> List[str]:
    """List all available adapters."""
    return list(_ADAPTER_MODULES.keys())


def register_adapter_module(name: str, module_path: str):
    """Register an adapter module path for lazy loading."""
    _ADAPTER_MODULES[name] = module_path
