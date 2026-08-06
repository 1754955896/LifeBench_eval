"""
Dataset loader registry.
"""
import importlib
from typing import Dict, Type, Optional

from src.loaders.base import BaseLoader

_LOADER_REGISTRY: Dict[str, Type["BaseLoader"]] = {}

_LOADER_MODULES: Dict[str, str] = {
    "locomo": "src.loaders.locomo",
}


def register_loader(name: str):
    """Decorator for registering loaders."""
    def decorator(cls: Type[BaseLoader]):
        _LOADER_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_loader_loaded(name: str):
    """Ensure loader is loaded."""
    if name in _LOADER_REGISTRY:
        return

    if name not in _LOADER_MODULES:
        raise ValueError(f"Unknown loader: {name}")

    module_path = _LOADER_MODULES[name]
    importlib.import_module(module_path)

    if name not in _LOADER_REGISTRY:
        raise RuntimeError(f"Loader {name} not registered")


def create_loader(name: str) -> BaseLoader:
    """Create loader instance."""
    _ensure_loader_loaded(name)
    return _LOADER_REGISTRY[name]()


def load_dataset(name: str, data_path: str, **kwargs) -> "Dataset":
    """Load dataset using appropriate loader."""
    from src.models.dataset import Dataset

    dataset_format = kwargs.pop("dataset_format", None) or "locomo"
    loader_name = {"lifebench": "locomo", "locomo": "locomo"}.get(dataset_format)
    if loader_name is None:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    _ensure_loader_loaded(loader_name)
    return _LOADER_REGISTRY[loader_name]().load(
        data_path,
        name=name,
        dataset_format=dataset_format,
        **kwargs,
    )


def register_loader_module(name: str, module_path: str):
    """Register a loader module path."""
    _LOADER_MODULES[name] = module_path
