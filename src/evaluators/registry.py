"""
Evaluator registry - provide evaluator registration and creation.
Uses lazy loading strategy.
"""
import importlib
from typing import Dict, Type, List

from src.evaluators.base import BaseEvaluator

_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}

_EVALUATOR_MODULES: Dict[str, str] = {
    "llm_judge": "src.evaluators.llm_judge",
    "exact_match": "src.evaluators.exact_match",
    "hybrid": "src.evaluators.hybrid",
}


def register_evaluator(name: str):
    """
    Decorator for registering evaluators.

    Usage:
        @register_evaluator("llm_judge")
        class LLMJudge(BaseEvaluator):
            ...
    """

    def decorator(cls: Type[BaseEvaluator]):
        _EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


def _ensure_evaluator_loaded(name: str):
    """
    Ensure specified evaluator is loaded (lazy loading strategy).

    Args:
        name: Evaluator name

    Raises:
        ValueError: If evaluator doesn't exist
        RuntimeError: If module loaded but not registered
    """
    if name in _EVALUATOR_REGISTRY:
        return

    if name not in _EVALUATOR_MODULES:
        raise ValueError(
            f"Unknown evaluator: {name}. "
            f"Available evaluators: {list(_EVALUATOR_MODULES.keys())}"
        )

    module_path = _EVALUATOR_MODULES[name]
    importlib.import_module(module_path)

    if name not in _EVALUATOR_REGISTRY:
        raise RuntimeError(
            f"Evaluator '{name}' module loaded but not registered. "
            f"Check if @register_evaluator('{name}') decorator is present."
        )


def create_evaluator(name: str, llm_provider=None) -> BaseEvaluator:
    """
    Create evaluator instance.

    Args:
        name: Evaluator name
        llm_provider: LLM provider (required by some evaluators)

    Returns:
        Evaluator instance
    """
    _ensure_evaluator_loaded(name)
    return _EVALUATOR_REGISTRY[name](llm_provider)


def list_evaluators() -> List[str]:
    """List all available evaluators."""
    return list(_EVALUATOR_MODULES.keys())


def register_evaluator_module(name: str, module_path: str):
    """Register an evaluator module path for lazy loading."""
    _EVALUATOR_MODULES[name] = module_path
