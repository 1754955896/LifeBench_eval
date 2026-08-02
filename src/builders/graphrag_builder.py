"""
GraphRAG Builder - validates graphrag is importable and API keys are set.

Runs from within the graphrag venv (Python >=3.11), so all checks are direct.
"""
import logging
import os
import re
from typing import Any, Dict, Optional

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder

logger = logging.getLogger(__name__)


def _resolve_env_var(value: Any) -> str:
    if not isinstance(value, str):
        return str(value) if value else ""
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)
    return re.sub(pattern, replacer, value)


@register_builder("graphrag")
class GraphRAGBuilder(BaseBuilder):
    """Validate the GraphRAG environment.

    Checks that:
      - graphrag is importable (we're running in the venv)
      - LLM / embedding API keys are configured (warn if missing)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self._ready = False

    async def build(self) -> bool:
        if self._ready:
            return True

        try:
            import graphrag  # noqa: F401
        except ImportError:
            logger.error(
                "graphrag not installed. Activate the venv and run: "
                "uv pip install graphrag"
            )
            return False

        llm_key = _resolve_env_var(self.config.get("llm", {}).get("api_key", ""))
        if not llm_key:
            logger.warning("LLM_API_KEY not configured — index/search will fail at runtime")

        embed_key = _resolve_env_var(self.config.get("embedding", {}).get("api_key", ""))
        if not embed_key:
            logger.warning("VECTORIZE_API_KEY not configured — embeddings will fail at runtime")

        self._ready = True
        logger.info("GraphRAG builder: ready, graphrag %s", self._graphrag_version())
        return True

    async def cleanup(self) -> bool:
        self._ready = False
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "GraphRAGBuilder",
            "ready": self._ready,
            "graphrag_version": self._graphrag_version() if self._ready else "unknown",
        }

    def _graphrag_version(self) -> str:
        try:
            import graphrag
            return getattr(graphrag, "__version__", "installed")
        except Exception:
            return "unknown"
