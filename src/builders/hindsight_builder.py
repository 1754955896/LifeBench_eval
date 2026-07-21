"""
Hindsight builder - prepares environment for MemoryEngine direct mode.

No Docker required for local operation (uses embedded pg0).
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder

logger = logging.getLogger(__name__)


def _load_env_config(project_env: Path) -> dict:
    """Load configuration from project .env file.

    Args:
        project_env: Path to LifeBench_eval/.env

    Returns:
        Dict with LLM, embedding, rerank settings
    """
    config = {}
    if project_env.exists():
        with open(project_env, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)

    config["llm_api_key"] = os.environ.get("LLM_API_KEY", "")
    config["llm_base_url"] = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    config["llm_model"] = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
    config["vectorize_api_key"] = os.environ.get("VECTORIZE_API_KEY", "")
    config["vectorize_base_url"] = os.environ.get("VECTORIZE_BASE_URL", "https://api.siliconflow.cn/v1")
    config["vectorize_model"] = os.environ.get("VECTORIZE_MODEL", "Qwen/Qwen3-Embedding-4B")
    config["vectorize_dimensions"] = os.environ.get("VECTORIZE_DIMENSIONS", "1024")
    config["rerank_api_key"] = os.environ.get("RERANK_API_KEY", os.environ.get("VECTORIZE_API_KEY", ""))
    config["rerank_base_url"] = os.environ.get("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
    config["rerank_model"] = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    return config


def _resolve_env_var(value: str) -> str:
    """Resolve ${VAR:default} style environment variable references.

    Args:
        value: String that may contain ${VAR} or ${VAR:default}

    Returns:
        Resolved string with env vars expanded
    """
    if not isinstance(value, str):
        return value

    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    return re.sub(pattern, replacer, value)


@register_builder("hindsight")
class HindsightBuilder(BaseBuilder):
    """Hindsight builder for MemoryEngine direct mode.

    No Docker required - uses embedded pg0 database.
    Creates a single MemoryEngine instance shared by all samples.
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self._started = False
        self._memory: Optional[Any] = None

    def _get_project_root(self) -> Optional[Path]:
        """Get project root from project_root config or script location."""
        if self.project_root:
            return Path(self.project_root)
        cli_path = Path(__file__).parent.parent.parent / "cli.py"
        if cli_path.exists():
            return cli_path.parent.resolve()
        return None

    async def build(self) -> bool:
        """Create MemoryEngine instance and store in config for reuse.

        1. Loads config from LifeBench_eval/.env
        2. Sets HINDSIGHT_API_* environment variables from config
        3. Creates single MemoryEngine instance (shared by all samples)
        """
        if self._started:
            logger.info("Hindsight builder already started")
            return True

        project_root = Path(self.project_root) if self.project_root else self._get_project_root()

        # 1. Load .env file from project root
        if project_root:
            project_env = project_root / ".env"
            if project_env.exists():
                env_config = _load_env_config(project_env)
                logger.info(f"Loaded config from {project_env}")
                logger.info(f"  LLM: {env_config['llm_model']} @ {env_config['llm_base_url']}")

        # 2. Set environment variables from config
        env_mappings = {
            # Database
            "HINDSIGHT_API_DATABASE_URL": self.config.get("db_url", "pg0"),

            # Memory LLM (for fact extraction/consolidation)
            "HINDSIGHT_API_LLM_PROVIDER": self.config.get("memory_llm_provider", "deepseek"),
            "HINDSIGHT_API_LLM_API_KEY": _resolve_env_var(self.config.get("memory_llm_api_key", "")),
            "HINDSIGHT_API_LLM_MODEL": self.config.get("memory_llm_model", "deepseek-v4-flash"),
            "HINDSIGHT_API_LLM_BASE_URL": self.config.get("memory_llm_base_url") or "https://api.deepseek.com",

            # Answer LLM (falls back to memory_llm if not set)
            "HINDSIGHT_API_ANSWER_LLM_PROVIDER": self.config.get("answer_llm_provider", "deepseek"),
            "HINDSIGHT_API_ANSWER_LLM_API_KEY": _resolve_env_var(self.config.get("answer_llm_api_key", "")),
            "HINDSIGHT_API_ANSWER_LLM_MODEL": self.config.get("answer_llm_model", "deepseek-v4-flash"),
            "HINDSIGHT_API_ANSWER_LLM_BASE_URL": self.config.get("answer_llm_base_url") or "https://api.deepseek.com",

            # Embeddings (SiliconFlow dedicated API)
            "HINDSIGHT_API_EMBEDDINGS_PROVIDER": self.config.get("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "siliconflow"),
            "HINDSIGHT_API_EMBEDDINGS_SILICONFLOW_API_KEY": _resolve_env_var(self.config.get("HINDSIGHT_API_EMBEDDINGS_SILICONFLOW_API_KEY", "")),
            "HINDSIGHT_API_EMBEDDINGS_SILICONFLOW_MODEL": self.config.get("HINDSIGHT_API_EMBEDDINGS_SILICONFLOW_MODEL", "Qwen/Qwen3-Embedding-4B"),
            "HINDSIGHT_API_EMBEDDINGS_SILICONFLOW_BASE_URL": self.config.get("HINDSIGHT_API_EMBEDDINGS_SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            "HINDSIGHT_API_EMBEDDINGS_SILICONFLOW_DIMENSIONS": str(self.config.get("HINDSIGHT_API_EMBEDDINGS_SILICONFLOW_DIMENSIONS", 1024)),

            # Reranker (SiliconFlow native rerank)
            "HINDSIGHT_API_RERANKER_PROVIDER": self.config.get("HINDSIGHT_API_RERANKER_PROVIDER", "siliconflow"),
            "HINDSIGHT_API_RERANKER_SILICONFLOW_API_KEY": _resolve_env_var(self.config.get("HINDSIGHT_API_RERANKER_SILICONFLOW_API_KEY", "")),
            "HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL": self.config.get("HINDSIGHT_API_RERANKER_SILICONFLOW_MODEL", "BAAI/bge-reranker-v2-m3"),
            "HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL": self.config.get("HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        }

        # Only set non-empty values to allow env vars to take precedence
        for key, value in env_mappings.items():
            if value:
                os.environ[key] = value
                # Mask API key in logs
                if "API_KEY" in key and value:
                    masked = value[:8] + "..." if len(value) > 8 else "***"
                    logger.info(f"  {key}={masked}")
                else:
                    logger.info(f"  {key}={value}")

        # 3. Create MemoryEngine instance (shared by all samples via config)
        try:
            from hindsight_api import MemoryEngine
            from hindsight_api.config import get_config

            # Configure logging
            get_config().configure_logging()

            db_url = self.config.get("db_url", os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"))
            memory_llm_provider = self.config.get("memory_llm_provider", os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"))
            memory_llm_api_key = _resolve_env_var(self.config.get("memory_llm_api_key", "")) or os.environ.get("LLM_API_KEY", "")
            memory_llm_model = self.config.get("memory_llm_model", os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"))
            memory_llm_base_url = self.config.get("memory_llm_base_url") or os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None

            self._memory = MemoryEngine(
                db_url=db_url,
                memory_llm_provider=memory_llm_provider,
                memory_llm_api_key=memory_llm_api_key,
                memory_llm_model=memory_llm_model,
                memory_llm_base_url=memory_llm_base_url,
            )
            await self._memory.initialize()
            logger.info("MemoryEngine initialized successfully")

            # Store in config so adapter can reuse the same instance
            self.config["_hindsight_memory"] = self._memory
            logger.info("MemoryEngine stored in config for adapter reuse")

        except ImportError as e:
            logger.error(f"MemoryEngine not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to create MemoryEngine: {e}")
            return False

        self._started = True
        logger.info("Hindsight builder completed (local mode, no Docker needed)")
        return True

    async def cleanup(self) -> bool:
        """Cleanup - close MemoryEngine and stop pg0."""
        if not self._started:
            logger.info("Hindsight builder not started, nothing to cleanup")
            return True

        # Remove from config first
        if "_hindsight_memory" in self.config:
            del self.config["_hindsight_memory"]

        # Close MemoryEngine (stops embedded pg0)
        if self._memory is not None:
            await self._memory.close()
            self._memory = None
            logger.info("MemoryEngine closed (pg0 stopped)")

        self._started = False
        logger.info("Hindsight builder cleanup completed")
        return True

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "mode": "local",
        }
