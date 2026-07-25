"""
Cognee builder — writes harness config into the vendored cognee .env file.

Cognee's __init__.py now loads `.env` from its own package directory via an
explicit `dotenv_path=`. This builder prepares that file before the adapter
imports cognee, so cognee always sees the correct configuration regardless
of CWD — no OS env var injection or dotenv monkey-patching needed.
"""

import logging
import os
import re
from typing import Any, Dict, Optional
from pathlib import Path

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _absolutize_path(value: str, project_root: Optional[str]) -> str:
    """Resolve relative paths to absolute — cognee's BaseConfig rejects them."""
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(("s3://", "gs://", "http://", "https://")):
        return value
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p)
    base = Path(project_root).expanduser() if project_root else Path.cwd()
    return str((base / p).resolve())


def _resolve_env_var(value: Any) -> Any:
    """Resolve ${VAR:default} style env-var references."""
    if not isinstance(value, str):
        return value
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    return re.sub(pattern, replacer, value)


def _mask_value(key: str, value: str) -> str:
    """Mask API keys for logging."""
    if "API_KEY" in key:
        return value[:8] + "..." if len(value) > 8 else "***"
    return value


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

@register_builder("cognee")
class CogneeBuilder(BaseBuilder):
    """In-process cognee SDK builder.

    Generates cognee's .env file from the harness YAML config so that
    cognee's own `dotenv.load_dotenv(dotenv_path=...)` picks up the right
    values at import time. No OS env var injection, no monkey-patching.

    Configuration (in YAML `config/systems/cognee.yaml`):
        llm:                    dict with provider, model, api_key, base_url,
                                temperature, max_tokens.
        embedding:              dict with provider, model, api_key, base_url,
                                dimensions, huggingface_tokenizer.
        data_root_directory:    cognee data path (DATA_ROOT_DIRECTORY).
        system_root_directory:  cognee system path (SYSTEM_ROOT_DIRECTORY).
        enable_backend_access_control:  bool, default false.
        structured_output_framework:    str, default "instructor".
        prune_on_init:          bool, wipe cognee state before run.
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self._started = False

    # -- BaseBuilder contract ------------------------------------------------

    async def build(self) -> bool:
        if self._started:
            logger.info("Cognee builder already started")
            return True

        env_path = self._resolve_env_path()
        content = self._generate_env_content()
        env_path.write_text(content, encoding="utf-8")
        logger.info("Cognee .env written: %s", env_path)

        if self.config.get("prune_on_init", False):
            await self._prune_state()

        self._started = True
        logger.info("Cognee builder completed (in-process mode)")
        return True

    async def cleanup(self) -> bool:
        if not self._started:
            return True
        self._started = False
        logger.info("Cognee builder cleanup completed")
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "mode": "in_process",
        }

    # -- .env file generation ------------------------------------------------

    def _resolve_env_path(self) -> Path:
        """Path to systems/cognee/.env.

        cognee_builder.py lives at:  src/builders/cognee_builder.py
        cognee .env is at:           systems/cognee/.env
        """
        return (Path(__file__).resolve().parents[2] / "systems" / "cognee" / ".env")

    def _generate_env_content(self) -> str:
        """Translate harness config into a cognee-format .env file."""
        lines: list[str] = []
        written: set[str] = set()  # track keys to avoid duplicates

        self._write_section(lines, "LLM Config (DeepSeek / OpenAI-compatible)")
        self._write_llm(lines, written)

        self._write_section(lines, "Embedding Config")
        self._write_embedding(lines, written)

        if self.config.get("rerank"):
            logger.warning(
                "cognee eval_framework does not implement a rerank model. "
                "The 'rerank' config block is ignored."
            )

        self._write_section(lines, "Data Directories")
        self._write_paths(lines, written)

        self._write_kv(lines, "ENABLE_BACKEND_ACCESS_CONTROL",
                       str(self.config.get("enable_backend_access_control", "false")).lower(),
                       written)
        self._write_kv(lines, "STRUCTURED_OUTPUT_FRAMEWORK",
                       str(self.config.get("structured_output_framework", "instructor")),
                       written)

        return "\n".join(lines) + "\n"

    def _write_section(self, lines: list[str], title: str) -> None:
        if lines:
            lines.append("")  # blank before each section
        lines.append(f"# {'=' * 60}")
        lines.append(f"# {title}")
        lines.append(f"# {'=' * 60}")

    def _write_llm(self, lines: list[str], written: set[str]) -> None:
        llm = self.config.get("llm", {}) or {}
        if not llm:
            return

        provider = str(_resolve_env_var(llm.get("provider", "openai")))
        model = _resolve_env_var(llm.get("model", ""))
        api_key = _resolve_env_var(llm.get("api_key", ""))
        base_url = _resolve_env_var(llm.get("base_url", ""))

        # cognee's litellm path expects 'provider/model' format, but the
        # 'default' sentinel bypasses litellm entirely and routes to
        # DeepSeekAdapter, which sends the model name verbatim to the API.
        if model and "/" not in model and provider != "default":
            model = f"{provider}/{model}"

        self._write_kv(lines, "LLM_PROVIDER", provider, written)
        if model:
            self._write_kv(lines, "LLM_MODEL", str(model), written)
        if api_key:
            self._write_kv(lines, "LLM_API_KEY", str(api_key), written)
        if base_url:
            self._write_kv(lines, "LLM_ENDPOINT", str(base_url), written)
        if llm.get("temperature") is not None:
            self._write_kv(lines, "LLM_TEMPERATURE", str(llm["temperature"]), written)
        if llm.get("max_tokens") is not None:
            self._write_kv(lines, "LLM_MAX_COMPLETION_TOKENS", str(llm["max_tokens"]), written)
        llm_args = llm.get("llm_args")
        if llm_args:
            if isinstance(llm_args, dict):
                import json
                self._write_kv(lines, "LLM_ARGS", json.dumps(llm_args), written)
            else:
                self._write_kv(lines, "LLM_ARGS", str(_resolve_env_var(llm_args)), written)

    def _write_embedding(self, lines: list[str], written: set[str]) -> None:
        emb = self.config.get("embedding", {}) or {}
        if not emb:
            return

        provider = str(_resolve_env_var(emb.get("provider", "openai_compatible")))
        model = _resolve_env_var(emb.get("model", ""))
        api_key = _resolve_env_var(emb.get("api_key", ""))
        base_url = _resolve_env_var(emb.get("base_url", ""))
        dims = emb.get("dimensions")
        tokenizer = emb.get("huggingface_tokenizer")

        self._write_kv(lines, "EMBEDDING_PROVIDER", provider, written)
        if model:
            self._write_kv(lines, "EMBEDDING_MODEL", str(model), written)
        if api_key:
            self._write_kv(lines, "EMBEDDING_API_KEY", str(api_key), written)
        if base_url:
            self._write_kv(lines, "EMBEDDING_ENDPOINT", str(base_url), written)
        if dims is not None:
            self._write_kv(lines, "EMBEDDING_DIMENSIONS", str(dims), written)
        if tokenizer:
            self._write_kv(lines, "HUGGINGFACE_TOKENIZER", str(_resolve_env_var(tokenizer)), written)

    def _write_paths(self, lines: list[str], written: set[str]) -> None:
        data_root = self.config.get("data_root_directory")
        if data_root:
            resolved = _absolutize_path(data_root, project_root=self.project_root)
            self._write_kv(lines, "DATA_ROOT_DIRECTORY", resolved, written)
        sys_root = self.config.get("system_root_directory")
        if sys_root:
            resolved = _absolutize_path(sys_root, project_root=self.project_root)
            self._write_kv(lines, "SYSTEM_ROOT_DIRECTORY", resolved, written)

    def _write_kv(self, lines: list[str], key: str, value: str,
                  written: set[str]) -> None:
        """Write a KEY=VALUE line, skipping empty values and duplicates."""
        if not value:
            return
        if key in written:
            logger.warning("Duplicate env key skipped: %s", key)
            return
        lines.append(f"{key}={value}")
        written.add(key)
        logger.info("  %s=%s", key, _mask_value(key, value))

    # -- state management ----------------------------------------------------

    async def _prune_state(self) -> None:
        """Wipe cognee graph + vector + raw data + metadata tables.

        Mirrors cognee's eval_framework per-run reset:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        """
        try:
            from cognee.api.v1.prune import prune as cognee_prune
            await cognee_prune.prune_data()
            await cognee_prune.prune_system(metadata=True)
            logger.info("  cognee state wiped via prune_data + prune_system(metadata=True)")
        except Exception as exc:
            logger.error(
                "cognee prune failed: %s — adapter will run on current state",
                exc, exc_info=True,
            )
