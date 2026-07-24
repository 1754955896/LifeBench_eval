"""
Cognee builder — prepares environment for in-process cognee SDK usage.

No Docker required. Mirrors the `hindsight_builder.py` pattern at the surface
level (build/cleanup lifecycle), but strips away anything that mediates state
between builder and adapter.

Responsibilities kept here:
    1. Set OS env vars (LLM_API_KEY, EMBEDDING_*, ENABLE_BACKEND_ACCESS_CONTROL,
       DATA_ROOT_DIRECTORY, ...) BEFORE the cognee module is touched. cognee's
       own __init__.py calls `dotenv.load_dotenv(override=True)`, which would
       otherwise clobber our env. Setting them in build() — which runs before
       the adapter is invoked — buys us a deterministic baseline.
    2. Pre-import cognee to surface import-time errors (e.g. missing dependency)
       before the eval run starts. Also forces DB migrations to be discovered
       early.

Responsibilities deliberately NOT here:
    - Building `LLMConfig` / `EmbeddingConfig` instances and stashing them on
      `self.config`. Those are lightweight pydantic-settings objects with no
      resources to share; the adapter constructs them itself from the same
      config dict. (See `CogneeAdapter._build_llm_config`.)

Reference: LifeBench_eval/systems/cognee/cognee/eval_framework/...
"""
import logging
import os
import re
from typing import Any, Dict, Optional
from pathlib import Path

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder

logger = logging.getLogger(__name__)


def _absolutize_path(value: str, project_root: Optional[str]) -> str:
    """Resolve relative paths to absolute against `project_root`.

    cognee's BaseConfig validates `data_root_directory` and
    `system_root_directory` as absolute paths. YAML configs typically
    express them as relative (`.cognee/data`), which Windows rejects as
    "Got relative path". We absolutize here so the values written into
    os.environ by the builder pass cognee's validation regardless of OS.

    S3 URLs, already-absolute paths, and non-string inputs pass through
    unchanged.
    """
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
    """Resolve ${VAR:default} style env-var references (mirrors hindsight_builder)."""
    if not isinstance(value, str):
        return value
    pattern = r'\$\{([^}:]+)(?::([^}]*))?}'

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    return re.sub(pattern, replacer, value)


@register_builder("cognee")
class CogneeBuilder(BaseBuilder):
    """In-process cognee SDK builder.

    Configuration (all optional — fall back to cognee's defaults / .env values):
        llm:                    dict with provider, model, api_key, base_url,
                                temperature, max_tokens — used as answer-LLM.
                                Also exported as OS env vars for any cognee
                                call site that doesn't go through our adapter.
        memory_llm:             dict (same shape) used during cognify for
                                entity extraction / summarization.
        embedding:              dict with provider, model, api_key, base_url,
                                dimensions, batch_size, huggingface_tokenizer.
        rerank:                 dict — logged but ignored (cognee has no
                                rerank in its eval_framework).
        data_root_directory:    cognee data path (DATA_ROOT_DIRECTORY env var).
        system_root_directory:  cognee system path (SYSTEM_ROOT_DIRECTORY).
        enable_backend_access_control: bool (default False — matches BEAM).
        structured_output_framework: str (default "instructor").

    Side effects on success:
        Sets OS env vars for the keys above so cognee's pydantic-settings
        can find them at import or first-call time.
        Tries `import cognee`; logs and proceeds on failure (the adapter
        will also fail loudly when actually used, but the build step is
        not blocked).
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self._started = False
        self._cognee_imported = False

    # ----- BaseBuilder contract ------------------------------------------------

    async def build(self) -> bool:
        if self._started:
            logger.info("Cognee builder already started")
            return True

        env_mappings = self._build_env_mappings()
        self._apply_env(env_mappings)

        # Pre-import cognee so import-time errors surface before the eval run.
        # Note: cognee's __init__.py calls dotenv.load_dotenv(override=True)
        # which would overwrite the env vars we just set if there is a .env
        # file in cwd. We accept that — the adapter passes per-call
        # llm_config/embedding_config overrides so its behavior does not
        # depend on the post-dotenv state.
        # Lifecycle step — wipe cognee state if explicitly requested.
        # Lives in the builder (not the adapter) because it's a run-level
        # concern, not a per-call concern. Matches BEAM's pre-run reset:
        #   cognee.prune.prune_data() + cognee.prune.prune_system(metadata=True)
        if self.config.get("prune_on_init", False):
            await self._prune_state()

        self._started = True
        logger.info(
            "Cognee builder completed (in-process mode, no Docker). "
            "cognee_available=%s, prune_on_init=%s",
            self._cognee_imported,
            self.config.get("prune_on_init", False),
        )
        return True

    async def _prune_state(self) -> None:
        """Wipe cognee's graph + vector + raw data + metadata tables.

        Mirrors cognee's eval_framework/corpus_builder_executor.py:59-60 pattern:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        Failure here is logged but does not abort `build()` — the adapter can
        still run on top of any partially-cleaned or pre-existing state.
        """
        try:
            import cognee
            from cognee.api.v1.prune import prune as cognee_prune
            await cognee_prune.prune_data()
            await cognee_prune.prune_system(metadata=True)
            logger.info("  cognee state wiped via prune_data + prune_system(metadata=True)")
        except Exception as exc:
            logger.error(
                "cognee prune failed: %s — adapter will run on whatever state "
                "cognee is currently in", exc, exc_info=True,
            )

    async def cleanup(self) -> bool:
        if not self._started:
            return True
        self._started = False
        self._cognee_imported = False
        logger.info("Cognee builder cleanup completed")
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "mode": "in_process",
            "cognee_available": self._cognee_imported,
        }

    # ----- env-var translation -------------------------------------------------

    def _build_env_mappings(self) -> Dict[str, str]:
        """Translate harness-level config dict into OS env-var mappings that
        cognee's pydantic-settings LLMConfig / EmbeddingConfig read at import."""
        mappings: Dict[str, str] = {}

        llm_cfg = self.config.get("llm", {}) or {}
        if llm_cfg:
            provider = _resolve_env_var(llm_cfg.get("provider") or "openai")
            model = _resolve_env_var(llm_cfg.get("model") or "")
            api_key = _resolve_env_var(llm_cfg.get("api_key") or "")
            base_url = _resolve_env_var(llm_cfg.get("base_url") or "")
            mappings["LLM_PROVIDER"] = str(provider)
            if model:
                # cognee expects 'provider/model' format
                if "/" not in model:
                    model = f"{provider}/{model}"
                mappings["LLM_MODEL"] = model
            if api_key:
                mappings["LLM_API_KEY"] = str(api_key)
            if base_url:
                mappings["LLM_ENDPOINT"] = str(base_url)
            if llm_cfg.get("temperature") is not None:
                mappings["LLM_TEMPERATURE"] = str(llm_cfg.get("temperature"))
            if llm_cfg.get("max_tokens") is not None:
                mappings["LLM_MAX_COMPLETION_TOKENS"] = str(llm_cfg.get("max_tokens"))

        emb_cfg = self.config.get("embedding", {}) or {}
        if emb_cfg:
            provider = _resolve_env_var(emb_cfg.get("provider") or "openai_compatible")
            model = _resolve_env_var(emb_cfg.get("model") or "")
            api_key = _resolve_env_var(emb_cfg.get("api_key") or "")
            base_url = _resolve_env_var(emb_cfg.get("base_url") or "")
            dims = emb_cfg.get("dimensions")
            mappings["EMBEDDING_PROVIDER"] = str(provider)
            if model:
                mappings["EMBEDDING_MODEL"] = str(model)
            if api_key:
                mappings["EMBEDDING_API_KEY"] = str(api_key)
            if base_url:
                mappings["EMBEDDING_ENDPOINT"] = str(base_url)
            if dims is not None:
                mappings["EMBEDDING_DIMENSIONS"] = str(dims)
            tokenizer = emb_cfg.get("huggingface_tokenizer")
            if tokenizer:
                mappings["HUGGINGFACE_TOKENIZER"] = str(_resolve_env_var(tokenizer))

        if self.config.get("rerank"):
            logger.warning(
                "cognee eval_framework does not implement a rerank model. "
                "The 'rerank' config block is recorded but ignored."
            )

        data_root = self.config.get("data_root_directory")
        if data_root:
            # Resolve relative paths against this builder's project root so
            # cognee's BaseConfig validation (`Path must be absolute`) passes
            # on Windows where dotted paths like `.cognee/data` are not
            # absolute. S3 URLs and already-absolute paths pass through
            # untouched.
            mappings["DATA_ROOT_DIRECTORY"] = _absolutize_path(
                data_root, project_root=self.project_root,
            )
        sys_root = self.config.get("system_root_directory")
        if sys_root:
            mappings["SYSTEM_ROOT_DIRECTORY"] = _absolutize_path(
                sys_root, project_root=self.project_root,
            )

        # Toggle access control off by default (BEAM posture).
        mappings.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
        mappings.setdefault(
            "STRUCTURED_OUTPUT_FRAMEWORK",
            str(self.config.get("structured_output_framework", "instructor")),
        )

        # Instructor mode and litellm extra_body knobs (`llm_instructor_mode`,
        # `llm_args`) used to live here as workarounds for DeepSeek-style
        # reasoning models fighting cognee's litellm/instructor path. They
        # were removed when we switched `llm.provider` to `default` (sentinel
        # that routes through our custom DeepSeek adapter instead). The custom
        # adapter auto-injects `extra_body: {thinking: {type: disabled}}` and
        # uses native `response_format: json_schema` — no env knobs needed.
        # cf. cognee/infrastructure/llm/.../deepseek/adapter.py.

        # Co-name alias bridge. The harness `.env` historically uses
        # `LLM_BASE_URL` / `VECTORIZE_BASE_URL` / `RERANK_BASE_URL`, while
        # cognee's LLMConfig and EmbeddingConfig read `LLM_ENDPOINT` /
        # `EMBEDDING_ENDPOINT`. Propagate between them so cognee's pydantic-
        # settings actually see the value when the user only set `*_BASE_URL`
        # in `.env`. We do it here (instead of touching cli.py) so the alias
        # bridge is contained in the cognee-specific builder.
        llm_url = mappings.get("LLM_ENDPOINT") or os.environ.get("LLM_BASE_URL")
        if llm_url:
            mappings.setdefault("LLM_ENDPOINT", llm_url)
            os.environ.setdefault("LLM_BASE_URL", llm_url)
        emb_url = mappings.get("EMBEDDING_ENDPOINT") or os.environ.get("VECTORIZE_BASE_URL")
        if emb_url:
            mappings.setdefault("EMBEDDING_ENDPOINT", emb_url)
            os.environ.setdefault("VECTORIZE_BASE_URL", emb_url)

        return mappings

    def _apply_env(self, mappings: Dict[str, str]) -> None:
        for key, value in mappings.items():
            if value in (None, ""):
                continue
            os.environ[key] = value
            if "API_KEY" in key:
                masked = value[:8] + "..." if len(value) > 8 else "***"
                logger.info("  %s=%s", key, masked)
            else:
                logger.info("  %s=%s", key, value)
