"""
Mem0 builder - starts Mem0 OSS server via docker compose.
"""

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder


# Provider-specific config-key mapping. mem0ai's provider config classes expect
# a provider-named base_url field (deepseek_base_url, openai_base_url, ...).
# We let users write a generic "base_url" in YAML and translate it here.
_BASE_URL_KEYS = {
    "deepseek": "deepseek_base_url",
    "openai": "openai_base_url",
    "anthropic": "anthropic_base_url",
    "gemini": "gemini_base_url",
}

_ENV_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


def _resolve_env(value: Any) -> Any:
    """Expand ${VAR} and ${VAR:default} references in string values."""
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else m.group(0))
        return _ENV_REF.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _map_config(provider: str, side_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a generic runtime_config block into a provider-specific one.

    Generic YAML shape:
        provider: "deepseek"
        model: "..."
        api_key: "..."
        base_url: "..."
        temperature: 0.1
        max_tokens: 32768

    Provider-specific shape (sent to mem0 /configure):
        model: "..."
        api_key: "..."
        deepseek_base_url: "..."   # field renamed by provider
        temperature: 0.1
        max_tokens: 32768
    """
    resolved = _resolve_env(side_cfg)
    base_url_key = _BASE_URL_KEYS.get(provider)
    out: Dict[str, Any] = {}
    for k, v in resolved.items():
        if k == "provider":
            continue
        if k == "base_url" and base_url_key:
            out[base_url_key] = v
        else:
            out[k] = v
    return out


def _summarize(block: Optional[Dict[str, Any]]) -> str:
    """One-line summary for logging: 'provider@base_url model=...'"""
    if not block:
        return "(unchanged)"
    provider = block.get("provider", "?")
    cfg = block.get("config", {}) or {}
    base_url = (
        cfg.get("deepseek_base_url")
        or cfg.get("openai_base_url")
        or cfg.get("anthropic_base_url")
        or cfg.get("gemini_base_url")
        or "?"
    )
    model = cfg.get("model", "?")
    return f"{provider}@{base_url} model={model}"

logger = logging.getLogger(__name__)


def _load_env_config(project_env: Path) -> dict:
    """Load configuration from project .env file.

    Args:
        project_env: Path to LifeBench_eval/.env

    Returns:
        Dict with api_key, base_url, model, vectorize_api_key, embedding settings
    """
    config = {}
    if project_env.exists():
        with open(project_env, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value  # Force override to ensure .env values are used

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


@register_builder("mem0")
class Mem0Builder(BaseBuilder):
    """Mem0 builder that starts/stops Mem0 OSS via docker compose.

    Configuration:
        docker_compose: Path to docker-compose.yaml (relative to project root)
        env_template: Path to .env.example template
        env_defaults: Dict of env var defaults to set
        docker_wait: Seconds to wait after starting services (default 30)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        self.docker_compose = config.get("docker_compose")
        self.env_defaults = config.get("env_defaults", {})
        self.docker_wait = config.get("docker_wait", 30)
        self.post_start_script = config.get("post_start_script")
        self._started = False

    async def build(self) -> bool:
        """
        Start Mem0 OSS server via docker compose.

        Returns:
            True if successful or already running, False on failure
        """
        if self._started:
            logger.info("Mem0 builder already started")
            return True

        project_root = Path(self.project_root) if self.project_root else self._get_project_root()
        if not project_root:
            logger.error("Cannot determine project root")
            return False

        # 1. Prepare .env file
        if not await self._prepare_env_file(project_root):
            logger.error("Failed to prepare .env file")
            return False

        # 2. Start docker services
        if self.docker_compose:
            if not await self._start_docker(project_root):
                return False
            # Wait for services to be ready
            logger.info(f"Waiting {self.docker_wait}s for services to be ready...")
            time.sleep(self.docker_wait)

        # 3. Fix pgvector table dimension mismatch from prior runs.
        # The pgvector volume persists across restarts. If a previous run
        # created the table with a different embedding size (e.g. 1536),
        # inserts will fail with "expected N dimensions, not M". We drop
        # the table so it is recreated with the correct dims on first add.
        if not await self._fix_pgvector_dims(project_root):
            logger.warning("Failed to fix pgvector dimensions; insert may fail")

        # 4. Push runtime LLM/embedder config to mem0 server via POST /configure.
        # The server's DEFAULT_CONFIG hardcodes provider=openai pointing at
        # api.openai.com (unreachable from most networks); override with
        # the user-defined providers from runtime_config.
        if not await self._configure_server(project_root):
            logger.warning(
                "Failed to push runtime LLM/embedder config; the server's "
                "default OpenAI config will likely fail at first add()."
            )

        # 5. Run post-start script
        if self.post_start_script:
            if not await self._run_post_start(project_root):
                logger.warning("Post-start script failed, continuing anyway")

        self._started = True
        logger.info("Mem0 builder completed successfully")
        return True

    async def cleanup(self) -> bool:
        """
        Stop Mem0 OSS server via docker compose.

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("Mem0 builder not started, nothing to cleanup")
            return True

        project_root = Path(self.project_root) if self.project_root else self._get_project_root()
        if not project_root or not self.docker_compose:
            return True

        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.warning(f"Docker compose file not found: {compose_path}")
            return True

        try:
            logger.info(f"Stopping docker services: {self.docker_compose}")
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "down"],
                cwd=str(compose_path.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Docker stop failed: {result.stderr}")
                return False
            logger.info("Docker services stopped")
            self._started = False
            return True
        except Exception as e:
            logger.error(f"Failed to stop docker services: {e}")
            return False

    def _get_project_root(self) -> Optional[Path]:
        """Get project root from project_root config or cli.py location."""
        if self.project_root:
            return Path(self.project_root)
        # Try to infer from cli.py location
        cli_path = Path(__file__).parent.parent.parent / "cli.py"
        if cli_path.exists():
            return cli_path.parent.resolve()
        return None

    async def _prepare_env_file(self, project_root: Path) -> bool:
        """Prepare .env file from LifeBench_eval/.env configuration."""
        compose_path = project_root / self.docker_compose
        env_path = compose_path.parent / ".env"

        # Load config from LifeBench_eval/.env
        project_env = project_root / ".env"
        env_config = _load_env_config(project_env)

        # DashScope API key from environment
        dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")

        # Default to SiliconFlow embedding, use DashScope if explicitly disabled
        use_dashscope_embedding = os.environ.get("MEM0_USE_DASHSCOPE_EMBEDDING", "false").lower() == "true"

        if use_dashscope_embedding:
            if not dashscope_api_key:
                logger.error("DASHSCOPE_API_KEY environment variable is not set")
                raise ValueError("DASHSCOPE_API_KEY is required when MEM0_USE_DASHSCOPE_EMBEDDING=true")
            # DashScope embedding
            embedding_config = f"""# DashScope Embedding settings
DASHSCOPE_API_KEY={dashscope_api_key}
DASHSCOPE_EMBEDDING_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3"""
            logger.info(f"  Embedding: DashScope text-embedding-v3 @ dashscope.aliyuncs.com")
        else:
            # SiliconFlow embedding (default, OpenAI-compatible API)
            embedding_config = f"""# SiliconFlow Embedding settings (OpenAI-compatible)
DASHSCOPE_API_KEY={env_config['vectorize_api_key']}
DASHSCOPE_EMBEDDING_URL={env_config['vectorize_base_url']}
DASHSCOPE_EMBEDDING_MODEL={env_config['vectorize_model']}"""
            logger.info(f"  Embedding: SiliconFlow {env_config['vectorize_model']} @ {env_config['vectorize_base_url']}")

        # Resolve the desired embedding vector size. Default to whatever the
        # framework .env says (VECTORIZE_DIMENSIONS), falling back to 1536.
        # This gets injected into mem0 server's DEFAULT_CONFIG so the pgvector
        # table is sized correctly on first startup.
        embed_dims = int(env_config.get('vectorize_dimensions') or
                         os.environ.get('MEM0_EMBEDDING_DIMS', '1536'))

        env_content = f"""# Mem0 Server Environment Configuration
# Generated by Mem0Builder from LifeBench_eval/.env

# DeepSeek LLM settings (native provider)
DEEPSEEK_API_KEY={env_config['llm_api_key']}
DEEPSEEK_API_BASE={env_config['llm_base_url']}

# The server's DEFAULT_CONFIG (server/main.py) hardcodes the embedder to the
# "openai" provider. To reuse the SiliconFlow OpenAI-compatible endpoint we
# already configured for the framework, we point OPENAI_API_KEY/BASE at it.
# Same trick for the LLM.
OPENAI_API_KEY={env_config['vectorize_api_key']}
OPENAI_API_BASE={env_config['vectorize_base_url']}

{embedding_config}

# Vector store: the server uses pgvector via the postgres container
# (DEFAULT_CONFIG in server/main.py). No standalone Qdrant service is started;
# the previous QDRANT_HOST=qdrant block was a leftover that caused
# "no such service: qdrant" on docker compose up.

# PostgreSQL settings (for app data)
POSTGRES_USER=postgres
POSTGRES_DB=postgres
POSTGRES_PASSWORD=mem0dev

# Application database
APP_DB_NAME=mem0_app

# Auth disabled for development
AUTH_DISABLED=true

# Default models
MEM0_DEFAULT_LLM_MODEL={env_config['llm_model']}
MEM0_DEFAULT_EMBEDDER_MODEL={env_config['vectorize_model']}

# Embedding dimensions — controls pgvector table size at startup.
# Must match the dims the embedder returns (set via runtime_config too).
MEM0_EMBEDDING_DIMS={embed_dims}

# Security settings
JWT_SECRET=test-secret-key-for-dev
ADMIN_API_KEY=admin123

# Telemetry
MEM0_TELEMETRY=true

# Reranker settings (SiliconFlow rerank API - Cohere-compatible)
MEM0_RERANKER_ENABLED=true
MEM0_RERANKER_PROVIDER=cohere
MEM0_RERANKER_MODEL={env_config['rerank_model']}
MEM0_RERANKER_API_KEY={env_config['rerank_api_key']}
MEM0_RERANKER_BASE_URL={env_config['rerank_base_url']}
MEM0_RERANKER_TOP_K=10
"""
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info(f"Created .env file: {env_path}")
            logger.info(f"  LLM: {env_config['llm_model']} @ {env_config['llm_base_url']}")
            logger.info(f"  Reranker: {env_config['rerank_model']} @ {env_config['rerank_base_url']}")
            return True
        except Exception as e:
            logger.error(f"Failed to create .env file: {e}")
            return False

    async def _start_docker(self, project_root: Path) -> bool:
        """Start docker compose services, checking each one individually."""
        compose_path = project_root / self.docker_compose
        if not compose_path.exists():
            logger.error(f"Docker compose file not found: {compose_path}")
            return False

        try:
            # Start postgres first (needed by mem0)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "postgres"],
                capture_output=True,
                text=True,
            )
            if "Up" not in result.stdout:
                logger.info(f"Starting postgres: {self.docker_compose}")
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "up", "-d", "postgres"],
                    cwd=str(compose_path.parent),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error(f"Docker start postgres failed: {result.stderr}")
                    return False

                # Wait for postgres to be ready
                if not await self._wait_for_postgres(project_root):
                    logger.error("Postgres did not become ready in time")
                    return False
            else:
                logger.info("Postgres already running")

            # Start mem0 main app (depends on postgres; pgvector runs inside it)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "mem0"],
                capture_output=True,
                text=True,
            )
            if "Up" not in result.stdout:
                logger.info(f"Starting mem0: {self.docker_compose}")
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "up", "-d", "mem0"],
                    cwd=str(compose_path.parent),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error(f"Docker start mem0 failed: {result.stderr}")
                    return False
                logger.info("Mem0 started")
            else:
                logger.info("Mem0 already running")

            # Start mem0-dashboard (optional UI)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "ps", "mem0-dashboard"],
                capture_output=True,
                text=True,
            )
            if "Up" not in result.stdout:
                logger.info(f"Starting mem0-dashboard: {self.docker_compose}")
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "up", "-d", "mem0-dashboard"],
                    cwd=str(compose_path.parent),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logger.error(f"Docker start mem0-dashboard failed: {result.stderr}")
                    return False
                logger.info("Mem0-dashboard started")
            else:
                logger.info("Mem0-dashboard already running")

            logger.info("Docker services ready")
            return True

        except Exception as e:
            logger.error(f"Failed to start docker services: {e}")
            return False

    async def _wait_for_postgres(self, project_root: Path) -> bool:
        """Wait for postgres container to be healthy."""
        compose_path = project_root / self.docker_compose
        max_retries = 90  # 90 * 2s = 180s
        for attempt in range(max_retries):
            try:
                # Use docker compose ps to check postgres status (works regardless of actual container name)
                result = subprocess.run(
                    ["docker", "compose", "-f", str(compose_path), "ps", "postgres"],
                    capture_output=True,
                    text=True,
                )
                # Check if postgres service is "Up" in the output
                if "Up" in result.stdout or "healthy" in result.stdout.lower():
                    logger.info("Postgres is healthy")
                    return True
                logger.debug(f"Postgres status: {result.stdout.strip()}, attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                logger.debug(f"Postgres check attempt {attempt + 1}/{max_retries}: {e}")

            if attempt < max_retries - 1:
                time.sleep(2)

        logger.error("Postgres did not become healthy in 180s")
        return False

    async def _fix_pgvector_dims(self, project_root: Path) -> bool:
        """Drop the pgvector table if its column dimension doesn't match the
        configured ``MEM0_EMBEDDING_DIMS``. The table is recreated lazily by
        mem0 on first insert with the correct size.

        Without this fix, a volume that was populated by an earlier run with
        a different embedding size will cause ``psycopg.errors.DataException:
        expected N dimensions, not M`` on every insert.
        """
        compose_path = project_root / self.docker_compose
        compose_parent = compose_path.parent

        # Read the expected dims from the .env we just wrote (or from os.environ).
        env_path = compose_parent / ".env"
        expected_dims: Optional[int] = None
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MEM0_EMBEDDING_DIMS="):
                    try:
                        expected_dims = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                    break
        if expected_dims is None:
            expected_dims = int(os.environ.get("MEM0_EMBEDDING_DIMS", "0"))
        if expected_dims <= 0:
            logger.info("MEM0_EMBEDDING_DIMS not set; skipping pgvector dim check")
            return True

        pg_user = os.environ.get("POSTGRES_USER", "postgres")
        pg_db = os.environ.get("POSTGRES_DB", "postgres")
        psql = [
            "docker", "compose", "-f", str(compose_path),
            "exec", "-T", "postgres",
            "psql", "-U", pg_user, "-d", pg_db,
            "-c",
        ]

        # Check if the memories table exists and what its vector dim is.
        try:
            result = subprocess.run(
                psql + [
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = 'memories'::regclass AND attname = 'vector';"
                ],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                if "does not exist" in (result.stderr or ""):
                    logger.info("pgvector table does not exist yet; no fix needed")
                    return True
                logger.warning("pgvector dim check failed: %s", result.stderr)
                return True  # not fatal — proceed and let insert fail if needed

            dim_str = ""
            for line in result.stdout.strip().split("\n"):
                stripped = line.strip()
                if stripped.isdigit():
                    dim_str = stripped
                    break
            if not dim_str:
                logger.warning("Could not parse vector dim from: %r", result.stdout)
                return True

            actual_dims = int(dim_str)
            if actual_dims == expected_dims:
                logger.info(
                    "pgvector table dims match: expected=%d actual=%d",
                    expected_dims, actual_dims,
                )
                return True

            logger.warning(
                "pgvector dimension mismatch: expected=%d actual=%d. "
                "Dropping table so it is recreated correctly.",
                expected_dims, actual_dims,
            )
            drop = subprocess.run(
                psql + ["DROP TABLE IF EXISTS memories CASCADE;"],
                capture_output=True, text=True, timeout=15,
            )
            if drop.returncode != 0:
                logger.error("Failed to drop memories table: %s", drop.stderr)
                return False
            logger.info("Dropped stale pgvector table; will be recreated on first insert")
            return True

        except Exception as exc:
            logger.warning("pgvector dim check error (non-fatal): %s", exc)
            return True

    async def _configure_server(self, project_root: Path) -> bool:
        """Push runtime LLM/embedder config to the mem0 server via POST /configure.

        Why this is needed:
          - The server's DEFAULT_CONFIG (server/main.py) pins the LLM to the
            "openai" provider pointing at api.openai.com, which is unreachable
            from most networks → first /memories call times out → 502.
          - DeepSeek's native provider is registered in mem0ai's factory but
            was NOT in the server's BUNDLED_LLM_PROVIDERS allowlist; we added
            "deepseek" to that tuple in server/main.py so /configure accepts it.

        Auth: /configure requires admin. The admin path accepts X-API-Key
        matching ADMIN_API_KEY (Bearer JWT would 401).
        """
        import aiohttp

        env_path = project_root / self.docker_compose
        env_path = env_path.parent / ".env"

        # Read ADMIN_API_KEY that _prepare_env_file just wrote
        admin_api_key = os.environ.get("MEM0_ADMIN_API_KEY", "")
        if not admin_api_key and env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("ADMIN_API_KEY="):
                    admin_api_key = line.split("=", 1)[1].strip()
                    break

        if not admin_api_key:
            logger.warning("ADMIN_API_KEY not found; skipping /configure push")
            return False

        # Re-load LifeBench_eval/.env into os.environ so the payload below
        # picks up the latest credentials.
        _load_env_config(project_root / ".env")

        host = os.environ.get("MEM0_HOST", "http://localhost:8888")
        cfg_url = f"{host.rstrip('/')}/configure"

        # Build payload from the system config's `runtime_config` block so that
        # LLM and embedder can each pick their own provider / model / base_url /
        # api_key independently. Field names like `base_url` map to provider-
        # specific keys (deepseek_base_url / openai_base_url) — see _map_config.
        payload: Dict[str, Any] = {}
        rc = self.config.get("runtime_config", {})
        for side in ("llm", "embedder"):
            side_cfg = rc.get(side)
            if not side_cfg:
                continue
            provider = side_cfg.get("provider", "openai")
            config_block = _map_config(provider, side_cfg)
            payload[side] = {"provider": provider, "config": config_block}

        if not payload:
            logger.info("runtime_config not set; keeping server DEFAULT_CONFIG")
            return True

        # Wait for the server to be ready. The mem0 server has no /health
        # endpoint, so we poll /configure (admin-gated) until it answers 200.
        # Probing /health alone is misleading because it 404s immediately
        # even before the app finishes initializing, which leads to a race
        # where the first /memories request hits a half-warm server.
        deadline = time.monotonic() + 60
        ready = False
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                try:
                    async with session.get(
                        f"{host.rstrip('/')}/configure",
                        headers={"X-API-Key": admin_api_key},
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r:
                        if r.status == 200:
                            await r.read()
                            ready = True
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(1)
            if not ready:
                logger.error(
                    "Timed out waiting for mem0 /configure to respond; "
                    "server may not be fully initialized."
                )
                return False

            headers = {"X-API-Key": admin_api_key, "Content-Type": "application/json"}
            try:
                async with session.post(
                    cfg_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as r:
                    body = await r.text()
                    if r.status == 200:
                        llm_summary = _summarize(payload.get("llm"))
                        emb_summary = _summarize(payload.get("embedder"))
                        logger.info(
                            "Pushed runtime LLM/embedder config: "
                            f"LLM={llm_summary}, embedder={emb_summary}"
                        )
                        return True
                    logger.error(f"POST {cfg_url} -> {r.status}: {body[:300]}")
                    return False
            except Exception as exc:
                logger.error(f"POST {cfg_url} failed: {exc}")
                return False

    async def _run_post_start(self, project_root: Path) -> bool:
        """Run post-start script."""
        script_path = project_root / self.post_start_script
        if not script_path.exists():
            logger.warning(f"Post-start script not found: {script_path}")
            return False

        try:
            logger.info(f"Running post-start script: {self.post_start_script}")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(f"Post-start script failed: {result.stderr}")
                return False
            logger.info("Post-start script completed")
            return True
        except Exception as e:
            logger.warning(f"Failed to run post-start script: {e}")
            return False

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "docker_compose": self.docker_compose,
        }