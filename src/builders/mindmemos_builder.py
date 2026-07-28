"""
MindMemOS Builder for LifeBench_eval.

Handles environment setup, API key generation, Docker service startup,
and FastAPI server lifecycle for MindMemOS evaluation.
"""

import logging
import os
import secrets
import subprocess
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from src.builders.base_builder import BaseBuilder
from src.builders.registry import register_builder

logger = logging.getLogger(__name__)

# Default algorithm profiles matching the official benchmark config.
# These are merged on top of the server's base config via update_config()
# when the API key is resolved, so only non-default overrides are needed.
_SCHEMA_PROJECT_OVERRIDE = {
    "algo_config": {
        "common": {
            "prompt_language": "EN",
        },
        "add": {
            "schema": {
                "entity_modeling_path": "config/presets/entity_modeling_locomo.json",
                "extraction": {
                    "enable_schema_selection": True,
                    "use_search_fields": True,
                    "search_fields_max": 10,
                    "episode_search_fields_augment": True,
                    "episode_augment_count": 4,
                },
                "merge": {
                    "enable_entity_merge_decision": True,
                    "entity_recall_top_k": 15,
                    "max_merge_retries": 8,
                    "use_property_merge": False,
                    "secondary_search_limit": 30,
                    "secondary_search_retries": 3,
                },
                "higher_order": {
                    "enabled": True,
                    "top_k": 10,
                    "min_evidence_count": 2,
                },
                "episode_edge": {
                    "top_k": 10,
                },
                "chunker": {
                    "split_mode": "llm",
                    "min_episode_length": 1,
                    "max_episode_length": 15,
                    "max_buffer_size": 1000,
                    "split_on_user_speaker": True,
                    "max_minutes_from_first": 30,
                },
                "drain": {
                    "episode_generation_max_retries": 3,
                },
            },
        },
        "search": {
            "request_top_k_max": 100,
            "schema_search": {
                "entity": {
                    "recall_size": 800,
                    "rrf_k": 80,
                    "top_k": 40,
                    "top_n": 16,
                    "use_reranker": True,
                    "max_rerank_candidates": 100,
                    "use_maxsim_rescore": False,
                    "maxsim_weight": 0.3,
                    "search_field_overfetch_factor": 3,
                },
                "property": {
                    "recall_size": 45,
                    "rrf_k": 60,
                    "top_k": 20,
                    "top_n": 16,
                    "alloc_min_factor": 0.5,
                    "alloc_max_factor": 1.5,
                    "use_property_extension": True,
                    "extension_step": 3,
                    "higher_order_ratio": 0.4,
                },
                "dual_path": {
                    "enabled": True,
                    "property_recall_size": 300,
                    "property_rrf_k": 80,
                    "property_top_k": 80,
                    "property_top_n": 25,
                },
                "entity_weights": {
                    "force_balanced_split": True,
                    "episode_weight": 0.7,
                    "non_episode_weight": 0.3,
                },
                "edge": {
                    "top_k": 2,
                    "min_relevance_score": 0.1,
                },
                "multi_hop": 2,
                "use_entity_agent_search": True,
            },
            "agentic": {
                "max_rounds": 3,
                "top_k_per_round": 20,
                "top_n_per_round": 10,
                "num_hops": 2,
                "use_rerank": True,
                "use_relevance_filter": False,
                "use_property_filter": False,
                "current_time_mode": "unknown",
                "min_time_window_days": 30,
                "include_edges": False,
                "output_max_edge_num": 10,
            },
        },
    },
}

_VANILLA_PROJECT_OVERRIDE = {
    "algo_config": {
        "common": {
            "prompt_language": "EN",
        },
        "add": {
            "vanilla": {
                "chunk_soft_token_budget": 26000,
                "chunk_hard_token_budget": 32000,
                "turn_hard_token_budget": 16000,
                "history_soft_token_budget": 2000,
                "history_hard_token_budget": 4000,
                "history_min_turn_count": 1,
                "compaction_soft_token_budget": 16000,
                "compaction_head_tokens": 4000,
                "compaction_tail_tokens": 4000,
                "compaction_summary_context_token_budget": 200000,
                "compaction_summary_output_token_budget": 8000,
                "time_gap_threshold_seconds": 1800,
                "template_tokens": 1000,
                "recall_budget": 2000,
                "output_headroom": 4000,
                "enable_entities": True,
            },
        },
        "search": {
            "request_top_k_max": 100,
            "vanilla": {
                "recall_size": 50,
                "use_reranker": True,
                "graph_enabled": True,
                "graph_seed_memory_limit": 5,
                "graph_related_per_seed": 3,
                "graph_max_candidates": 10,
                "graph_decay": 0.5,
                "graph_score": 0.01,
            },
            "rerank": {
                "max_query_length": 200,
                "max_doc_length": 2000,
            },
        },
    },
}

ALGORITHM_PROFILES: dict[str, dict] = {
    "vanilla": _VANILLA_PROJECT_OVERRIDE,
    "schema": _SCHEMA_PROJECT_OVERRIDE,
}


@register_builder("mindmemos")
class MindMemOSBuilder(BaseBuilder):
    """Builder for MindMemOS memory system."""

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)
        if project_root:
            self.system_dir = Path(project_root) / "systems" / "MindMemOS"
        else:
            self.system_dir = Path(__file__).parent.parent.parent / "systems" / "MindMemOS"
        self.config_name = config.get("config_name", "dev")
        self.memory_algorithm = config.get("memory_algorithm", "vanilla")
        self.docker_compose_file = self.system_dir / "dockers" / "docker-compose.memory.yml"
        self.service_url = config.get("service_url", "http://127.0.0.1:8000")
        self._api_process: Optional[subprocess.Popen] = None
        self._api_owned: bool = False
        self._generated_api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # API key generation
    # ------------------------------------------------------------------

    def _generate_api_keys(self) -> str:
        """Generate API key entries and write them to the server's api_keys.yaml.

        Creates one key for the selected memory_algorithm, with the matching
        project_override_config so the server enables the right processing
        pipelines (vanilla_add vs schema_add, fast vs agentic search, etc.).

        Returns the generated api_key string so the adapter can use it.
        """
        timestamp = datetime.now(timezone.utc)
        suffix = f"{timestamp:%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"
        algorithm = self.memory_algorithm
        benchmark = self.config.get("benchmark_name", "lifebench")

        key_id = f"key_{benchmark}_{algorithm}_{suffix}"
        api_key = f"dev-api-key-{benchmark}-{algorithm}-{suffix}".replace("_", "-")
        project_id = f"proj_{benchmark}_{algorithm}_{suffix}"

        # Allow user-provided overrides to take precedence, otherwise use built-in
        user_override = self.config.get("project_override_config")
        if user_override is not None:
            project_override_config = user_override
        else:
            project_override_config = ALGORITHM_PROFILES.get(algorithm)

        entry = {
            "key_id": key_id,
            "api_key": api_key,
            "project_id": project_id,
            "memory_algorithm": algorithm,
            "enabled": True,
            "scopes": ["memory:read", "memory:write"],
        }
        if project_override_config is not None:
            entry["project_override_config"] = project_override_config

        # Write to the config directory where the server reads it (auth.api_key_file)
        api_keys_path = self.system_dir / "config" / "mindmemos" / "api_keys.yaml"
        api_keys_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing keys so that concurrent / prior keys are preserved.
        existing: dict[str, Any] = {}
        if api_keys_path.exists():
            with api_keys_path.open("r", encoding="utf-8") as fh:
                existing = yaml.safe_load(fh) or {}
        existing_keys: list[dict[str, Any]] = existing.get("api_keys") or []
        # Remove any previous entry with the same api_key to allow idempotent re-runs
        existing_keys = [k for k in existing_keys if k.get("api_key") != api_key]
        existing_keys.append(entry)
        existing["api_keys"] = existing_keys

        text = yaml.safe_dump(existing, sort_keys=False, allow_unicode=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(api_keys_path.parent), prefix=".api_keys.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp_name, str(api_keys_path))
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

        self._generated_api_key = api_key
        logger.info("Generated API keys written to %s", api_keys_path)
        logger.info("  algorithm: %s", algorithm)
        logger.info("  key_id: %s", key_id)
        logger.info("  has project_override_config: %s", project_override_config is not None)

        return api_key

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def setup_environment(self) -> None:
        """Set up environment variables and configuration files for MindMemOS."""
        os.environ["MINDMEMOS_CONFIG_NAME"] = self.config_name
        os.environ["MINDMEMOS_SERVICE_URL"] = self.service_url
        os.environ["MINDMEMOS_MEMORY_ALGORITHM"] = self.memory_algorithm

        # Copy .env and dev.yaml from examples if missing
        env_file = self.system_dir / ".env"
        env_example = self.system_dir / ".env.example"
        if not env_file.exists() and env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)

        config_dir = self.system_dir / "config" / "mindmemos"
        dev_config = config_dir / f"{self.config_name}.yaml"
        dev_example = config_dir / f"{self.config_name}.example.yaml"
        if not dev_config.exists() and dev_example.exists():
            import shutil
            shutil.copy(dev_example, dev_config)

        # Generate API keys and export to env so the adapter picks them up
        api_key = self._generate_api_keys()
        os.environ["MINDMEMOS_API_KEY"] = api_key
        logger.info("  MINDMEMOS_API_KEY env set for adapter")

        # Add MindMemOS src to sys.path
        mindmemos_src = self.system_dir / "src"
        if str(mindmemos_src) not in sys.path:
            sys.path.insert(0, str(mindmemos_src))

        logger.info("MindMemOS environment configured")
        logger.info("  Config: %s", self.config_name)
        logger.info("  Memory algorithm: %s", self.memory_algorithm)
        logger.info("  Service URL: %s", self.service_url)
        logger.info("  System dir: %s", self.system_dir)

    # ------------------------------------------------------------------
    # Docker services
    # ------------------------------------------------------------------

    def _check_docker_service(self, container_name: str) -> bool:
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=mindmemos-" + container_name,
                 "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return f"mindmemos-{container_name}" in result.stdout
        except Exception:
            return False

    def _wait_for_service(self, url: str, timeout: int = 60) -> bool:
        import urllib.request

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                req = urllib.request.Request(url)
                urllib.request.urlopen(req, timeout=5)
                return True
            except Exception:
                time.sleep(2)
        return False

    def start_docker_services(self) -> None:
        logger.info("Starting Docker services for MindMemOS...")

        if not self.docker_compose_file.exists():
            logger.warning("Docker compose file not found: %s", self.docker_compose_file)
            return

        try:
            if (self._check_docker_service("qdrant")
                    and self._check_docker_service("neo4j")
                    and self._check_docker_service("kafka")):
                logger.info("Docker services already running")
            else:
                subprocess.run(
                    [
                        "docker", "compose",
                        "--env-file", str(self.system_dir / ".env"),
                        "-f", str(self.docker_compose_file),
                        "up", "-d", "--wait",
                        "qdrant", "neo4j", "kafka", "kafka-ui", "kafka-exporter",
                    ],
                    cwd=str(self.system_dir),
                    check=True,
                    timeout=120,
                )
                logger.info("Docker services started")

            logger.info("Waiting for Qdrant...")
            qdrant_ready = self._wait_for_service("http://localhost:6333", timeout=30)
            if qdrant_ready:
                logger.info("Qdrant is ready")
            else:
                logger.warning("Qdrant not ready after 30s")

            logger.info("Waiting for Neo4j...")
            neo4j_ready = self._wait_for_service("http://localhost:7474", timeout=30)
            if neo4j_ready:
                logger.info("Neo4j is ready")
            else:
                logger.warning("Neo4j not ready after 30s")

        except subprocess.CalledProcessError as e:
            logger.error("Failed to start Docker services: %s", e)
        except Exception as e:
            logger.error("Error starting Docker services: %s", e)

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------

    def _is_api_running(self) -> bool:
        return self._wait_for_service(f"{self.service_url}/healthz", timeout=3)

    def _find_venv_python(self) -> Optional[Path]:
        for candidate in (
            self.system_dir / ".venv" / "Scripts" / "python.exe",
            self.system_dir / ".venv" / "bin" / "python",
            self.system_dir / ".venv" / "bin" / "python3",
        ):
            if candidate.exists():
                return candidate
        return None

    def start_api_server(self) -> Optional[subprocess.Popen]:
        if self._is_api_running():
            logger.info("MindMemOS API server already running at %s", self.service_url)
            self._api_owned = False
            return None

        logger.info("Starting MindMemOS API server...")
        venv_python = self._find_venv_python()
        if venv_python is None:
            logger.error("MindMemOS .venv not found at %s", self.system_dir / ".venv")
            return None

        env = os.environ.copy()
        env["MINDMEMOS_CONFIG_NAME"] = self.config_name

        import platform
        is_windows = platform.system() == "Windows"

        if is_windows:
            cmd = (
                f'"{venv_python}" -m uvicorn mindmemos.api.app:app '
                f'--host 127.0.0.1 --port 8000'
            )
            process = subprocess.Popen(
                cmd,
                cwd=str(self.system_dir),
                env=env,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            cmd = [
                str(venv_python), "-m", "uvicorn",
                "mindmemos.api.app:app",
                "--host", "127.0.0.1", "--port", "8000",
            ]
            process = subprocess.Popen(
                cmd,
                cwd=str(self.system_dir),
                env=env,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

        logger.info("MindMemOS API server starting...")
        self._api_owned = True

        time.sleep(10)
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            logger.error("API server process died. Output: %s", output)
            return None

        if self._wait_for_service(f"{self.service_url}/healthz", timeout=90):
            logger.info("MindMemOS API ready at %s", self.service_url)
            return process
        else:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                logger.error("API server died during startup. Output: %s", output[:2000])
            else:
                logger.error(
                    "MindMemOS API not ready after 90s. "
                    "Check 'docker ps' and API server logs."
                )
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            return None

    # ------------------------------------------------------------------
    # NLP assets & DB reset
    # ------------------------------------------------------------------

    def _setup_dev_environment(self) -> None:
        venv_python = self._find_venv_python()
        if venv_python is None:
            logger.warning("MindMemOS .venv not found; skipping NLP assets. "
                           "Run 'make dev-setup' manually.")
            return

        nlp_script = self.system_dir / "scripts" / "install_nlp_assets.py"
        if not nlp_script.exists():
            logger.warning("NLP asset script not found: %s", nlp_script)
            return

        logger.info("Installing NLP assets via %s...", nlp_script)
        try:
            result = subprocess.run(
                [str(venv_python), str(nlp_script)],
                cwd=str(self.system_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.warning("NLP asset installation had errors: %s", result.stderr[:500])
            else:
                logger.info("NLP assets installed successfully")
        except subprocess.TimeoutExpired:
            logger.warning("NLP asset installation timed out (5 min)")
        except Exception as e:
            logger.warning("Failed to install NLP assets: %s", e)

    def _reset_databases(self) -> None:
        logger.info("Resetting MindMemOS databases...")
        try:
            import urllib.request

            qdrant_url = os.environ.get("MINDMEMOS_QDRANT_URL", "http://localhost:6333")
            collections = ["memory_item_v1", "entity_item_v1", "source_item_v1"]
            for coll in collections:
                try:
                    req = urllib.request.Request(
                        f"{qdrant_url}/collections/{coll}", method="DELETE"
                    )
                    urllib.request.urlopen(req, timeout=10)
                    logger.debug("  Qdrant collection '%s' deleted", coll)
                except Exception as e:
                    logger.debug("  Qdrant collection '%s' skip: %s", coll, e)
            logger.info("  Qdrant collections reset")
        except Exception as e:
            logger.warning("  Qdrant reset failed (continuing): %s", e)

    # ------------------------------------------------------------------
    # Build / Cleanup
    # ------------------------------------------------------------------

    async def build(self) -> bool:
        logger.info("=" * 60)
        logger.info("Building MindMemOS environment...")
        logger.info("=" * 60)

        # 1. Setup environment (env vars, config files, API keys)
        self.setup_environment()

        # 2. Install NLP assets
        self._setup_dev_environment()

        # 3. Start Docker
        self.start_docker_services()

        logger.info("Checking Docker services...")
        if not self._check_docker_service("qdrant"):
            logger.error("Qdrant is not running. Please fix Docker setup.")
            return False
        if not self._check_docker_service("neo4j"):
            logger.error("Neo4j is not running. Please fix Docker setup.")
            return False
        logger.info("Docker services are running")

        # 4. Start API server
        self._api_process = self.start_api_server()
        if not self._is_api_running():
            logger.error("API server is not running after start attempt")
            return False

        # 5. Reset databases if requested
        if self.config.get("reset_db", False):
            self._reset_databases()

        logger.info("MindMemOS build completed successfully")
        return True

    async def cleanup(self) -> bool:
        logger.info("Cleaning up MindMemOS resources...")
        cleanup_ok = True

        if self._api_owned and self._api_process is not None:
            logger.info("  Stopping API server...")
            try:
                self._api_process.terminate()
                try:
                    self._api_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._api_process.kill()
                    self._api_process.wait(timeout=5)
                logger.info("  API server stopped")
            except Exception as e:
                logger.warning("  Failed to stop API server: %s", e)
                cleanup_ok = False
            self._api_process = None

        if self.config.get("stop_docker_on_cleanup", False):
            if self.docker_compose_file.exists():
                logger.info("  Stopping Docker services...")
                try:
                    subprocess.run(
                        ["docker", "compose", "-f", str(self.docker_compose_file), "down"],
                        cwd=str(self.system_dir),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    logger.info("  Docker services stopped")
                except Exception as e:
                    logger.warning("  Failed to stop Docker services: %s", e)
                    cleanup_ok = False
        else:
            logger.info("  Docker services left running (set stop_docker_on_cleanup=true to stop)")

        return cleanup_ok
