"""
GraphRAG Builder - validates AstraDB connection for GraphRAG adapter.

Note: AstraDB is a managed cloud service, so no local containers need to be started.
The builder validates configuration and tests connectivity.
"""
import logging
import os
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
        Dict with api_endpoint, token settings
    """
    config = {}
    if project_env.exists():
        with open(project_env, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)

    config["api_endpoint"] = os.environ.get("ASTRA_DB_API_ENDPOINT", "")
    config["token"] = os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "")

    return config


@register_builder("graphrag")
class GraphRAGBuilder(BaseBuilder):
    """GraphRAG builder that validates AstraDB connection.

    Configuration:
        api_endpoint: AstraDB API endpoint URL
        token: AstraDB application token
        collection_name: Collection name to use (default: lifebench_graphrag)
        embedding_model: Embedding model (default: thenlper/gte-large-zh)
        cleanup_collection: Whether to cleanup collection on exit (default: False)
    """

    def __init__(self, config: dict, project_root: Optional[str] = None):
        super().__init__(config, project_root)

        # AstraDB connection settings
        self.api_endpoint = config.get("api_endpoint", "")
        self.token = config.get("token", "")
        self.collection_name = config.get("collection_name", "lifebench_graphrag")

        # Embedding settings (nested config: embedding.model)
        embed_cfg = config.get("embedding", {})
        self.embedding_model = embed_cfg.get("model", "thenlper/gte-large-zh")

        # Retrieval settings
        retrieval_cfg = config.get("retrieval", {})
        self.edges = retrieval_cfg.get("edges", [("habitat", "habitat")])

        # Cleanup settings
        self.cleanup_collection = config.get("cleanup_collection", False)

        self._started = False

    async def build(self) -> bool:
        """
        Validate AstraDB connection and initialize collection.

        Returns:
            True if successful or already validated, False on failure
        """
        if self._started:
            logger.info("GraphRAG builder already validated")
            return True

        # Load env config from project .env
        project_root = Path(self.project_root) if self.project_root else self._get_project_root()
        if project_root:
            project_env = project_root / ".env"
            env_config = _load_env_config(project_env)
            # Override with env config if not already set
            if not self.api_endpoint and env_config.get("api_endpoint"):
                self.api_endpoint = env_config["api_endpoint"]
            if not self.token and env_config.get("token"):
                self.token = env_config["token"]

        # Validate required config
        if not self.api_endpoint:
            logger.error("ASTRA_DB_API_ENDPOINT not configured")
            return False
        if not self.token:
            logger.error("ASTRA_DB_APPLICATION_TOKEN not configured")
            return False

        # Test connection
        if not await self._validate_connection():
            logger.error("Failed to validate AstraDB connection")
            return False

        self._started = True
        logger.info("GraphRAG builder validated successfully")
        return True

    async def cleanup(self) -> bool:
        """
        Cleanup GraphRAG resources.

        Args:
            cleanup_collection: If True, delete the collection

        Returns:
            True if successful
        """
        if not self._started:
            logger.info("GraphRAG builder not started, nothing to cleanup")
            return True

        try:
            if self.cleanup_collection:
                logger.info(f"Would cleanup collection: {self.collection_name}")
                # Note: Actual collection deletion would require AstraDB admin client
                # For now, we just log the intent

            self._started = False
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")
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

    async def _validate_connection(self) -> bool:
        """Validate AstraDB connection by testing API access."""
        try:
            from langchain_astradb import AstraDBVectorStore
            from langchain_huggingface import HuggingFaceEmbeddings

            embedding = HuggingFaceEmbeddings(
                model=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Try to create/get the collection
            vector_store = AstraDBVectorStore(
                collection_name=self.collection_name,
                api_endpoint=self.api_endpoint,
                token=self.token,
                embedding=embedding,
            )

            # Test by checking collection info
            logger.info(f"Validating connection to AstraDB collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"AstraDB connection validation failed: {e}")
            return False

    def get_status(self):
        """Return builder status."""
        return {
            "name": self.__class__.__name__,
            "started": self._started,
            "api_endpoint": self.api_endpoint[:50] + "..." if self.api_endpoint else "",
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
        }
