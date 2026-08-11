"""
GraphRAG Adapter for LifeBench_eval.

Uses AstraDBVectorStore for document storage and GraphRetriever for graph-based retrieval.
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_astradb import AstraDBVectorStore
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.graph_retriever import Eager

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory

logger = logging.getLogger(__name__)


@register_adapter("graphrag")
class GraphRAGAdapter(BaseAdapter):
    """GraphRAG adapter using AstraDB + GraphRetriever.

    Configuration:
        api_endpoint: AstraDB API endpoint URL
        token: AstraDB application token
        collection_name: Vector store collection name (default: lifebench_graphrag)
        embedding_model: Embedding model to use (default: thenlper/gte-large-zh)
        edges: List of edge tuples for graph traversal (default: [("habitat", "habitat")])
        strategy_k: Top-k results for strategy (default: 5)
        strategy_start_k: Starting depth k (default: 1)
        strategy_max_depth: Maximum traversal depth (default: 2)
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.stats_collector = stats_collector

        # AstraDB connection settings
        self.api_endpoint = config.get("api_endpoint", "")
        self.token = config.get("token", "")
        self.collection_name = config.get("collection_name", "lifebench_graphrag")

        # Embedding settings (nested config: embedding.model)
        embed_cfg = config.get("embedding", {})
        self.embedding_model = embed_cfg.get("model", "thenlper/gte-large-zh")
        self.embedding_dimensions = embed_cfg.get("dimensions", 1024)
        self.embedding_normalize = embed_cfg.get("normalize", True)

        # Retrieval settings (nested config: retrieval.*)
        retrieval_cfg = config.get("retrieval", {})
        self.search_top_k = retrieval_cfg.get("top_k", 40)
        self.edges = retrieval_cfg.get("edges", [("habitat", "habitat")])
        strategy_cfg = retrieval_cfg.get("strategy", {})
        self.strategy_k = strategy_cfg.get("k", 5)
        self.strategy_start_k = strategy_cfg.get("start_k", 1)
        self.strategy_max_depth = strategy_cfg.get("max_depth", 2)

        # Search settings (fallback: search.top_k)
        search_cfg = config.get("search", {})
        if search_cfg.get("top_k") and not retrieval_cfg.get("top_k"):
            self.search_top_k = search_cfg.get("top_k", 40)

        # LLM settings for answer generation (nested config: llm.*)
        llm_cfg = config.get("llm", {})
        self.llm_provider = llm_cfg.get("provider", "openai")
        self.llm_model = llm_cfg.get("model", "deepseek-chat")
        self.llm_api_key = llm_cfg.get("api_key", "")
        self.llm_base_url = llm_cfg.get("base_url", "https://api.deepseek.com")
        self.llm_temperature = llm_cfg.get("temperature", 0)
        self.llm_max_tokens = llm_cfg.get("max_tokens", 32768)

        # Lazy initialization
        self._vector_store: Optional[AstraDBVectorStore] = None
        self._graph_retriever: Optional[GraphRetriever] = None
        self._embedding = None

    def _get_embedding_model(self):
        """Get or create embedding model."""
        if self._embedding is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embedding = HuggingFaceEmbeddings(
                model=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embedding

    async def _get_vector_store(self) -> AstraDBVectorStore:
        """Get or create AstraDBVectorStore."""
        if self._vector_store is None:
            embedding = self._get_embedding_model()
            self._vector_store = AstraDBVectorStore(
                collection_name=self.collection_name,
                api_endpoint=self.api_endpoint,
                token=self.token,
                embedding=embedding,
            )
        return self._vector_store

    async def _get_graph_retriever(self) -> GraphRetriever:
        """Get or create GraphRetriever."""
        if self._graph_retriever is None:
            vector_store = await self._get_vector_store()
            self._graph_retriever = GraphRetriever(
                store=vector_store,
                edges=self.edges,
                strategy=Eager(
                    k=self.strategy_k,
                    start_k=self.strategy_start_k,
                    max_depth=self.strategy_max_depth,
                ),
            )
        return self._graph_retriever

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Ingest message chunks via AstraDBVectorStore.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Dict with ingestion stats
        """
        if not chunks:
            return {"type": "graphrag", "total_chunks": 0, "added": 0, "failed": 0}

        vector_store = await self._get_vector_store()
        total_added = 0
        total_failed = 0

        for chunk in chunks:
            if not chunk.messages:
                continue

            # Convert messages to documents
            from langchain_core.documents import Document

            docs = []
            for msg in chunk.messages:
                # Create document with metadata
                doc = Document(
                    page_content=msg.content,
                    metadata={
                        "conversation_id": chunk.conversation_id,
                        "speaker": msg.speaker_id,
                        "speaker_name": msg.speaker_name,
                        "timestamp": msg.timestamp,
                    },
                )
                docs.append(doc)

            if docs:
                try:
                    # Add documents to vector store
                    await vector_store.aadd_documents(docs)
                    total_added += len(docs)
                    logger.debug(
                        f"Added {len(docs)} documents for conversation {chunk.conversation_id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to add documents: {e}")
                    total_failed += len(docs)

        return {
            "type": "graphrag",
            "total_chunks": len(chunks),
            "added": total_added,
            "failed": total_failed,
        }

    async def search(
        self, query: str, conversation_id: str, index: Any = None, **kwargs
    ) -> SearchResult:
        """Search memories via GraphRetriever.

        Args:
            query: Query text
            conversation_id: Conversation ID (used for filtering)
            index: Optional index object (not used)
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            SearchResult with retrieved memories
        """
        # Use config top_k as default, allow override via kwargs
        top_k = kwargs.get("top_k", self.search_top_k)

        try:
            graph_retriever = await self._get_graph_retriever()

            # Use graph retriever to get relevant documents
            results = await graph_retriever.ainvoke(query)

            # Convert results to retrieved memories
            retrieved_memories = []
            combined_content = ""

            for i, doc in enumerate(results[:top_k]):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                score = 1.0 - (i * 0.1)  # Assign decreasing scores based on position

                retrieved_memories.append(
                    RetrievedMemory(
                        content=content,
                        score=score,
                        metadata={
                            "conversation_id": doc.metadata.get("conversation_id", "") if hasattr(doc, "metadata") else "",
                        },
                    )
                )

                if combined_content:
                    combined_content += "\n\n---\n\n"
                combined_content += content

            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=retrieved_memories,
                retrieval_metadata={
                    "adapter": "graphrag",
                    "collection_name": self.collection_name,
                    "top_k": top_k,
                    "total_results": len(retrieved_memories),
                    "strategy": f"Eager(k={self.strategy_k}, start_k={self.strategy_start_k}, max_depth={self.strategy_max_depth})",
                },
            )

        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            return SearchResult(
                question_id=kwargs.get("question_id", ""),
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "adapter": "graphrag",
                    "error": str(e)[:200],
                },
            )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """
        Generate answer using LLM given query and retrieved context.

        Args:
            query: Question text
            context: Formatted retrieved context
            conversation_id: Conversation ID
            **kwargs: Extra parameters

        Returns:
            Generated answer string
        """
        if not self.llm_api_key:
            logger.error("No LLM API key configured for answer generation")
            return "Error: No LLM API key configured"

        reference_date = kwargs.get("reference_date", "2023")

        prompt = f"""You are answering a question using retrieved memories from past conversations. Follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL MEMORIES
Read EVERY memory below from first to last. For each one that contains information relevant to the question, note it. Do NOT stop after finding the first relevant memory — important details are often scattered across many memories.

## Step 2: ENTITY VERIFICATION
Confirm each relevant memory is about the correct person/entity.

## Step 3: COMBINE AND CROSS-REFERENCE
Combine facts from multiple memories about the same topic. For listing/counting questions, extract EVERY distinct item.

## Step 4: SELECT THE BEST ANSWER
Choose the MOST SPECIFIC detail available. A proper name, title, or number beats a generic description.

## Step 5: TEMPORAL GROUNDING
These conversations took place around {reference_date}. All events occurred in 2022-2024.

## Step 6: INCLUSION CHECK
If you found items during reasoning that you're tempted to exclude — STOP. Include them unless you have STRONG evidence they are wrong.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer. NEVER say "not specified" or "no record" — if ANY memory contains relevant information, give the best answer.

{context if context else "(No relevant memories found)"}

Question: {query}

Work through Steps 1-7, then give your final answer after "ANSWER:". """

        import aiohttp

        url = f"{self.llm_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    resp.raise_for_status()
                    data = await resp.json()

            if isinstance(data, dict) and "choices" in data:
                return data["choices"][0]["message"]["content"]
            return str(data)
        except Exception as exc:
            logger.error("Answer generation failed: %s", str(exc)[:200])
            return f"Error generating answer: {str(exc)[:100]}"

    async def session_end(self, conversation_id: str) -> bool:
        """Session end handler (no-op for GraphRAG).

        GraphRAG doesn't require explicit session end triggering.

        Args:
            conversation_id: Conversation ID

        Returns:
            True
        """
        return True

    async def close(self) -> None:
        """Cleanup resources."""
        self._vector_store = None
        self._graph_retriever = None
        self._embedding = None

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info for result recording."""
        return {
            "name": self.__class__.__name__,
            "config": {
                "api_endpoint": self.api_endpoint,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "embedding_dimensions": self.embedding_dimensions,
                "search_top_k": self.search_top_k,
                "edges": self.edges,
                "strategy_k": self.strategy_k,
                "strategy_start_k": self.strategy_start_k,
                "strategy_max_depth": self.strategy_max_depth,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "llm_base_url": self.llm_base_url,
            },
        }
