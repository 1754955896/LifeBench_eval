"""
Example adapter - simple in-memory implementation for demonstration.
"""
from typing import Any, Dict, List

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.message import Conversation
from src.models.search import SearchResult, RetrievedMemory


@register_adapter("example")
class ExampleAdapter(BaseAdapter):
    """
    Example adapter - simple in-memory search for testing.

    This adapter demonstrates the adapter interface and provides
    a simple baseline implementation.
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.memories: Dict[str, List[str]] = {}
        self._chunks_processed = 0

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Store all chunk messages as simple string memories.

        This demonstrates chunk-based ingestion - each chunk is processed
        individually and added to the conversation's memory list.
        """
        for chunk in chunks:
            for msg in chunk.messages:
                content = f"{msg.speaker_name}: {msg.content}"
                if chunk.conversation_id not in self.memories:
                    self.memories[chunk.conversation_id] = []
                self.memories[chunk.conversation_id].append(content)
            self._chunks_processed += 1

        return {
            "type": "chunked",
            "total_convs": len(set(c.conversation_id for c in chunks)),
            "total_chunks": len(chunks),
            "chunks_processed": self._chunks_processed,
        }

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """Simple keyword-based search."""
        memories = self.memories.get(conversation_id, [])

        results = []
        query_lower = query.lower()

        for i, mem in enumerate(memories):
            score = 0.0
            mem_lower = mem.lower()

            if query_lower in mem_lower:
                score = 1.0
            else:
                query_words = query_lower.split()
                for word in query_words:
                    if word in mem_lower:
                        score += 0.3

            if score > 0:
                results.append(RetrievedMemory(
                    content=mem,
                    score=min(score, 1.0),
                    metadata={"index": i}
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        top_k = kwargs.get("top_k", 10)
        results = results[:top_k]

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={"adapter": "example"}
        )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """Simple answer generation - just use the query and context."""
        if context:
            return f"Based on my memories: {context[:100]}..."
        return "I don't have relevant memories to answer this question."
