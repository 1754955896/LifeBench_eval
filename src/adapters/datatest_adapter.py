"""
DataTest adapter - saves all inputs for testing and returns placeholder results.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from src.adapters.base import BaseAdapter, ChunkedMessage
from src.adapters.registry import register_adapter
from src.models.search import SearchResult, RetrievedMemory


def _get_results_dir() -> str:
    """Get or create the results/data_test directory under LifeBench_eval."""
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "results",
        "data_test"
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _save_input(operation: str, data: Any, extra_info: str = "") -> str:
    """Save input data to results folder and return the file path."""
    results_dir = _get_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{operation}_{extra_info}_{timestamp}.json" if extra_info else f"{operation}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    # Convert data to serializable format
    serializable_data = _make_serializable(data)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    return filepath


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return _make_serializable(obj.__dict__)
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


@register_adapter("datatest")
class DataTestAdapter(BaseAdapter):
    """
    DataTest adapter for testing data flow.

    Saves all inputs to LifeBench_eval/results folder and returns placeholder results.
    All add_chunks calls are accumulated into add_chunks.json.
    All search calls are accumulated into search.json.
    """

    def __init__(self, config: dict, output_dir=None, stats_collector=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.add_chunks_data: List[Dict[str, Any]] = []
        self.search_data: List[Dict[str, Any]] = []

    def _save_file(self, filename: str, data: Any) -> str:
        """Save data to a JSON file in results directory."""
        results_dir = _get_results_dir()
        filepath = os.path.join(results_dir, filename)
        serializable_data = _make_serializable(data)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        return filepath

    async def add_chunks(
        self, chunks: List[ChunkedMessage], **kwargs
    ) -> Dict[str, Any]:
        """Save chunk inputs to add_chunks.json."""
        self.add_chunks_data.append({
            "chunks": chunks,
            "kwargs": kwargs,
        })
        self._save_file("add_chunks.json", self.add_chunks_data)
        return {
            "type": "datatest",
            "chunks_processed": len(chunks),
        }

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """Save search inputs to search.json."""
        self.search_data.append({
            "query": query,
            "conversation_id": conversation_id,
            "index": index,
            "kwargs": kwargs,
        })
        self._save_file("search.json", self.search_data)
        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=[
                RetrievedMemory(
                    content="[PLACEHOLDER] This is a placeholder result from DataTestAdapter",
                    score=0.95,
                    metadata={"source": "datatest_adapter"}
                )
            ],
            retrieval_metadata={"adapter": "datatest"}
        )

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """Return placeholder answer."""
        return "[PLACEHOLDER] This is a placeholder answer from DataTestAdapter"
