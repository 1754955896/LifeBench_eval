"""
Search result models.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RetrievedMemory:
    """A single retrieved memory with score."""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Standard search result format."""
    question_id: str
    query: str
    conversation_id: str
    results: List[RetrievedMemory] = field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
