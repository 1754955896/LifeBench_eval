"""
Answer result models.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class AnswerResult:
    """Standard answer result format."""
    question_id: str
    question: str
    answer: str
    golden_answer: str
    category: Optional[str] = None
    conversation_id: str = ""
    formatted_context: str = ""
    search_results: List[Any] = field(default_factory=list)  # List[SearchResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
