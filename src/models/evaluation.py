"""
Evaluation result models.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class QuestionTypeStats:
    """Statistics for a single question type."""
    name: str
    count: int
    correct: int
    accuracy: float
    weighted_score: Optional[float] = None


@dataclass
class EvaluationResult:
    """Standard evaluation result format."""
    total_questions: int
    correct: int
    accuracy: float
    weighted_score: Optional[float] = None
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    question_type_stats: Dict[str, QuestionTypeStats] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
