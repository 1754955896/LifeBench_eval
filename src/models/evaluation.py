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
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    question_type_stats: Dict[str, QuestionTypeStats] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> Optional[float]:
        """Average weighted score across all question types."""
        scored = [
            s.weighted_score
            for s in self.question_type_stats.values()
            if s.weighted_score is not None
        ]
        return sum(scored) / len(scored) if scored else None
