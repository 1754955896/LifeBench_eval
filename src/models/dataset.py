"""
Dataset and QA pair models.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class QAPair:
    """Standard QA pair format."""
    question_id: str
    question: str
    answer: str
    category: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """Standard dataset format."""
    dataset_name: str
    conversations: List[Any] = field(default_factory=list)  # Legacy field
    qa_pairs: List[QAPair] = field(default_factory=list)    # Legacy field
    samples: List[Any] = field(default_factory=list)        # New structured samples
    metadata: Dict[str, Any] = field(default_factory=dict)
