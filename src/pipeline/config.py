"""
Pipeline configuration.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    adapter_name: str = ""
    evaluator_name: str = "llm_judge"
    stages: List[str] = field(
        default_factory=lambda: ["add", "search", "answer", "evaluate"]
    )
    use_checkpoint: bool = True
    run_name: Optional[str] = None
    filter_categories: List[str] = field(default_factory=list)
    from_conversation: int = 0
    to_conversation: Optional[int] = None
    smoke_test: bool = False
    smoke_messages: int = 10
    smoke_questions: int = 3
    system_overrides: Dict[str, Any] = field(default_factory=dict)
