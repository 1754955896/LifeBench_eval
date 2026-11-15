"""
Pipeline stages enumeration.
"""
from enum import Enum, auto
from typing import Optional


class Stage(Enum):
    """Pipeline stages in execution order."""

    ADD = auto()
    SEARCH = auto()
    ANSWER = auto()
    EVALUATE = auto()

    @classmethod
    def from_string(cls, name: str) -> "Stage":
        """Parse stage from string."""
        return cls[name.upper()]

    @property
    def previous(self) -> Optional["Stage"]:
        """Get the previous stage."""
        stages = list(cls)
        idx = stages.index(self)
        return stages[idx - 1] if idx > 0 else None

    @property
    def next(self) -> Optional["Stage"]:
        """Get the next stage."""
        stages = list(cls)
        idx = stages.index(self)
        return stages[idx + 1] if idx < len(stages) - 1 else None
