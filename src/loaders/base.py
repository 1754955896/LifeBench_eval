"""
Base loader abstract class.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.dataset import Dataset


class BaseLoader(ABC):
    """Base dataset loader."""

    @abstractmethod
    def load(self, data_path: str, **kwargs) -> "Dataset":
        """
        Load dataset from file.

        Args:
            data_path: Path to data file
            **kwargs: Additional arguments

        Returns:
            Dataset object
        """
