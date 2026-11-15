"""
Base builder class for memory system initialization.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseBuilder(ABC):
    """Base class for system builders."""

    def __init__(self, config: dict, project_root: Optional[str] = None):
        """
        Initialize builder.

        Args:
            config: System configuration dict
            project_root: Project root path
        """
        self.config = config
        self.project_root = project_root

    @abstractmethod
    async def build(self) -> bool:
        """
        Execute builder to start/initialize the memory system.

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Cleanup resources after evaluation.

        Returns:
            True if successful, False otherwise
        """

    def get_status(self) -> Dict[str, Any]:
        """Return builder status information."""
        return {"name": self.__class__.__name__, "config": self.config}