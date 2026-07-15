"""
Adapter base class - define unified memory system adapter interface.
"""
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.message import Conversation, Message
    from src.models.search import SearchResult


@dataclass
class ChunkedMessage:
    """A message chunk for batch ingestion.

    Attributes:
        messages: List of messages in this chunk
        conversation_id: ID of the conversation this chunk belongs to
        session_id: Optional session identifier for timestamp association
        timestamp: Optional timestamp for this chunk (unix epoch)
    """
    messages: List["Message"]
    conversation_id: str
    session_id: Optional[str] = None
    timestamp: Optional[int] = None


class BaseAdapter(ABC):
    """Memory system adapter base class."""

    def __init__(self, config: dict):
        """
        Initialize adapter.

        Args:
            config: System config dict
        """
        self.config = config

    async def add(self, conversations: List["Conversation"], **kwargs) -> Any:
        """
        Ingest conversation data and build index (Add stage).

        Creates one ChunkedMessage per conversation (all messages in a single chunk),
        then delegates to add_chunks().

        Args:
            conversations: Standard format conversation list
            **kwargs: Extra parameters

        Returns:
            Index object (system-specific)
        """
        chunks = [
            ChunkedMessage(
                messages=conv.messages,
                conversation_id=conv.conversation_id,
                session_id=conv.metadata.get("session_id", "default"),
                timestamp=self._conv_timestamp(conv),
            )
            for conv in conversations
        ]
        return await self.add_chunks(chunks, **kwargs)

    def _conv_timestamp(self, conv: "Conversation") -> Optional[int]:
        """Get unix timestamp from conversation metadata."""
        ts = conv.metadata.get("timestamp")
        if ts is None:
            return None
        if isinstance(ts, datetime.datetime):
            return int(ts.timestamp())
        return ts

    async def add_chunks(self, chunks: List[ChunkedMessage], **kwargs) -> Any:
        """
        Ingest message chunks (alternative to add() for chunk-based systems).

        Override this method for systems that require per-chunk ingestion.

        Args:
            chunks: List of ChunkedMessage objects
            **kwargs: Extra parameters

        Returns:
            Index object (system-specific)
        """
        raise NotImplementedError(
            "add_chunks() must be implemented for chunk-based ingestion"
        )

    @abstractmethod
    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> "SearchResult":
        """
        Retrieve relevant memories (Search stage).

        Args:
            query: Query text
            conversation_id: Conversation ID
            index: Index object (returned by add())
            **kwargs: Extra parameters (e.g., top_k)

        Returns:
            Standard format search result
        """

    async def answer(
        self, query: str, context: str, conversation_id: str, **kwargs
    ) -> str:
        """
        Generate answer given retrieved context (Answer stage).

        Args:
            query: Question text
            context: Formatted retrieved context
            conversation_id: Conversation ID
            **kwargs: Extra parameters

        Returns:
            Generated answer string
        """
        raise NotImplementedError("Answer method not implemented for this adapter")

    async def prepare(self, conversations: List["Conversation"], **kwargs) -> None:
        """
        Preparation stage: operations executed before add.

        Args:
            conversations: Standard format conversation list
            **kwargs: Extra parameters
        """
        pass

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info for result recording."""
        return {"name": self.__class__.__name__, "config": self.config}

    def build_lazy_index(
        self, conversations: List["Conversation"], output_dir: Any
    ) -> Any:
        """
        Build lazy-loaded index metadata.

        Args:
            conversations: Conversation list
            output_dir: Output directory

        Returns:
            Index object or metadata
        """
        return None

    async def close(self) -> None:
        """Cleanup resources."""
        pass
