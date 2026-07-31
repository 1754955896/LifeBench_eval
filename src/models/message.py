"""
Message and Conversation models.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class Message:
    """Standard message format."""
    speaker_name: str
    content: str
    timestamp: Optional[datetime] = None
    dia_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Standard conversation format."""
    conversation_id: str
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
