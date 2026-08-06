"""
LoCoMo dataset loader.
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.loaders.base import BaseLoader
from src.loaders.registry import register_loader
from src.models.dataset import Dataset, QAPair


@dataclass
class Message:
    """A single message in a session."""
    speaker_name: str       # e.g., "Caroline"
    content: str            # e.g., "Hey Mel! Good to see you!"
    dia_id: str             # e.g., "D1:1" (dialogue ID)
    blip_caption: str = ""  # Image caption from BLIP model
    query: str = ""         # Image query/context


@dataclass
class Session:
    """A session within a conversation."""
    session_time: str            # Format: YYYY-MM-DD
    session_id: str            # e.g., "session_1"
    messages: List[Message]    # List of messages in this session
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_time_original: str = ""  # Original full time string, e.g., "1:56 pm on 8 May, 2023"


@dataclass
class Sample:
    """A sample containing conversation sessions and QA pairs."""
    conversation_id: str
    sessions: List[Session] = field(default_factory=list)
    qa_pairs: List[QAPair] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@register_loader("locomo")
class LoCoMoLoader(BaseLoader):
    """LoCoMo conversation format loader."""

    # Date format patterns for parsing locomo data
    DATE_PATTERNS = [
        "%Y-%m-%d",                     # "2025-01-01"
        "%I:%M %p on %d %B, %Y",        # "1:56 pm on 8 May, 2023"
        "%I:%M %p on %d %B %Y",         # "1:56 pm on 8 May 2023"
    ]

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not isinstance(date_str, str) or not date_str.strip():
            return None
        for fmt in self.DATE_PATTERNS:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    def _format_date(self, dt: datetime) -> str:
        """Format datetime to YYYY-MM-DD string."""
        return dt.strftime("%Y-%m-%d")

    def load(self, data_path: str, **kwargs) -> Dataset:
        """Load LoCoMo dataset from JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []

        for person_data in data:
            sample_id = person_data.get("sample_id", "")
            conversation_data = person_data.get("conversation", {})

            # Extract speaker names
            speaker_a = conversation_data.get("speaker_a", "")
            speaker_b = conversation_data.get("speaker_b", "")

            # Collect all session keys and their dates
            sessions_data: Dict[str, Dict[str, Any]] = {}

            # First pass: collect session messages
            for key, value in conversation_data.items():
                if key.startswith("session_") and "_date_time" not in key:
                    if isinstance(value, list):
                        sessions_data[key] = {"messages": value, "session_time": ""}

            # Second pass: assign dates to sessions
            for key, value in conversation_data.items():
                if key.startswith("session_") and "_date_time" in key:
                    session_num_match = re.search(r"session_(\d+)_date_time", key)
                    if session_num_match:
                        session_key = f"session_{session_num_match.group(1)}"
                        parsed_dt = self._parse_date(value)
                        if parsed_dt and session_key in sessions_data:
                            sessions_data[session_key]["session_time"] = self._format_date(parsed_dt)
                            # Also preserve the original full time string for accurate timestamps
                            sessions_data[session_key]["session_time_original"] = value  # e.g., "1:56 pm on 8 May, 2023"

            # Sort sessions by session number
            session_keys = sorted(
                [k for k in sessions_data.keys() if k.startswith("session_")],
                key=lambda x: int(re.search(r"session_(\d+)", x).group(1))
            )

            # Build sessions list
            sessions = []
            for session_key in session_keys:
                session_info = sessions_data.get(session_key, {})
                # Convert raw message dicts to Message objects
                raw_messages = session_info.get("messages", [])
                messages = [
                    Message(
                        speaker_name=msg.get("speaker", ""),
                        content=msg.get("text", ""),
                        dia_id=msg.get("dia_id", ""),
                        blip_caption=msg.get("blip_caption", ""),
                        query=msg.get("query", ""),
                    )
                    for msg in raw_messages
                ]
                sessions.append(Session(
                    session_time=session_info.get("session_time", ""),
                    session_id=session_key,
                    messages=messages,
                    metadata={},
                    session_time_original=session_info.get("session_time_original", ""),
                ))

            # Parse QA pairs
            qa_pairs = []
            for qa in person_data.get("qa", []):
                qa_pairs.append(QAPair(
                    question_id=qa.get("question_id", ""),
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    category=str(qa.get("category", "")),
                    evidence=qa.get("evidence", []),
                    metadata={
                        "ask_time": qa.get("ask_time", ""),
                        "question_type": qa.get("question_type", []),
                        "score_points": qa.get("score_points", []),
                        "person_id": sample_id,
                        "conversation_id": sample_id,
                    }
                ))

            # Create sample
            samples.append(Sample(
                conversation_id=sample_id,
                sessions=sessions,
                qa_pairs=qa_pairs,
                metadata={
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                }
            ))

        return Dataset(
            dataset_name=kwargs.get("name", "locomo"),
            samples=samples,
            metadata={"format": kwargs.get("dataset_format", "locomo")},
        )
