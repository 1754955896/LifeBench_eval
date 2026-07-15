"""
LoCoMo dataset loader.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.loaders.base import BaseLoader
from src.loaders.registry import register_loader
from src.models.dataset import Dataset, QAPair
from src.models.message import Conversation, Message


@register_loader("locomo")
class LoCoMoLoader(BaseLoader):
    """LoCoMo conversation format loader."""

    def load(self, data_path: str, **kwargs) -> Dataset:
        """Load LoCoMo dataset from JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversations = []
        qa_pairs = []

        for person_data in data:
            sample_id = person_data.get("sample_id", "")
            conversation_data = person_data.get("conversation", {})

            # Extract speaker names from conversation
            speaker_a = conversation_data.get("speaker_a", "")
            speaker_b = conversation_data.get("speaker_b", "")

            # Build raw conversation dict with all sessions
            raw_conv = {}
            for key, value in conversation_data.items():
                if key.startswith("session_") and "_date_time" not in key:
                    if isinstance(value, list):
                        raw_conv[key] = value

            # Extract date-ordered sessions
            session_keys = sorted(
                [k for k in raw_conv.keys() if k.startswith("session_")],
                key=lambda x: int(re.search(r"session_(\d+)", x).group(1))
            )

            # Build session_id -> messages mapping for ordering
            sessions_by_conversation = {}
            for session_key in session_keys:
                if session_key in raw_conv:
                    sessions_by_conversation[session_key] = raw_conv[session_key]

            # Also pass date_time info
            for key, value in conversation_data.items():
                if key.startswith("session_") and "_date_time" in key:
                    sessions_by_conversation[key] = value

            # Create conversation
            conv = Conversation(
                conversation_id=sample_id,
                messages=[],  # Will be reordered by pipeline if needed
                metadata={
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    "timestamp": None,
                    "_raw_conversation": sessions_by_conversation,
                }
            )
            conversations.append(conv)

            # Parse QA pairs
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

        return Dataset(
            dataset_name="locomo",
            conversations=conversations,
            qa_pairs=qa_pairs,
        )