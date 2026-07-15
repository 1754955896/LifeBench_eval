"""
Checkpoint management module - supports resume from interruption.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from datetime import datetime


class CheckpointManager:
    """
    Checkpoint manager with fine-grained tracking.

    - add/search: tracked by date (e.g., add_2025-01-01, search_2025-01-01)
    - answer/eval: tracked by completed QA IDs
    """

    def __init__(self, output_dir: Path, run_name: str = "default"):
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.checkpoint_file = self.output_dir / f"checkpoint_{run_name}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[Dict[str, Any]] = None

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists."""
        if self._cache is not None:
            return self._cache

        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            return self._cache
        except Exception:
            return None

    def save(self, checkpoint: Dict[str, Any]) -> None:
        """Save checkpoint."""
        checkpoint["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            self._cache = checkpoint
        except Exception:
            pass

    def load_or_create(self) -> Dict[str, Any]:
        """Load existing checkpoint or create new one."""
        cp = self.load() or {
            "run_name": self.run_name,
            "completed_stages": [],
            "answered_qa_ids": [],
        }
        if "sample_add_completed" not in cp:
            cp["sample_add_completed"] = []
        if "sample_search_completed" not in cp:
            cp["sample_search_completed"] = []
        if "date_add_completed" not in cp:
            cp["date_add_completed"] = {}
        if "date_search_completed" not in cp:
            cp["date_search_completed"] = {}
        return cp

    def mark_add_complete(self, date_str: str) -> None:
        """Mark add for a specific date as complete."""
        checkpoint = self.load_or_create()
        stage = f"add_{date_str}"
        if stage not in checkpoint["completed_stages"]:
            checkpoint["completed_stages"].append(stage)
        self.save(checkpoint)

    def mark_search_complete(self, date_str: str) -> None:
        """Mark search for a specific date as complete."""
        checkpoint = self.load_or_create()
        stage = f"search_{date_str}"
        if stage not in checkpoint["completed_stages"]:
            checkpoint["completed_stages"].append(stage)
        self.save(checkpoint)

    def mark_answer_complete(self, qa_ids: List[str]) -> None:
        """Mark QAs as answered."""
        checkpoint = self.load_or_create()
        answered = set(checkpoint.get("answered_qa_ids", []))
        answered.update(qa_ids)
        checkpoint["answered_qa_ids"] = list(answered)
        self.save(checkpoint)

    def mark_evaluate_complete(self) -> None:
        """Mark evaluate as complete."""
        checkpoint = self.load_or_create()
        if "evaluate" not in checkpoint["completed_stages"]:
            checkpoint["completed_stages"].append("evaluate")
        self.save(checkpoint)

    def mark_sample_add_complete(self, sample_id: str) -> None:
        """Mark add for a sample as complete."""
        checkpoint = self.load_or_create()
        if sample_id not in checkpoint["sample_add_completed"]:
            checkpoint["sample_add_completed"].append(sample_id)
        self.save(checkpoint)

    def is_sample_add_complete(self, sample_id: str) -> bool:
        """Check if add for sample is complete."""
        return sample_id in self.load_or_create().get("sample_add_completed", [])

    def mark_sample_search_complete(self, sample_id: str) -> None:
        """Mark search for a sample as complete."""
        checkpoint = self.load_or_create()
        if sample_id not in checkpoint["sample_search_completed"]:
            checkpoint["sample_search_completed"].append(sample_id)
        self.save(checkpoint)

    def is_sample_search_complete(self, sample_id: str) -> bool:
        """Check if search for sample is complete."""
        return sample_id in self.load_or_create().get("sample_search_completed", [])

    def mark_date_add_complete(self, sample_id: str, date_str: str) -> None:
        """Mark add for a specific date within a sample as complete."""
        checkpoint = self.load_or_create()
        if sample_id not in checkpoint["date_add_completed"]:
            checkpoint["date_add_completed"][sample_id] = []
        if date_str not in checkpoint["date_add_completed"][sample_id]:
            checkpoint["date_add_completed"][sample_id].append(date_str)
        self.save(checkpoint)

    def is_date_add_complete(self, sample_id: str, date_str: str) -> bool:
        """Check if add for a specific date within a sample is complete."""
        date_completed = self.load_or_create().get("date_add_completed", {})
        return date_str in date_completed.get(sample_id, [])

    def mark_date_search_complete(self, sample_id: str, date_str: str) -> None:
        """Mark search for a specific date within a sample as complete."""
        checkpoint = self.load_or_create()
        if sample_id not in checkpoint["date_search_completed"]:
            checkpoint["date_search_completed"][sample_id] = []
        if date_str not in checkpoint["date_search_completed"][sample_id]:
            checkpoint["date_search_completed"][sample_id].append(date_str)
        self.save(checkpoint)

    def is_date_search_complete(self, sample_id: str, date_str: str) -> bool:
        """Check if search for a specific date within a sample is complete."""
        date_completed = self.load_or_create().get("date_search_completed", {})
        return date_str in date_completed.get(sample_id, [])

    def is_add_complete(self, date_str: str) -> bool:
        """Check if add for date is complete."""
        return f"add_{date_str}" in self.load_or_create().get("completed_stages", [])

    def is_search_complete(self, date_str: str) -> bool:
        """Check if search for date is complete."""
        return f"search_{date_str}" in self.load_or_create().get("completed_stages", [])

    def is_evaluate_complete(self) -> bool:
        """Check if evaluate is complete."""
        return "evaluate" in self.load_or_create().get("completed_stages", [])

    def get_answered_qa_ids(self) -> Set[str]:
        """Get set of answered QA IDs."""
        return set(self.load_or_create().get("answered_qa_ids", []))

    def get_completed_stages(self) -> List[str]:
        """Get list of completed stages."""
        return self.load_or_create().get("completed_stages", [])

    def has_any_progress(self) -> bool:
        """Check if checkpoint has any progress recorded."""
        cp = self.load()
        if not cp:
            return False
        return bool(
            cp.get("completed_stages") or
            cp.get("sample_add_completed") or
            cp.get("sample_search_completed") or
            cp.get("answered_qa_ids") or
            cp.get("date_add_completed") or
            cp.get("date_search_completed")
        )

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of checkpoint progress for logging."""
        cp = self.load()
        if not cp:
            return {"has_checkpoint": False}

        return {
            "has_checkpoint": True,
            "last_updated": cp.get("last_updated", "Unknown"),
            "sample_add_completed": cp.get("sample_add_completed", []),
            "sample_search_completed": cp.get("sample_search_completed", []),
            "date_add_completed": cp.get("date_add_completed", {}),
            "date_search_completed": cp.get("date_search_completed", {}),
            "answered_qa_ids": cp.get("answered_qa_ids", []),
            "evaluate_complete": "evaluate" in cp.get("completed_stages", []),
        }

    def delete(self) -> None:
        """Delete checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                self._cache = None
            except Exception:
                pass