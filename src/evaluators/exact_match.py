"""
Exact Match evaluator - direct answer comparison.
"""
import re
from typing import List

from src.evaluators.base import BaseEvaluator
from src.evaluators.registry import register_evaluator
from src.models.evaluation import EvaluationResult, QuestionTypeStats


@register_evaluator("exact_match")
class ExactMatch(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.case_sensitive = config.get("case_sensitive", False)
        self.normalize_whitespace = config.get("normalize_whitespace", True)
        self.extract_choice = config.get("extract_choice", True)

    async def evaluate(self, answer_results: List) -> EvaluationResult:
        """Evaluate answers using exact match."""
        print(f"\n{'=' * 60}")
        print(f"Stage 4/4: Evaluate  [Exact Match]")
        print(f"{'=' * 60}")

        detailed_results = []
        total_correct = 0

        for ar in answer_results:
            is_correct = self._check_match(ar.golden_answer, ar.answer)
            if is_correct:
                total_correct += 1

            detailed_results.append({
                "question_id": ar.question_id,
                "question": ar.question,
                "golden_answer": ar.golden_answer,
                "generated_answer": ar.answer,
                "is_correct": is_correct,
                "category": ar.category,
            })

        total = len(answer_results)
        accuracy = total_correct / total if total else 0.0

        print(f"\n✅ Evaluation complete:")
        print(f"   - Total questions: {total}")
        print(f"   - Correct: {total_correct}")
        print(f"   - Accuracy: {accuracy:.2%}")

        return EvaluationResult(
            total_questions=total,
            correct=total_correct,
            accuracy=accuracy,
            detailed_results=detailed_results,
            metadata={
                "evaluator": "exact_match",
                "case_sensitive": self.case_sensitive,
            },
        )

    def _check_match(self, golden: str, generated: str) -> bool:
        """Check if two answers match."""
        golden_processed = self._preprocess(golden)
        generated_processed = self._preprocess(generated)

        if self.extract_choice:
            extracted = self._extract_choice(generated_processed)
            if extracted:
                generated_processed = extracted

        if self.case_sensitive:
            return golden_processed == generated_processed
        return golden_processed.lower() == generated_processed.lower()

    def _preprocess(self, text: str) -> str:
        """Preprocess text."""
        if not text:
            return ""
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_choice(self, text: str) -> str:
        """Extract choice from text."""
        match = re.search(r'\(([a-zA-Z])\)', text)
        if match:
            return f"({match.group(1).lower()})"
        return text
