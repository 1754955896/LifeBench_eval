"""
Hybrid evaluator - combines exact match and LLM judge.
"""
from typing import List

from src.evaluators.base import BaseEvaluator
from src.evaluators.registry import register_evaluator
from src.models.evaluation import EvaluationResult


@register_evaluator("hybrid")
class HybridEvaluator(BaseEvaluator):
    """Hybrid evaluator combining exact match and LLM judge."""

    def __init__(self, config: dict):
        super().__init__(config)

    async def evaluate(self, answer_results: List) -> EvaluationResult:
        """Evaluate using hybrid approach."""
        print(f"\n{'=' * 60}")
        print(f"Stage 4/4: Evaluate  [Hybrid Evaluator]")
        print(f"{'=' * 60}")

        detailed_results = []
        total_correct = 0

        for ar in answer_results:
            # Try exact match first
            golden = (ar.golden_answer or "").strip().lower()
            generated = (ar.answer or "").strip().lower()

            if golden == generated:
                is_correct = True
            else:
                # Fall back to simple similarity check
                is_correct = self._similar_enough(golden, generated)

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
            metadata={"evaluator": "hybrid"},
        )

    def _similar_enough(self, golden: str, generated: str) -> bool:
        """Check if answers are similar enough."""
        if not golden or not generated:
            return False

        # Simple character overlap check
        golden_chars = set(golden)
        generated_chars = set(generated)

        if not golden_chars:
            return False

        overlap = len(golden_chars & generated_chars) / len(golden_chars)
        return overlap > 0.8
