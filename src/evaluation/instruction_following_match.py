"""Instruction-following görevleri için deterministik evaluator sağlar.

Amaç yalnızca reference cevabın aynısını aramak değil,
istenen format ve kısıtların takip edilip edilmediğini ölçmektir.

İlk sürüm özellikle mevcut benchmark'taki structured instruction
örneklerini destekler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.normalize_answer import normalize_answer
from src.evaluation.run_inference import PredictionRecord


@dataclass(frozen=True)
class InstructionFollowingResult:
    """Tek bir instruction-following prediction sonucunu temsil eder."""

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    reference_answer: str
    instruction_match: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Result nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


def instruction_following_score(
    prediction: str, 
    reference_answer: str,
    category: str | None = None
) -> int:
    """Prediction verilen output formatını tam olarak takip ediyor mu kontrol eder.

    Bu ilk sürüm strict normalized equality kullanır.

    Örnek:

        reference = "RESULT: red"
        prediction = "RESULT: red"
        -> 1

        prediction = "The red car is faster. RESULT: red"
        -> 0

    Bu davranış instruction-following için bilinçli olarak strict tutulur.
    """

    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference_answer)

    if category == 'extraction':
        return int(
            normalized_reference
            in normalized_prediction
        )

    if category == 'transformation':
        return int(
            prediction.strip()
            == reference_answer.strip()
        )

    if category == 'format_following':
        return int(
            prediction.strip()
            == reference_answer.strip()
        )

    if category == 'constraint_following':
        return int(
            prediction.strip()
            == reference_answer.strip()
        )

    if category == "multi_step_instruction":
        return int(
            prediction.strip()
            == reference_answer.strip()
        )
    

    return int(
        normalized_prediction
        == normalized_reference
    )



def evaluate_instruction_following_prediction(
    record: PredictionRecord
) -> InstructionFollowingResult:
    """Tek bir instruction-following prediction değerlendirir."""

    score = instruction_following_score(
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        category=record.metadata.get("category")
    )


    return InstructionFollowingResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        instruction_match=score,
        metadata=dict(record.metadata)
    )


def evaluate_instruction_following(
    records: list[PredictionRecord],
) -> list[InstructionFollowingResult]:
    """Instruction-following kayıtlarını değerlendirir."""

    return [
        evaluate_instruction_following_prediction(
            record
        )
        for record in records
    ]



def calculate_instruction_following_accuracy(
    results: list[InstructionFollowingResult]
) -> float:
    """Instruction-following sonuçlarından accuracy hesaplar."""

    if not results:
        raise ValueError(
            "Cannot calculate accuracy from empty instruction-following results."
        )

    correct = sum(
        result.instruction_match
        for result in results
    )

    accuracy = correct / len(results)

    return accuracy


def summarize_instruction_following(
    results: list[InstructionFollowingResult]
) -> dict[str, Any]:
    """Instruction-following sonuçlarının temel özetini oluşturur."""

    if not results:
        raise ValueError(
            "Cannot summarize empty instruction-following results."
        )

    correct = sum(
        result.instruction_match
        for result in results
    )

    total = len(results)

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total
    }