"""Semantik olarak eşdeğer kısa cevapları değerlendirir.

Bu evaluator özellikle literal exact-match veya basit token containment
yaklaşımının yeterli olmadığı durumlar için kullanılır.

Örnek:

    reference = "Came to or achieved"
    prediction = "Reached means to come to a conclusion."

Bu tür cevaplar aynı temel anlamı ifade edebilir ancak exact-match ile
yakalanmayabilir.

İlk sürüm yalnızca açıkça tanımlanmış semantic alias kümelerini kullanır.
"""


from __future__ import annotations

import re

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.normalize_answer import normalize_answer
from src.evaluation.run_inference import PredictionRecord


@dataclass(frozen=True)
class SemanticAnswerMatchResult:
    """Tek bir semantic-answer evaluation sonucunu temsil eder."""

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    reference_answer: str
    semantic_match: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Result nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


SEMANTIC_ALIASES = {
    "came to or achieved": {
        "came to",
        "achieved",
        "come to",
        "reached",
        "came to a conclusion",
        "come to a conclusion"
    },
    "razılıq əldə etdilər": {
        "razılığa gəldilər",
        "razılıq əldə etdilər",
        "razılığa nail oldular"
    }
}


def contains_semantic_contradiction(
    prediction: str
) -> bool:
    """Prediction içinde açık semantic contradiction sinyali var mı kontrol eder."""

    normalized_prediction = normalize_answer(
        prediction
    )

    contradiction_patterns = (
        r"\bbecame angry\b",
        r"\bmeans angry\b",
        r"\banger\b"
    )

    return any(
        re.search(
            pattern,
            normalized_prediction
        )
        for pattern in contradiction_patterns
    )


def semantic_answer_match_score(
    prediction: str,
    reference_answer: str
) -> int:
    """Prediction reference cevabın semantik alias'larından birini içeriyor mu kontrol eder."""

    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference_answer)

    if normalized_prediction == normalized_reference:
        return 1

    if contains_semantic_contradiction(
        prediction
    ):
        return 0

    aliases = SEMANTIC_ALIASES.get(
        normalized_reference,
        set()
    )

    for alias in aliases:
        normalized_alias = normalize_answer(alias)

        if normalized_alias in normalized_prediction:
            return 1

    return 0


def evaluate_semantic_answer_prediction(
    record: PredictionRecord
) -> SemanticAnswerMatchResult:
    """Tek bir prediction için semantic-answer match üretir."""

    score = semantic_answer_match_score(
        record.prediction,
        record.reference_answer
    )

    return SemanticAnswerMatchResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        semantic_match=score,
        metadata=dict(record.metadata)
    )


SEMANTIC_ANSWER_TASKS = {
    "linguistic_understanding"
}



def is_semantic_answer_task(
    task: str
) -> bool:
    """Task'in semantic-answer evaluator kullanıp kullanmayacağını belirler."""

    return task in SEMANTIC_ANSWER_TASKS


def filter_semantic_answer_records(
    records: list[PredictionRecord]
) -> list[PredictionRecord]:
    """Yalnızca semantic-answer task kayıtlarını döndürür."""

    return [
        record
        for record in records
        if is_semantic_answer_task(record.task)
    ]

def evaluate_semantic_answer_matches(
    records: list[PredictionRecord]
) -> list[SemanticAnswerMatchResult]:
    """Prediction listesini semantic-answer evaluator ile değerlendirir."""

    return [
        evaluate_semantic_answer_prediction(record)
        for record in records
    ]


def calculate_semantic_answer_accuracy(
    results: list[SemanticAnswerMatchResult]
) -> float:
    """Semantic-answer sonuçlarından accuracy hesaplar."""

    if not results:
        raise ValueError(
            "Cannot calculate accuracy from empty semantic-answer results."
        )

    correct = sum(
        result.semantic_match
        for result in results
    )

    accuracy = correct / len(results)

    return accuracy


def summarize_semantic_answer_matches(
    results: list[SemanticAnswerMatchResult]
) -> dict[str, Any]:
    """Semantic-answer evaluation sonuçlarının temel özetini oluşturur."""

    if not results:
        raise ValueError(
            "Cannot summarize empty semantic-answer results."
        )

    correct = sum(
        result.semantic_match
        for result in results
    )

    total = len(results)

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total
    }