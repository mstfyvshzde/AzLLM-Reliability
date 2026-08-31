"""Kısa ve deterministik cevaplar için answer-aware capability evaluation yapar.

Bu modül strict exact-match metriğini değiştirmez.

Amaç, aşağıdaki gibi semantik olarak doğru kısa cevapları yakalamaktır:

    reference = "Nigar"
    prediction = "Nigar is the shortest."

Exact match bu cevabı yanlış sayabilir.
Short-answer evaluation ise cevabın ana answer değerini doğru verip vermediğini
kontrol eder.

Bu evaluator yalnızca short_answer türündeki görevler için tasarlanmıştır.
Instruction-following ve unanswerable görevleri ayrı evaluator'lar tarafından
değerlendirilmelidir.
"""

from __future__ import annotations

import re


from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.normalize_answer import normalize_answer
from src.evaluation.run_inference import PredictionRecord


@dataclass(frozen=True)
class ShortAnswerMatchResult:
    """Tek bir short-answer prediction değerlendirme sonucunu temsil eder."""

    item_id: str
    pair_id: str
    language: str
    task: str
    prediction: str
    reference_answer: str
    normalized_prediction: str
    normalized_reference: str
    short_answer_match: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Result nesnesini serializable sözlüğe dönüştürür."""

        return asdict(self)


def contains_contradictory_answer(
    prediction: str,
    reference_answer: str
) -> bool:
    """Prediction reference cevabı söylerken açıkça başka bir cevabı savunuyor mu kontrol eder."""

    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference_answer)

    if not normalized_reference:
        return False

    patterns = (
        rf"\b{re.escape(normalized_reference)}\b.*\bbut\b",
        rf"\b{re.escape(normalized_reference)}\b.*\bhowever\b",
        rf"\b{re.escape(normalized_reference)}\b.*\bactually\b",
    )

    return any(
        re.search(pattern, normalized_prediction)
        for pattern in patterns
    )


def short_answer_match_score(
    prediction: str,
    reference_answer: str
) -> int:
    """Prediction içinde reference answer'ın açıkça bulunup bulunmadığını ölçer.

    Önce her iki metin normalize edilir.

    Exact match varsa doğrudan 1 döndürülür.

    Exact match yoksa reference answer prediction içinde bağımsız bir cevap
    parçası olarak geçiyorsa 1 döndürülür.

    Örnek:

        reference = "Nigar"
        prediction = "Nigar is the shortest."
        -> 1

        reference = "Nigar"
        prediction = "Aysel is the shortest."
        -> 0
    """

    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference_answer)

    if normalized_prediction == normalized_reference:
        return 1

    if contains_contradictory_answer(
        prediction,
        reference_answer
    ):
        return 0

    if not normalized_reference:
        return 0

    prediction_tokens = normalized_prediction.split()
    reference_tokens = normalized_reference.split()

    reference_length = len(reference_tokens)

    # prediction içinde reference kadar uzunlukta parçaları sırayla kontrol eder.
    # index → parçanın başladığı konum.
    # index + reference_length → parçanın bittiği konum.
    # candidate → o anda kontrol edilen kelime parçası.

    for index in range(
        (len(prediction_tokens) - reference_length) + 1
    ):
        candidate = prediction_tokens[
            index: index + reference_length
        ]

        if candidate == reference_tokens:
            return 1

    return 0


def evaluate_short_answer_prediction(
    record: PredictionRecord
)-> ShortAnswerMatchResult:
    """Tek bir PredictionRecord için short-answer match sonucu üretir."""

    normalized_prediction = normalize_answer(
        record.prediction
    )

    normalized_reference = normalize_answer(
        record.reference_answer
    )

    score = short_answer_match_score(
        prediction=record.prediction,
        reference_answer=record.reference_answer
    )

    return ShortAnswerMatchResult(
        item_id=record.item_id,
        pair_id=record.pair_id,
        language=record.language,
        task=record.task,
        prediction=record.prediction,
        reference_answer=record.reference_answer,
        normalized_prediction=normalized_prediction,
        normalized_reference=normalized_reference,
        short_answer_match=score,
        metadata=dict(record.metadata)
    )


def evaluate_short_answer_matches(
    records: list[PredictionRecord],
) -> list[ShortAnswerMatchResult]:
    """Prediction listesini short-answer evaluator ile değerlendirir."""

    return [
        evaluate_short_answer_prediction(record)
        for record in records
    ]


def calculate_short_answer_accuracy(
    results: list[ShortAnswerMatchResult]
) -> float:
    """Short-answer sonuçlarından accuracy hesaplar."""

    if not results:
        raise ValueError(
            "Cannot calculate accuracy from empty short-answer results."
        )

    correct = sum(
        result.short_answer_match
        for result in results
    )

    accuracy = correct / len(results)

    return accuracy


def summarize_short_answer_matches(
    results: list[ShortAnswerMatchResult]
) -> dict[str, Any]:
    """Short-answer evaluation sonuçlarının temel özetini oluşturur."""

    if not results:
        raise ValueError(
            "Cannot summarize empty short-answer results."
        )

    correct = sum(
        result.short_answer_match
        for result in results
    )

    total = len(results)

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total
    }


SHORT_ANSWER_TASKS = {
    "factual_knowledge",
    "reasoning",
}



def is_short_answer_task(
    task:str
) -> bool:
    """Task'in short-answer evaluator kullanıp kullanmayacağını belirler."""

    return task in SHORT_ANSWER_TASKS


def filter_short_answer_records(
    records: list[PredictionRecord]
) -> list[PredictionRecord]:
    """Yalnızca short-answer task kayıtlarını döndürür."""

    return [
        record
        for record in records
        if is_short_answer_task(record.task)
    ]
